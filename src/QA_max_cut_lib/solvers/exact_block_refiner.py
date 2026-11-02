"""Exact conditional block refinement for weighted Max-Cut."""

from __future__ import annotations

from copy import deepcopy
import math
from time import perf_counter

import numpy as np

from ..core.graph import WeightedGraph

__all__ = ["ExactBlockRefiner"]


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _int_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be an integer >= 0.")
    return int(value)


class ExactBlockRefiner:
    """Re-optimize small vertex blocks while keeping the outside spins fixed.

    Parameters
    ----------
    config : dict
        ``parameters`` has ``max_size``, ``batch_size``, ``max_passes``,
        ``max_blocks_total`` and acceptance tolerances.  ``selector`` is
        ``explicit_blocks`` or ``pair_greedy_abs_weight``.

    Notes
    -----
    The conditional Hamiltonian contains boundary fields.  It is not the
    induced zero-field Max-Cut on a block.  Blocks are applied sequentially,
    because one accepted block changes the fields seen by the next block.
    """

    type = "exact_block"

    def __init__(self, config=None):
        config = deepcopy(config or {})
        if set(config) - {"type", "parameters", "selector"} or config.get(
            "type", self.type
        ) != self.type:
            raise ValueError("refiner type must be exact_block.")
        params = config.setdefault("parameters", {})
        allowed = {
            "max_size",
            "batch_size",
            "max_passes",
            "max_blocks_total",
            "improvement_atol",
            "improvement_rtol",
            "max_states_per_block",
        }
        if set(params) - allowed:
            raise ValueError("Unknown exact-block refiner parameter.")
        params.setdefault("max_size", 16)
        params.setdefault("batch_size", 4096)
        params.setdefault("max_passes", 1)
        params.setdefault("max_blocks_total", None)
        params.setdefault("improvement_atol", 1e-10)
        params.setdefault("improvement_rtol", 1e-12)
        params.setdefault("max_states_per_block", 1 << 20)
        if isinstance(params["max_size"], bool) or not isinstance(params["max_size"], int) or not 1 <= params["max_size"] <= 22:
            raise ValueError("max_size must be an integer in 1..22.")
        if isinstance(params["batch_size"], bool) or not isinstance(params["batch_size"], int) or params["batch_size"] < 1:
            raise ValueError("batch_size must be a positive integer.")
        if isinstance(params["max_passes"], bool) or not isinstance(params["max_passes"], int) or params["max_passes"] < 0:
            raise ValueError("max_passes must be a nonnegative integer.")
        if params["max_blocks_total"] is not None:
            _int_nonnegative(params["max_blocks_total"], "max_blocks_total")
        for key in ("improvement_atol", "improvement_rtol"):
            value = _finite(params[key], key)
            if value < 0:
                raise ValueError(f"{key} must be nonnegative.")
        _int_nonnegative(params["max_states_per_block"], "max_states_per_block")
        self._config = {
            "type": self.type,
            "parameters": params,
            "selector": self._validate_selector(config.get("selector")),
        }
        if self._config["selector"]["type"] == "pair_greedy_abs_weight" and params["max_size"] < 2:
            raise ValueError("pair_greedy_abs_weight requires max_size >= 2.")
        self._last_refinement = None

    def _validate_selector(self, selector):
        if not isinstance(selector, dict):
            raise TypeError("selector must be a mapping.")
        kind = selector.get("type")
        if kind not in ("explicit_blocks", "pair_greedy_abs_weight"):
            raise ValueError("Unsupported refiner selector.")
        if set(selector) - {"type", "parameters"}:
            raise ValueError("Unknown selector field.")
        params = selector.setdefault("parameters", {})
        if kind == "explicit_blocks":
            if set(params) - {"blocks"}:
                raise ValueError("Unknown explicit_blocks parameter.")
            blocks = params.get("blocks")
            if not isinstance(blocks, list):
                raise TypeError("explicit_blocks requires a list of blocks.")
            return {"type": kind, "parameters": {"blocks": blocks}}
        if set(params) - {"max_seed_pairs", "ranked_pairs"}:
            raise ValueError("Unknown pair_greedy_abs_weight parameter.")
        params.setdefault("max_seed_pairs", None)
        params.setdefault("ranked_pairs", None)
        if params["max_seed_pairs"] is not None:
            _int_nonnegative(params["max_seed_pairs"], "max_seed_pairs")
        if params["ranked_pairs"] is not None and not isinstance(params["ranked_pairs"], list):
            raise TypeError("ranked_pairs must be a list or null.")
        return {"type": kind, "parameters": params}

    @property
    def config(self):
        """Return a resolved configuration copy."""
        return deepcopy(self._config)

    @property
    def last_refinement(self):
        """Return the last refinement record, or None."""
        return None if self._last_refinement is None else deepcopy(self._last_refinement)

    @property
    def max_size(self):
        """Return the maximum block size."""
        return self._config["parameters"]["max_size"]

    def _explicit_blocks(self, graph):
        params = self._config["selector"]["parameters"]
        blocks = []
        for index, raw in enumerate(params["blocks"]):
            if not isinstance(raw, (list, tuple)) or not raw:
                raise ValueError("Every explicit block must be a non-empty sequence.")
            nodes = tuple(int(node) for node in raw)
            if len(set(nodes)) != len(nodes):
                raise ValueError("An explicit block contains a duplicate node.")
            if len(nodes) > self.max_size:
                raise ValueError("An explicit block exceeds max_size.")
            if not set(nodes) <= set(graph.nodes):
                raise ValueError("An explicit block contains a node outside the graph.")
            blocks.append(
                {
                    "nodes": tuple(sorted(nodes)),
                    "source": "explicit",
                    "source_event_id": None,
                    "priority": index,
                }
            )
        return blocks

    def _candidate_pair_records(self, graph):
        params = self._config["selector"]["parameters"]
        if params["ranked_pairs"] is not None:
            records = []
            for index, item in enumerate(params["ranked_pairs"]):
                if not isinstance(item, dict):
                    raise TypeError("ranked_pairs entries must be mappings.")
                pair = item.get("pair")
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError("ranked_pairs entries need a two-node pair.")
                u, v = int(pair[0]), int(pair[1])
                if u == v or u not in graph.nodes or v not in graph.nodes:
                    raise ValueError("A ranked pair has invalid endpoints.")
                priority = _finite(item.get("priority", index), "priority")
                records.append(
                    {
                        "pair": (min(u, v), max(u, v)),
                        "priority": priority,
                        "source_event_id": item.get("source_event_id"),
                        "source_graph_version": item.get("source_graph_version"),
                        "source_index": index,
                    }
                )
            records.sort(
                key=lambda item: (
                    item["priority"],
                    item["pair"][0],
                    item["pair"][1],
                    item["source_index"],
                )
            )
        else:
            records = [
                {
                    "pair": (int(u), int(v)),
                    "priority": -abs(float(weight)),
                    "source_event_id": None,
                    "source_graph_version": None,
                    "source_index": index,
                }
                for index, (u, v, weight) in enumerate(
                    sorted(
                        graph.edges,
                        key=lambda item: (-abs(float(item[2])), min(item[0], item[1]), max(item[0], item[1])),
                    )
                )
            ]
        limit = params["max_seed_pairs"]
        if limit is not None:
            records = records[:limit]
        return records

    def _grow_pair_block(self, graph, pair, source):
        block = list(pair)
        used = set(block)
        while len(block) < self.max_size:
            candidates = []
            for node in graph.nodes:
                if node in used:
                    continue
                if any(graph.weight(node, other) != 0.0 for other in block):
                    candidates.append(node)
            if not candidates:
                break
            scored = []
            for node in candidates:
                score = math.fsum(abs(graph.weight(node, other)) for other in block)
                scored.append((-score, int(node)))
            scored.sort()
            chosen = scored[0][1]
            block.append(chosen)
            used.add(chosen)
        return tuple(sorted(block)), source

    def _pair_blocks(self, graph):
        blocks = []
        for record in self._candidate_pair_records(graph):
            nodes, source = self._grow_pair_block(graph, record["pair"], record)
            blocks.append(
                {
                    "nodes": nodes,
                    "source": "pair_greedy_abs_weight",
                    "source_event_id": source.get("source_event_id"),
                    "priority": source.get("priority"),
                    "seed_pair": list(source["pair"]),
                }
            )
        return blocks

    def _blocks_for_pass(self, graph):
        if self._config["selector"]["type"] == "explicit_blocks":
            raw = self._explicit_blocks(graph)
        else:
            raw = self._pair_blocks(graph)
        unique = {}
        result = []
        for block in raw:
            key = tuple(block["nodes"])
            if key in unique:
                unique[key]["duplicates"] += 1
                unique[key]["sources"].append(block["source"])
                continue
            item = dict(block)
            item["duplicates"] = 0
            item["sources"] = [block["source"]]
            unique[key] = item
            result.append(item)
        return result

    def _conditional_data(self, graph, block, spins):
        nodes = tuple(sorted(int(node) for node in block))
        chosen = set(nodes)
        boundary_edges = []
        internal_edges = []
        fields = {node: 0.0 for node in nodes}
        touching = 0.0
        for u, v, weight in graph.edges:
            u_inside = u in chosen
            v_inside = v in chosen
            if u_inside and v_inside:
                internal_edges.append((u, v, float(weight)))
                touching += float(weight)
            elif u_inside or v_inside:
                boundary_edges.append((u, v, float(weight)))
                touching += float(weight)
                if u_inside:
                    fields[u] += 0.5 * float(weight) * spins[v]
                else:
                    fields[v] += 0.5 * float(weight) * spins[u]
        outside = [node for node in graph.nodes if node not in chosen]
        outside_cut = graph.subgraph(outside).cut_value(
            {node: spins[node] for node in outside}
        )
        constant = -outside_cut - 0.5 * touching
        return {
            "nodes": nodes,
            "internal_edges": internal_edges,
            "boundary_edges": boundary_edges,
            "fields": fields,
            "constant": constant,
            "touching_weight": touching,
        }

    def _best_block_state(self, data, batch_size):
        nodes = data["nodes"]
        k = len(nodes)
        internal = data["internal_edges"]
        fields = data["fields"]
        total = 1 << k
        index = np.arange(k, dtype=np.uint64)
        best_value = None
        best_state = None
        for start in range(0, total, batch_size):
            states = np.arange(start, min(total, start + batch_size), dtype=np.uint64)
            bits = (states[:, None] >> index[None, :]) & np.uint64(1)
            spins = 1 - 2 * bits.astype(np.float64)
            energy = np.zeros(len(states), dtype=np.float64)
            for u, v, weight in internal:
                iu = nodes.index(u)
                iv = nodes.index(v)
                energy += 0.5 * weight * spins[:, iu] * spins[:, iv]
            for i, node in enumerate(nodes):
                energy += fields[node] * spins[:, i]
            local = int(np.argmin(energy))
            value = float(energy[local])
            if best_value is None or value < best_value:
                best_value = value
                best_state = spins[local].astype(int).tolist()
        return best_value, best_state, total

    def _accepted(self, candidate, current, atol, rtol):
        delta = candidate - current
        threshold = atol + rtol * max(1.0, abs(candidate), abs(current))
        return delta > threshold, delta

    def refine(self, graph, spins, *, context=None):
        """Return an improved independent spin assignment.

        Parameters
        ----------
        graph : WeightedGraph
            Original graph.  It is not modified.
        spins : mapping of int to int
            Complete assignment with one spin for every graph node.
        context : mapping, optional
            Optional shared resource record.  ``refinement_states_total``
            limits enumerated states; it is charged once per full block.

        Returns
        -------
        dict of int to int
            A new complete assignment.  The input mapping is not changed.

        Notes
        -----
        The output is never worse than the input within the configured
        acceptance tolerance.  Each completed block is exact only for the
        outside assignment fixed when that block starts.
        """
        if not isinstance(graph, WeightedGraph):
            raise TypeError("graph must be a WeightedGraph.")
        if not isinstance(spins, dict) or set(spins) != set(graph.nodes):
            raise ValueError("spins must contain exactly every graph node.")
        current = {}
        for node in graph.nodes:
            value = spins[node]
            if isinstance(value, bool) or not isinstance(value, int) or value not in (-1, 1):
                raise ValueError("Every input spin must be integer +1 or -1.")
            current[int(node)] = int(value)
        params = self._config["parameters"]
        initial_cut = graph.cut_value(current)
        final_cut = initial_cut
        max_states = params["max_states_per_block"]
        states_limit = None
        if context is not None:
            states_limit = context.get("refinement_states_total")
        if states_limit is not None:
            _int_nonnegative(states_limit, "refinement_states_total")
        states_used = 0
        blocks_started = 0
        blocks_completed = 0
        blocks_accepted = 0
        passes = 0
        history = []
        status = "completed"
        stop_reason = "completed"
        started = perf_counter()
        while passes < params["max_passes"]:
            accepted_in_pass = False
            blocks = self._blocks_for_pass(graph)
            for block in blocks:
                if params["max_blocks_total"] is not None and blocks_started >= params["max_blocks_total"]:
                    status = "stopped"
                    stop_reason = "max_blocks_total"
                    break
                nodes = block["nodes"]
                k = len(nodes)
                states_needed = 1 << k
                if states_needed > max_states:
                    raise ValueError(
                        f"Block size {k} requires {states_needed} states above max_states_per_block."
                    )
                if states_limit is not None and states_used + states_needed > states_limit:
                    status = "stopped"
                    stop_reason = "refinement_states_total"
                    break
                data = self._conditional_data(graph, nodes, current)
                if not data["internal_edges"] and not data["boundary_edges"]:
                    states_used += 0
                    blocks_started += 1
                    blocks_completed += 1
                    history.append(
                        {
                            "pass": passes,
                            "block_id": blocks_started - 1,
                            "nodes": list(nodes),
                            "source": block["source"],
                            "c_before": float(final_cut),
                            "c_candidate": float(final_cut),
                            "c_after": float(final_cut),
                            "delta": 0.0,
                            "accepted": False,
                            "states_evaluated": 0,
                            "shortcut": "no_touching_edges",
                        }
                    )
                    continue
                blocks_started += 1
                local_best, state, enumerated = self._best_block_state(
                    data, params["batch_size"]
                )
                states_used += enumerated
                candidate = dict(current)
                for index, node in enumerate(nodes):
                    candidate[node] = int(state[index])
                candidate_cut = graph.cut_value(candidate)
                accepted, delta = self._accepted(
                    candidate_cut,
                    final_cut,
                    params["improvement_atol"],
                    params["improvement_rtol"],
                )
                if accepted:
                    current = candidate
                    final_cut = candidate_cut
                    blocks_accepted += 1
                    accepted_in_pass = True
                blocks_completed += 1
                history.append(
                    {
                        "pass": passes,
                        "block_id": blocks_started - 1,
                        "nodes": list(nodes),
                        "source": block["source"],
                        "seed_pair": block.get("seed_pair"),
                        "c_before": float(final_cut - delta if accepted else final_cut),
                        "c_candidate": float(candidate_cut),
                        "c_after": float(final_cut),
                        "delta": float(delta),
                        "accepted": bool(accepted),
                        "states_evaluated": int(enumerated),
                        "fields": {int(node): float(value) for node, value in data["fields"].items()},
                        "constant": float(data["constant"]),
                    }
                )
            passes += 1
            if status == "stopped":
                break
            if not accepted_in_pass:
                break
        if status != "stopped" and passes >= params["max_passes"] and params["max_passes"] > 0:
            # A finite pass limit is reported when another pass could exist.
            status = "stopped" if any(item["accepted"] for item in history) else "completed"
            stop_reason = "max_passes" if status == "stopped" else "no_improvement"
        record = {
            "status": status,
            "stop_reason": stop_reason,
            "selector": deepcopy(self._config["selector"]["type"]),
            "initial_cut": float(initial_cut),
            "final_cut": float(final_cut),
            "gain": float(final_cut - initial_cut),
            "passes": int(passes),
            "blocks_started": int(blocks_started),
            "blocks_completed": int(blocks_completed),
            "blocks_accepted": int(blocks_accepted),
            "states_evaluated": int(states_used),
            "refinement_seconds": perf_counter() - started,
            "history": history,
        }
        self._last_refinement = {"solution": deepcopy(current), **record}
        return deepcopy(current)


def main(argv=None):
    """Run a small explicit-block refinement example."""
    from ..core.graph import WeightedGraph

    graph = WeightedGraph(
        [(10, 30, 1), (10, 80, 4), (30, 80, 3)],
        nodes=[10, 30, 80],
    )
    refiner = ExactBlockRefiner(
        {
            "type": "exact_block",
            "parameters": {"max_size": 8},
            "selector": {
                "type": "explicit_blocks",
                "parameters": {"blocks": [[10, 30]]},
            },
        }
    )
    print(refiner.refine(graph, {10: 1, 30: 1, 80: 1}))
    print(refiner.last_refinement)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
