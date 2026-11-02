"""Versioned subgraph recipes for incremental LS-RQAOA.

The builders are intentionally pure: they read a graph snapshot and return
membership records.  They never execute a solver or mutate the graph.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from ..core.graph import WeightedGraph

__all__ = [
    "DensestEdgeCountBuilder",
    "build_incremental_initial_builder",
    "survivor_neighbor_cover",
]


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer.")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return value


def _label_list(graph):
    return sorted(int(node) for node in graph.nodes)


def _degree(graph, node):
    return sum(1 for u, v, weight in graph.edges if float(weight) != 0.0 and (u == node or v == node))


def _degree_map(graph):
    return {int(node): _degree(graph, int(node)) for node in graph.nodes}


class DensestEdgeCountBuilder:
    """Find one node set by maximizing the count of internal nonzero edges.

    This is the ``single_densest_edge_count_v1`` recipe.  The objective is
    topological; edge signs and magnitudes are deliberately ignored.
    ``m(S)`` is not a cut and the selected set is not claimed optimal.
    """

    type = "single_densest_edge_count_v1"
    defaults = {
        "max_size": 6,
        "rounds": 1,
        "full_graph_when_fits": True,
        "n_restarts": 20,
        "local_iter": 500,
        "candidate_pool": 64,
        "boundary": "induced_no_fields",
    }

    def __init__(self, config=None, *, seed=None):
        config = deepcopy(config or {})
        if set(config) - {"type", "parameters"}:
            raise ValueError("Unknown densest-builder field.")
        if config.get("type", self.type) != self.type:
            raise ValueError("Densest builder type mismatch.")
        parameters = config.setdefault("parameters", {})
        if set(parameters) - set(self.defaults):
            raise ValueError("Unknown densest-builder parameter.")
        values = dict(self.defaults)
        values.update(parameters)
        values["max_size"] = _integer(values["max_size"], "max_size", 2)
        values["rounds"] = _integer(values["rounds"], "rounds", 1)
        values["n_restarts"] = _integer(values["n_restarts"], "n_restarts", 1)
        values["local_iter"] = _integer(values["local_iter"], "local_iter", 0)
        values["candidate_pool"] = _integer(values["candidate_pool"], "candidate_pool", 1)
        if values["boundary"] != "induced_no_fields":
            raise ValueError("Only induced_no_fields is supported.")
        if type(values["full_graph_when_fits"]) is not bool:
            raise TypeError("full_graph_when_fits must be boolean.")
        self._config = {"type": self.type, "parameters": values}
        self._rng = np.random.default_rng(seed)

    @property
    def config(self):
        return deepcopy(self._config)

    @property
    def max_size(self):
        return self._config["parameters"]["max_size"]

    def _internal_edges(self, graph, members):
        chosen = set(int(node) for node in members)
        return sum(
            1
            for u, v, weight in graph.edges
            if float(weight) != 0.0 and u in chosen and v in chosen
        )

    def _grow(self, graph, start, free, k, rng):
        members = [int(start)]
        used = {int(start)}
        while len(members) < k:
            candidates = [node for node in free if node not in used]
            if not candidates:
                break
            candidates = sorted(candidates)
            priorities = rng.random(len(candidates))
            degree = _degree_map(graph)
            scored = []
            for index, node in enumerate(candidates):
                gain = sum(
                    1
                    for member in members
                    if float(graph.weight(node, member)) != 0.0
                )
                scored.append((-gain, -degree[node], -float(priorities[index]), int(node)))
            selected = min(scored)[3]
            members.append(selected)
            used.add(selected)
        return members

    def _swap_search(self, graph, members, free, k, rng, local_iter):
        members = list(members)
        inside = set(members)
        outside = set(free) - inside
        degree = _degree_map(graph)
        for _ in range(local_iter):
            gains = {}
            for node in list(inside) + list(outside):
                gains[node] = sum(
                    1
                    for other in inside
                    if other != node and float(graph.weight(node, other)) != 0.0
                )
            remove = sorted(inside, key=lambda node: (gains[node], node))[: self._config["parameters"]["candidate_pool"]]
            add = sorted(outside, key=lambda node: (-gains[node], node))[: self._config["parameters"]["candidate_pool"]]
            best = None
            for u in remove:
                for v in add:
                    delta = gains[v] - (1 if float(graph.weight(u, v)) != 0.0 else 0) - gains[u]
                    if delta <= 0:
                        continue
                    candidate = (-delta, u, v)
                    if best is None or candidate < best[0]:
                        best = (candidate, u, v)
            if best is None:
                break
            _, u, v = best
            inside.remove(u)
            inside.add(v)
            outside.remove(v)
            outside.add(u)
            members = [v if node == u else node for node in members]
        return members

    def _best_for_start(self, graph, start, free, k, rng):
        members = self._grow(graph, start, free, k, rng)
        members = self._swap_search(
            graph,
            members,
            free,
            k,
            rng,
            self._config["parameters"]["local_iter"],
        )
        return tuple(sorted(int(node) for node in members))

    def build(self, graph, *, version=0):
        if not isinstance(graph, WeightedGraph):
            raise TypeError("graph must be a WeightedGraph.")
        nodes = _label_list(graph)
        if not nodes:
            return []
        k = min(self.max_size, len(nodes))
        if self._config["parameters"]["full_graph_when_fits"] and k == len(nodes):
            blocks = [tuple(nodes)]
        else:
            degree = _degree_map(graph)
            starts = sorted(nodes, key=lambda node: (-degree[node], node))
            starts = list(starts[: min(self._config["parameters"]["n_restarts"], len(nodes))])
            while len(starts) < self._config["parameters"]["n_restarts"]:
                starts.append(int(self._rng.choice(nodes)))
            free = set(nodes)
            best_members = None
            best_key = None
            for start in starts:
                members = self._best_for_start(graph, start, free, k, self._rng)
                key = (-self._internal_edges(graph, members), members)
                if best_key is None or key < best_key:
                    best_key = key
                    best_members = members
            blocks = [best_members]
        tasks = []
        for round_index in range(self._config["parameters"]["rounds"]):
            for block_index, block in enumerate(blocks):
                tasks.append(
                    {
                        "round": round_index,
                        "block": block_index,
                        "nodes": tuple(int(node) for node in block),
                        "task_id": f"r{int(version)}:densest:{round_index}:{block_index}",
                        "boundary": "induced_no_fields",
                        "builder_type": self.type,
                        "internal_edge_count": self._internal_edges(graph, block),
                        "density": (
                            self._internal_edges(graph, block)
                            / (len(block) * (len(block) - 1) / 2)
                            if len(block) > 1
                            else 0.0
                        ),
                    }
                )
        return tasks


def build_incremental_initial_builder(config, *, seed=None):
    """Build either a legacy partition or the new densest builder."""
    config = deepcopy(config or {})
    kind = config.get("type", "greedy_abs_weight_partition")
    if kind == DensestEdgeCountBuilder.type:
        return DensestEdgeCountBuilder(config, seed=seed)
    from .subgraphs import build_subgraph_builder

    return build_subgraph_builder(config, seed=seed)


def _connectivity(graph, node, members, growth_metric):
    if growth_metric == "count":
        return sum(
            1
            for member in members
            if float(graph.weight(node, member)) != 0.0
        )
    if growth_metric == "abs_weight":
        return sum(abs(float(graph.weight(node, member))) for member in members)
    raise ValueError("growth_metric must be count or abs_weight.")


def survivor_neighbor_cover(
    graph,
    center,
    *,
    max_size,
    growth_metric="count",
    rounds=1,
    version=0,
    full_graph_when_fits=True,
):
    """Cover direct neighbors of one retained center with overlapping blocks.

    Each returned block contains ``center`` and is at most ``max_size``.
    Every direct neighbor is covered.  No arbitrary disconnected filler is
    inserted; the last block may use a 2-hop neighbor only when it has a
    positive current connectivity to the block.
    """
    if not isinstance(graph, WeightedGraph):
        raise TypeError("graph must be a WeightedGraph.")
    labels = _label_list(graph)
    center = int(center)
    if center not in labels:
        raise ValueError("center is not active.")
    k = _integer(max_size, "max_size", 2)
    rounds = _integer(rounds, "rounds", 1)
    if growth_metric not in ("count", "abs_weight"):
        raise ValueError("growth_metric must be count or abs_weight.")
    if full_graph_when_fits and len(labels) <= k:
        return [
            {
                "round": 0,
                "block": 0,
                "nodes": tuple(labels),
                "task_id": f"r{int(version)}:cover:{center}:full",
                "boundary": "induced_no_fields",
                "builder_type": "survivor_neighbor_cover_v1",
                "center": center,
            }
        ]
    direct = sorted(
        int(v if u == center else u)
        for u, v, weight in graph.edges
        if float(weight) != 0.0 and (u == center or v == center)
    )
    direct = sorted(set(direct))
    if not direct:
        blocks = [tuple([center])]
    else:
        uncovered = set(direct)
        blocks = []
        while uncovered:
            seed = max(
                uncovered,
                key=lambda node: (_degree(graph, node), -int(node)),
            )
            members = [center, int(seed)]
            uncovered.discard(seed)
            while len(members) < k:
                candidates = sorted(uncovered - set(members))
                if not candidates:
                    break
                chosen = max(
                    candidates,
                    key=lambda node: (
                        _connectivity(graph, node, members, growth_metric),
                        -int(node),
                    ),
                )
                if _connectivity(graph, chosen, members, growth_metric) <= 0:
                    break
                members.append(chosen)
                uncovered.discard(chosen)
            if len(members) < k:
                second_hop = set()
                for member in members:
                    for u, v, weight in graph.edges:
                        if float(weight) == 0.0:
                            continue
                        if u == member:
                            second_hop.add(v)
                        elif v == member:
                            second_hop.add(u)
                candidates = sorted(
                    (second_hop - set(members)) - set(uncovered)
                )
                while len(members) < k and candidates:
                    chosen = max(
                        candidates,
                        key=lambda node: (
                            _connectivity(graph, node, members, growth_metric),
                            -int(node),
                        ),
                    )
                    if _connectivity(graph, chosen, members, growth_metric) <= 0:
                        break
                    members.append(chosen)
                    candidates.remove(chosen)
            blocks.append(tuple(sorted(int(node) for node in members)))

    tasks = []
    for round_index in range(rounds):
        for block_index, block in enumerate(blocks):
            tasks.append(
                {
                    "round": round_index,
                    "block": block_index,
                    "nodes": tuple(block),
                    "task_id": f"r{int(version)}:cover:{center}:{round_index}:{block_index}",
                    "boundary": "induced_no_fields",
                    "builder_type": "survivor_neighbor_cover_v1",
                    "center": center,
                }
            )
    return tasks
