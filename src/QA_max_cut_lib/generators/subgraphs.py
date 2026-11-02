"""Deterministic subgraph builders for local-search RQAOA.

The builders only choose vertex sets.  They do not evaluate QAOA, alter a
reduction state, or touch a global random number generator.  All external
labels are preserved.
"""

from __future__ import annotations

from copy import deepcopy
from numbers import Integral
import numpy as np

from ..core.graph import WeightedGraph

__all__ = [
    "RandomPartitionBuilder",
    "GreedyAbsWeightPartitionBuilder",
    "build_subgraph_builder",
]


def _positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _node_list(graph):
    return sorted(int(node) for node in graph.nodes)


def _split_sorted(nodes, size):
    return [tuple(nodes[start : start + size]) for start in range(0, len(nodes), size)]


def _grow_greedy(graph, start, free, max_size):
    """Grow one block by maximum absolute connection weight.

    The start vertices are inserted in the order supplied.  Remaining free
    vertices are chosen by the largest sum of absolute weights to the block;
    equal gains use the smaller external integer label.
    """
    block = []
    used = set()
    for node in start:
        node = int(node)
        if node in free and node not in used:
            block.append(node)
            used.add(node)
            if len(block) >= max_size:
                return block
    while len(block) < max_size:
        available = [node for node in free if node not in used]
        if not available:
            break
        best = None
        best_gain = None
        for node in available:
            gain = sum(abs(graph.weight(node, other)) for other in block)
            key = (-gain, node)
            if best is None or key < best:
                best = key
                best_gain = gain
        block.append(best[1])
        used.add(best[1])
    return block


class _BasePartitionBuilder:
    """Shared validation and one-task shortcut."""

    type = None
    defaults = {}

    def __init__(self, config=None, *, seed=None):
        config = deepcopy(config or {})
        if set(config) - {"type", "parameters"}:
            raise ValueError("Unknown subgraph-builder field.")
        if config.get("type", self.type) != self.type:
            raise ValueError("The builder type does not match this class.")
        parameters = config.setdefault("parameters", {})
        if set(parameters) - set(self.defaults) - {"boundary"}:
            raise ValueError("Unknown subgraph-builder parameter.")
        values = dict(self.defaults)
        values.update(parameters)
        values["max_size"] = _positive_int(values["max_size"], "max_size", minimum=2)
        values["rounds"] = _positive_int(values["rounds"], "rounds")
        full = values.setdefault("full_graph_when_fits", True)
        if not isinstance(full, bool):
            raise TypeError("full_graph_when_fits must be bool.")
        boundary = values.setdefault("boundary", "induced_no_fields")
        if boundary != "induced_no_fields":
            raise ValueError("Only boundary='induced_no_fields' is implemented.")
        self._config = {"type": self.type, "parameters": values}
        self._rng = np.random.default_rng(seed)

    @property
    def config(self):
        return deepcopy(self._config)

    @property
    def max_size(self):
        return self._config["parameters"]["max_size"]

    def build(self, graph, *, version=0):
        """Return planned vertex blocks for the current graph.

        Parameters
        ----------
        graph : WeightedGraph
            Current working graph.  It is not modified.
        version : int, optional
            Stable parent reduction version used in task identifiers.

        Returns
        -------
        list of dict
            Each item has ``round``, ``block``, ``nodes``, ``task_id`` and
            ``boundary``.  Duplicate sets from different rounds are kept.
        """
        if not isinstance(graph, WeightedGraph):
            raise TypeError("graph must be a WeightedGraph.")
        nodes = _node_list(graph)
        size = self.max_size
        if not nodes:
            return []
        if self._config["parameters"]["full_graph_when_fits"] and len(nodes) <= size:
            return [
                {
                    "round": 0,
                    "block": 0,
                    "nodes": tuple(nodes),
                    "task_id": f"r{int(version)}:p0:b0",
                    "boundary": "induced_no_fields",
                }
            ]
        blocks = []
        rounds = self._config["parameters"]["rounds"]
        for round_index in range(rounds):
            for block_index, block in enumerate(self._round_blocks(graph, nodes, round_index)):
                blocks.append(
                    {
                        "round": round_index,
                        "block": block_index,
                        "nodes": tuple(sorted(int(node) for node in block)),
                        "task_id": f"r{int(version)}:p{round_index}:b{block_index}",
                        "boundary": "induced_no_fields",
                    }
                )
        return blocks

    def _round_blocks(self, graph, nodes, round_index):
        raise NotImplementedError


class RandomPartitionBuilder(_BasePartitionBuilder):
    """Shuffle sorted active labels and split them into consecutive blocks."""

    type = "random_partition"
    defaults = {
        "max_size": 10,
        "rounds": 1,
        "full_graph_when_fits": True,
    }

    def _round_blocks(self, graph, nodes, round_index):
        order = self._rng.permutation(np.asarray(nodes, dtype=np.int64)).tolist()
        return _split_sorted([int(node) for node in order], self.max_size)


class GreedyAbsWeightPartitionBuilder(_BasePartitionBuilder):
    """Grow blocks by largest absolute connection weight."""

    type = "greedy_abs_weight_partition"
    defaults = {
        "max_size": 10,
        "rounds": 1,
        "full_graph_when_fits": True,
    }

    def _round_blocks(self, graph, nodes, round_index):
        free = set(int(node) for node in nodes)
        starts = self._rng.permutation(np.asarray(nodes, dtype=np.int64)).tolist()
        blocks = []
        while free:
            start = None
            for candidate in starts:
                candidate = int(candidate)
                if candidate in free:
                    start = candidate
                    break
            if start is None:
                break
            block = _grow_greedy(graph, [start], free, self.max_size)
            for node in block:
                free.discard(node)
            blocks.append(tuple(sorted(block)))
        return blocks


def build_subgraph_builder(config, *, seed=None):
    """Construct a supported builder from a JSON-compatible section."""
    config = deepcopy(config or {})
    kind = config.get("type", "random_partition")
    if kind == "random_partition":
        return RandomPartitionBuilder(config, seed=seed)
    if kind == "greedy_abs_weight_partition":
        return GreedyAbsWeightPartitionBuilder(config, seed=seed)
    if kind == "single_densest_edge_count_v1":
        from .incremental import DensestEdgeCountBuilder

        return DensestEdgeCountBuilder(config, seed=seed)
    raise ValueError(f"Unknown subgraph builder: {kind!r}.")


def greedy_anchor_block(graph, first_pair, max_size):
    """Build a repair block around one pair using the greedy rule."""
    pair = sorted(int(node) for node in first_pair)
    if len(pair) != 2 or pair[0] == pair[1]:
        raise ValueError("first_pair must contain two distinct nodes.")
    if pair[0] not in graph.nodes or pair[1] not in graph.nodes:
        raise ValueError("first_pair must belong to the graph.")
    size = _positive_int(max_size, "max_size", minimum=2)
    free = set(int(node) for node in graph.nodes)
    return tuple(sorted(_grow_greedy(graph, pair, free, size)))
