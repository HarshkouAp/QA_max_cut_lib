"""Versioned graph recipes for reproducibility and benchmark construction.

The old ``schema_version=1`` producer stays in :mod:`graphs`.  This module
adds ``schema_version=2`` families and records the resolved recipe instead
of pretending that a recipe is a canonical initial state.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import itertools
import math
from numbers import Integral

import numpy as np

from ..core.graph import WeightedGraph

__all__ = ["build_extended_recipe", "resolve_recipe"]


def _int(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite.")
    return float(value)


def _labels(parameters):
    n = _int(parameters.get("n_vertices"), "n_vertices", 0)
    labels = parameters.get("nodes", list(range(n)))
    if not isinstance(labels, (list, tuple)) or len(labels) != n:
        raise ValueError("nodes must have exactly n_vertices entries.")
    if len(set(labels)) != len(labels):
        raise ValueError("nodes must be unique.")
    if any(isinstance(label, bool) or not isinstance(label, Integral) for label in labels):
        raise ValueError("nodes must be integer labels.")
    return n, [int(label) for label in labels]


def _assign_weights(graph, pairs, weight_config, rng):
    config = deepcopy(weight_config or {"type": "constant", "parameters": {"value": 1.0}})
    kind = config.get("type")
    if kind == "constant":
        value = _finite(config.get("parameters", {}).get("value", 1.0), "weight.value")
        for u, v in pairs:
            graph.set_weight(u, v, value)
    elif kind == "uniform":
        parameters = config.get("parameters", {})
        low = _finite(parameters.get("low", 0.0), "weight.low")
        high = _finite(parameters.get("high", 1.0), "weight.high")
        if not low < high:
            raise ValueError("uniform weight bounds need low < high.")
        for u, v in pairs:
            graph.set_weight(u, v, float(rng.uniform(low, high)))
    elif kind == "signed_discrete":
        values = config.get("parameters", {}).get("values")
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError("signed_discrete needs values.")
        values = [_finite(value, "weight value") for value in values]
        for u, v in pairs:
            graph.set_weight(u, v, float(rng.choice(values)))
    else:
        raise ValueError("unknown weight recipe.")
    return graph


def _resolved(recipe, graph, *, topology_seed, weight_seed, extra=None):
    resolved = deepcopy(recipe)
    resolved["schema_version"] = 2
    resolved["resolved"] = {
        "topology_seed": int(topology_seed) if topology_seed is not None else None,
        "weight_seed": int(weight_seed) if weight_seed is not None else None,
        "topology_hash": _topology_hash(graph),
        "weighted_hash": graph.snapshot()["fingerprint"],
        "labels": list(graph.nodes),
        "extras": deepcopy(extra or {}),
        "limits": deepcopy(recipe.get("limits", {})),
    }
    graph._recipe = resolved
    return graph


def _topology_hash(graph):
    canonical = {
        "nodes": sorted(graph.nodes),
        "edges": sorted((u, v) for u, v, _ in graph.edges),
    }
    return hashlib.sha256(repr(canonical).encode("utf-8")).hexdigest()


def _check_limits(graph, limits, n):
    if not limits:
        return
    max_nodes = limits.get("max_vertices")
    max_edges = limits.get("max_edges")
    if max_nodes is not None and n > _int(max_nodes, "max_vertices", 0):
        raise ValueError("n_vertices exceeds limits.max_vertices.")
    if max_edges is not None and len(graph.edges) > _int(max_edges, "max_edges", 0):
        raise ValueError("edge count exceeds limits.max_edges.")


def _add_weights_after_topology(graph, weight_config, topology_seed, extra=None, recipe=None):
    weight_seed = _weight_seed(weight_config, topology_seed)
    rng = np.random.default_rng(weight_seed)
    _assign_weights(graph, [(u, v) for u, v, _ in graph.edges], weight_config, rng)
    return _resolved(recipe or {}, graph, topology_seed=topology_seed, weight_seed=weight_seed, extra=extra)


def _weight_seed(weight_config, default):
    seed = weight_config.get("seed", default) if weight_config else default
    return None if seed is None else int(seed)


def build_extended_recipe(recipe):
    recipe = deepcopy(recipe)
    if recipe.get("schema_version") != 2:
        raise ValueError("extended recipes require schema_version=2.")
    if set(recipe) - {"schema_version", "type", "parameters", "seed", "weights", "limits"}:
        raise ValueError("Unknown extended-recipe field.")
    kind = recipe.get("type")
    parameters = recipe.setdefault("parameters", {})
    seed = recipe.get("seed")
    weight_config = recipe.get("weights", {"type": "constant", "parameters": {"value": 1.0}})
    limits = recipe.get("limits", {})
    if kind in ("complete", "gnm", "erdos_renyi"):
        # Reuse the v1 producer without changing its old semantics.
        from .graphs import GraphFactory

        old = deepcopy(recipe)
        old["schema_version"] = 1
        old.pop("limits", None)
        graph = GraphFactory.from_recipe(old)
        _check_limits(graph, limits, len(graph))
        return _resolved(recipe, graph, topology_seed=seed, weight_seed=_weight_seed(weight_config, seed))
    if kind == "wishart_projected_v1":
        from .structured import build_wishart_projected, build_wishart_from_factor
        n = _int(parameters.get("n_vertices"), "n_vertices", 2)
        labels = parameters.get("nodes", list(range(n)))
        target = parameters.get("target")
        factor = parameters.get("factor")
        if factor is not None:
            graph, record = build_wishart_from_factor(
                np.asarray(factor, dtype=float), target, labels=labels
            )
        else:
            graph, record = build_wishart_projected(
                n, _int(parameters.get("n_factors"), "n_factors", 1),
                target=target, labels=labels, seed=seed,
            )
        graph._recipe = {"schema_version": 2, "type": kind, "resolved": record}
        return graph
    if kind == "hebbian_patterns_v1":
        from .structured import build_hebbian_patterns
        patterns = parameters.get("patterns")
        if patterns is None:
            raise ValueError("hebbian_patterns needs explicit patterns.")
        graph, record = build_hebbian_patterns(
            patterns,
            coefficients=parameters.get("coefficients"),
            labels=parameters.get("nodes"),
            profile=parameters.get("profile", "canonical_v1"),
        )
        graph._recipe = {"schema_version": 2, "type": kind, "resolved": record}
        return graph
    n, labels = _labels(parameters)
    if kind == "gnm_density_v1":
        if n < 0:
            raise ValueError("n_vertices must be nonnegative.")
        density = _finite(parameters.get("density"), "density")
        if not 0.0 <= density <= 1.0:
            raise ValueError("density must be in [0,1].")
        total = n * (n - 1) // 2
        m = int(math.floor(density * total))
        graph = _gnm(n, labels, m, seed, limits)
        return _add_weights_after_topology(graph, weight_config, seed, {"density": density, "m_requested": m}, recipe)
    if kind == "connected_path_fill_v1":
        if n < 1:
            raise ValueError("connected_path_fill needs n >= 1.")
        m = _int(parameters.get("n_edges"), "n_edges", 0)
        if m < n - 1 or m > n * (n - 1) // 2:
            raise ValueError("connected_path_fill needs n-1 <= n_edges <= n(n-1)/2.")
        graph = WeightedGraph(nodes=labels)
        rng = np.random.default_rng(seed)
        path_pairs = [(labels[i], labels[i + 1]) for i in range(n - 1)]
        for pair in path_pairs:
            graph.set_weight(*pair, 1.0)
        remaining = [
            (u, v) for u, v in itertools.combinations(sorted(labels), 2)
            if graph.weight(u, v) == 0.0
        ]
        need = m - (n - 1)
        if need:
            for index in rng.choice(len(remaining), need, replace=False):
                graph.set_weight(*remaining[int(index)], 1.0)
        _check_limits(graph, limits, n)
        return _add_weights_after_topology(graph, weight_config, seed, {"m_requested": m, "path_pairs": len(path_pairs)}, recipe)
    if kind == "regular_v1":
        degree = _int(parameters.get("degree"), "degree", 0)
        if n * degree % 2 or degree >= n:
            raise ValueError("regular needs n*d even and d < n.")
        graph = _regular_graph(n, labels, degree, seed)
        return _add_weights_after_topology(graph, weight_config, seed, {"degree": degree}, recipe)
    if kind == "barabasi_albert_v1":
        m = _int(parameters.get("attachment"), "attachment", 1)
        if m >= n:
            raise ValueError("attachment must be < n.")
        graph = _barabasi(n, labels, m, seed)
        return _add_weights_after_topology(graph, weight_config, seed, {"attachment": m}, recipe)
    if kind == "watts_strogatz_v1":
        k = _int(parameters.get("neighbors"), "neighbors", 2)
        p = _finite(parameters.get("rewire_probability"), "rewire_probability")
        if k % 2 or k >= n or not 0 <= p <= 1:
            raise ValueError("WS needs even k<n and 0<=p<=1.")
        graph = _watts_strogatz(n, labels, k, p, seed)
        return _add_weights_after_topology(graph, weight_config, seed, {"neighbors": k, "rewire_probability": p}, recipe)
    if kind == "sbm_v1":
        sizes = parameters.get("block_sizes")
        matrix = parameters.get("probability_matrix")
        if not isinstance(sizes, list) or not sizes or any(
            isinstance(size, bool) or not isinstance(size, Integral) or int(size) < 1 for size in sizes
        ):
            raise ValueError("block_sizes must be positive integers.")
        if sum(sizes) != n:
            raise ValueError("block_sizes must sum to n_vertices.")
        if not isinstance(matrix, list) or len(matrix) != len(sizes) or any(len(row) != len(sizes) for row in matrix):
            raise ValueError("probability_matrix must be square with one row per block.")
        probabilities = [[_finite(value, "block probability") for value in row] for row in matrix]
        if any(value < 0 or value > 1 for row in probabilities for value in row):
            raise ValueError("probabilities must be in [0,1].")
        graph = _sbm(n, labels, sizes, probabilities, seed)
        return _add_weights_after_topology(graph, weight_config, seed, {"block_sizes": list(sizes), "probability_matrix": probabilities}, recipe)
    if kind == "geometric_v1":
        dimension = _int(parameters.get("dimension", 2), "dimension", 1)
        radius = _finite(parameters.get("radius"), "radius")
        if radius <= 0:
            raise ValueError("radius must be positive.")
        graph, positions = _geometric(n, labels, dimension, radius, seed)
        return _add_weights_after_topology(graph, weight_config, seed, {"dimension": dimension, "radius": radius}, recipe)
    if kind == "periodic_grid_v1":
        dimensions = parameters.get("dimensions")
        periodic = parameters.get("periodic", [True] * len(dimensions or []))
        if not isinstance(dimensions, list) or not dimensions or any(
            isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1 for value in dimensions
        ):
            raise ValueError("dimensions must be positive integers.")
        if len(periodic) != len(dimensions) or any(type(value) is not bool for value in periodic):
            raise ValueError("periodic must be a boolean list matching dimensions.")
        n_grid = int(np.prod(dimensions))
        if n_grid != n:
            raise ValueError("n_vertices must equal product(dimensions).")
        graph, coordinates = _periodic_grid(labels, dimensions, periodic, seed)
        _check_limits(graph, limits, n_grid)
        return _add_weights_after_topology(
            graph, weight_config, seed,
            {"dimensions": list(dimensions), "periodic": list(periodic), "coordinates": coordinates},
            recipe,
        )
    raise ValueError(f"unknown extended graph family: {kind!r}")


def _gnm(n, labels, m, seed, limits):
    if m < 0 or m > n * (n - 1) // 2:
        raise ValueError("invalid n_edges.")
    if limits.get("max_edges") is not None and m > _int(limits["max_edges"], "max_edges", 0):
        raise ValueError("n_edges exceeds limits.max_edges.")
    graph = WeightedGraph(nodes=labels)
    rng = np.random.default_rng(seed)
    pairs = list(itertools.combinations(sorted(labels), 2))
    for index in rng.choice(len(pairs), m, replace=False) if m else []:
        graph.set_weight(*pairs[int(index)], 1.0)
    return graph


def _regular_graph(n, labels, degree, seed):
    if n == 0:
        return WeightedGraph(nodes=[])
    if degree % 2 and n % 2:
        raise ValueError("odd degree requires even n.")
    graph = WeightedGraph(nodes=labels)
    for i, u in enumerate(labels):
        for offset in range(1, degree // 2 + 1):
            v = labels[(i + offset) % n]
            if u != v:
                graph.set_weight(u, v, 1.0)
        if degree % 2:
            v = labels[(i + n // 2) % n]
            if u != v:
                graph.set_weight(u, v, 1.0)
    return graph

def _barabasi(n, labels, m, seed):
    rng = np.random.default_rng(seed)
    graph = WeightedGraph(nodes=labels)
    if n <= m:
        raise ValueError("BA needs n > m.")
    initial = labels[:m + 1]
    for u, v in itertools.combinations(initial, 2):
        graph.set_weight(u, v, 1.0)
    targets = list(initial)
    for node in labels[m + 1:]:
        chosen = set()
        while len(chosen) < m:
            chosen.add(int(rng.choice(targets)))
        for other in chosen:
            graph.set_weight(node, other, 1.0)
        targets.extend([node] * m)
    return graph


def _watts_strogatz(n, labels, k, probability, seed):
    rng = np.random.default_rng(seed)
    graph = WeightedGraph(nodes=labels)
    for i, u in enumerate(labels):
        for offset in range(1, k // 2 + 1):
            v = labels[(i + offset) % n]
            graph.set_weight(u, v, 1.0)
    for i, u in enumerate(labels):
        for offset in range(1, k // 2 + 1):
            v = labels[(i + offset) % n]
            if rng.random() < probability:
                candidates = [
                    w for w in labels
                    if w != u and graph.weight(u, w) == 0.0
                ]
                if candidates:
                    graph.set_weight(u, int(rng.choice(candidates)), 1.0)
    return graph


def _sbm(n, labels, sizes, probabilities, seed):
    rng = np.random.default_rng(seed)
    graph = WeightedGraph(nodes=labels)
    blocks = []
    position = {int(label): index for index, label in enumerate(labels)}
    start = 0
    for size in sizes:
        blocks.append(labels[start:start + size])
        start += size
    for i, block_a in enumerate(blocks):
        for j, block_b in enumerate(blocks):
            if j < i:
                continue
            for u in block_a:
                for v in block_b:
                    if u == v or (j == i and position[v] <= position[u]):
                        continue
                    if rng.random() < probabilities[i][j]:
                        graph.set_weight(u, v, 1.0)
    return graph


def _geometric(n, labels, dimension, radius, seed):
    rng = np.random.default_rng(seed)
    positions = rng.random((n, dimension))
    graph = WeightedGraph(nodes=labels)
    for i in range(n):
        for j in range(i + 1, n):
            if float(np.linalg.norm(positions[i] - positions[j])) <= radius:
                graph.set_weight(labels[i], labels[j], 1.0)
    return graph, [[float(value) for value in row] for row in positions]


def _periodic_grid(labels, dimensions, periodic, seed):
    coordinates = list(itertools.product(*[range(size) for size in dimensions]))
    index = {coordinate: labels[i] for i, coordinate in enumerate(coordinates)}
    graph = WeightedGraph(nodes=labels)
    for coordinate in coordinates:
        for axis, size in enumerate(dimensions):
            if size == 1:
                continue
            step = [0] * len(dimensions)
            step[axis] = 1
            neighbour = list(coordinate)
            neighbour[axis] += 1
            if neighbour[axis] >= size:
                if not periodic[axis]:
                    continue
                neighbour[axis] = 0
            pair = (index[tuple(coordinate)], index[tuple(neighbour)])
            if pair[0] != pair[1]:
                graph.set_weight(*pair, 1.0)
    return graph, [list(coordinate) for coordinate in coordinates]


def resolve_recipe(recipe):
    """Return a resolved schema-2 recipe without forcing a caller to solve it."""
    graph = build_extended_recipe(recipe)
    return deepcopy(graph.snapshot().get("recipe"))
