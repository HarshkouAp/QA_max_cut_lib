"""Historical benchmark resolvers and small materialization helpers.

Resolver functions are pure plans.  They never create a graph.  This
keeps a 297-ID catalogue cheap to inspect and lets a caller apply its own
generation limits.
"""

from __future__ import annotations

from copy import deepcopy
import itertools
import math
from numbers import Integral

import numpy as np

from ..core.graph import WeightedGraph

__all__ = [
    "BENCH16_SPECS",
    "BENCH297_FAMILIES",
    "resolve_bench16",
    "resolve_bench297",
    "materialize_bench16",
    "materialize_bench297",
    "bench297_ids",
]


BENCH16_SPECS = (
    {"id": 0, "n": 120, "density": 0.2, "weights": "random"},
    {"id": 1, "n": 120, "density": 0.4, "weights": "random"},
    {"id": 2, "n": 120, "density": 0.6, "weights": "random"},
    {"id": 3, "n": 120, "density": 0.8, "weights": "random"},
    {"id": 4, "n": 120, "density": 1.0, "weights": "random"},
    {"id": 5, "n": 120, "density": 0.2, "weights": "pos_neg"},
    {"id": 6, "n": 120, "density": 0.4, "weights": "pos_neg"},
    {"id": 7, "n": 120, "density": 0.6, "weights": "pos_neg"},
    {"id": 8, "n": 120, "density": 0.8, "weights": "pos_neg"},
    {"id": 9, "n": 120, "density": 1.0, "weights": "pos_neg"},
    {"id": 10, "n": 250, "density": 0.1, "weights": "random"},
    {"id": 11, "n": 500, "density": 0.05, "weights": "random"},
    {"id": 12, "n": 1000, "density": 0.025, "weights": "random"},
    {"id": 13, "dimensions": (10, 12), "weights": "random"},
    {"id": 14, "dimensions": (5, 5, 5), "weights": "random"},
    {"id": 15, "dimensions": (4, 3, 3, 3), "weights": "random"},
)

BENCH297_FAMILIES = (
    "er_sparse",
    "er_dense",
    "random_regular",
    "barabasi_albert",
    "watts_strogatz",
    "sbm",
    "planted_cut",
    "random_geometric",
    "toroidal_grid",
    "adversarial_mixed",
    "complete",
)
_WEIGHTS = ("unweighted", "positive_continuous", "signed_discrete")
_SIZES = (100, 100, 100, 200, 200, 200, 300, 300, 300)
_OFFSETS = (101, 202, 303)


def resolve_bench16(benchmark_id):
    if isinstance(benchmark_id, bool) or not isinstance(benchmark_id, Integral):
        raise ValueError("benchmark_id must be an integer.")
    benchmark_id = int(benchmark_id)
    if not 0 <= benchmark_id <= 15:
        raise ValueError("BenchGraph16 id must be in [0,15].")
    return deepcopy(BENCH16_SPECS[benchmark_id])


def resolve_bench297(benchmark_id, base_seed=42):
    if isinstance(benchmark_id, bool) or not isinstance(benchmark_id, Integral):
        raise ValueError("benchmark_id must be an integer.")
    benchmark_id = int(benchmark_id)
    if not 0 <= benchmark_id <= 296:
        raise ValueError("benchmark297 id must be in [0,296].")
    base_seed = int(base_seed)
    q = benchmark_id
    topology_group = q // 3
    weight_index = q % 3
    family_index = topology_group // 9
    slot = topology_group % 9
    n = _SIZES[slot]
    topology_seed = base_seed + 10000 * topology_group
    weight_seed = topology_seed + _OFFSETS[weight_index]
    return {
        "benchmark_id": benchmark_id,
        "family_index": family_index,
        "family": BENCH297_FAMILIES[family_index],
        "slot": slot,
        "n_vertices": n,
        "repetition": slot % 3,
        "weight_index": weight_index,
        "weights": _WEIGHTS[weight_index],
        "topology_seed": topology_seed,
        "weight_seed": weight_seed,
        "base_seed": base_seed,
        "topology_group": topology_group,
        "profile": "benchmark297_legacy_v1",
    }


def bench297_ids():
    """Return all 297 decoded plan records without generating graphs."""
    return [resolve_bench297(index) for index in range(297)]


def _assign_bench_weights(graph, weight_seed, mode):
    rng = np.random.default_rng(weight_seed)
    pairs = list(graph.edges)
    if mode == "unweighted":
        values = [1.0] * len(pairs)
    elif mode == "positive_continuous":
        values = [float(rng.uniform(0.1, 1.0)) for _ in pairs]
    elif mode == "signed_discrete":
        values = [float(rng.choice([-5, -3, -1, 1, 3, 5])) for _ in pairs]
    else:
        raise ValueError("unknown benchmark weight mode.")
    for (u, v, _), value in zip(pairs, values):
        graph.set_weight(u, v, value)
    return graph


def _bernoulli_graph(labels, probability, rng):
    graph = WeightedGraph(nodes=labels)
    for u, v in itertools.combinations(labels, 2):
        if rng.random() < probability:
            graph.set_weight(u, v, 1.0)
    return graph


def _connected_repair(graph, labels):
    """Add a deterministic component chain, recording no hidden weights."""
    if len(graph) <= 1:
        return []
    adjacency = {node: [] for node in labels}
    for u, v, _ in graph.edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    seen = set()
    components = []
    for start in labels:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for other in adjacency[node]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        components.append(component)
    repairs = []
    for first, second in zip(components, components[1:]):
        repairs.append((first[0], second[0]))
        graph.set_weight(first[0], second[0], 1.0)
    return repairs


def _regular_edges(labels, degree, rng):
    n = len(labels)
    if degree % 2 and n % 2:
        raise ValueError("odd degree requires even n.")
    edges = []
    for i, u in enumerate(labels):
        for offset in range(1, degree // 2 + 1):
            v = labels[(i + offset) % n]
            if u != v:
                edges.append((min(u, v), max(u, v)))
        if degree % 2:
            v = labels[(i + n // 2) % n]
            if u != v:
                edges.append((min(u, v), max(u, v)))
    return sorted(set(edges))

def _topology(family, n, seed, labels):
    rng = np.random.default_rng(seed)
    if family in ("er_sparse", "er_dense"):
        low, high = (6 / n, 12 / n) if family == "er_sparse" else (0.15, 0.45)
        p = float(rng.uniform(low, high))
        return _bernoulli_graph(labels, p, rng), {"p": p}
    if family == "random_regular":
        degree = int(rng.choice([3, 4, 5, 8, 10, 12]))
        if degree >= n:
            degree = max(2, n // 2)
        if n * degree % 2:
            degree += 1
        graph = WeightedGraph(nodes=labels)
        for u, v in _regular_edges(labels, degree, rng):
            graph.set_weight(u, v, 1.0)
        return graph, {"degree": degree}
    if family == "barabasi_albert":
        m = int(rng.choice([2, 3, 4, 5, 6]))
        graph = WeightedGraph(nodes=labels)
        initial = labels[: m + 1]
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
        return graph, {"attachment": m}
    if family == "watts_strogatz":
        k = int(rng.choice([4, 6, 8, 10]))
        p = float(rng.uniform(0.05, 0.35))
        graph = WeightedGraph(nodes=labels)
        for i, u in enumerate(labels):
            for offset in range(1, k // 2 + 1):
                v = labels[(i + offset) % n]
                graph.set_weight(u, v, 1.0)
        for i, u in enumerate(labels):
            for offset in range(1, k // 2 + 1):
                if rng.random() < p:
                    candidates = [w for w in labels if w != u and graph.weight(u, w) == 0.0]
                    if candidates:
                        graph.set_weight(u, int(rng.choice(candidates)), 1.0)
        return graph, {"neighbors": k, "rewire_probability": p}
    if family in ("sbm", "adversarial_mixed"):
        blocks = int(rng.choice([3, 4, 5]))
        sizes = [n // blocks] * blocks
        sizes[-1] += n - sum(sizes)
        p_in = float(rng.uniform(0.08, 0.20))
        p_out = float(rng.uniform(0.01, 0.05))
        graph = WeightedGraph(nodes=labels)
        bounds = []
        start = 0
        for size in sizes:
            bounds.append((start, start + size))
            start += size
        position = {int(label): index for index, label in enumerate(labels)}
        block_of = {}
        for bi, (lo, hi) in enumerate(bounds):
            for position_value in range(lo, hi):
                block_of[position_value] = bi
        for u in labels:
            for v in labels:
                if u >= v:
                    continue
                bi = block_of[position[u]]
                bj = block_of[position[v]]
                probability = p_in if bi == bj else p_out
                if rng.random() < probability:
                    graph.set_weight(u, v, 1.0)
        extras = {"block_sizes": sizes, "p_in": p_in, "p_out": p_out}
        if family == "adversarial_mixed":
            hubs = [int(rng.choice(labels)) for _ in range(max(1, n // 20))]
            added = []
            for hub in hubs:
                for other in rng.choice(labels, size=min(5, n - 1), replace=False):
                    other = int(other)
                    if other != hub and graph.weight(hub, other) == 0.0:
                        graph.set_weight(hub, other, 1.0)
                        added.append([hub, other])
            extras["hubs"] = hubs
            extras["added_edges"] = added
        return graph, extras
    if family == "planted_cut":
        half = n // 2
        p_between = float(rng.uniform(0.10, 0.30))
        p_inside = float(rng.uniform(0.01, 0.06))
        graph = WeightedGraph(nodes=labels)
        target = {label: (1 if index < half else -1) for index, label in enumerate(labels)}
        for u, v in itertools.combinations(labels, 2):
            probability = p_between if target[u] != target[v] else p_inside
            if rng.random() < probability:
                graph.set_weight(u, v, 1.0)
        repairs = _connected_repair(graph, labels)
        return graph, {"target_partition": {str(k): v for k, v in target.items()}, "p_between": p_between, "p_inside": p_inside, "repair_edges": repairs}
    if family == "random_geometric":
        ranges = {100: (0.16, 0.23), 200: (0.12, 0.18), 300: (0.10, 0.15)}
        radius = float(rng.uniform(*ranges[n]))
        positions = rng.random((n, 2))
        graph = WeightedGraph(nodes=labels)
        for i in range(n):
            for j in range(i + 1, n):
                if float(np.linalg.norm(positions[i] - positions[j])) <= radius:
                    graph.set_weight(labels[i], labels[j], 1.0)
        return graph, {"radius": radius, "positions": positions.tolist()}
    if family == "toroidal_grid":
        dimensions = {100: [10, 10], 200: [10, 20], 300: [15, 20]}[n]
        graph = WeightedGraph(nodes=labels)
        coordinates = list(itertools.product(*[range(d) for d in dimensions]))
        lookup = {coord: labels[i] for i, coord in enumerate(coordinates)}
        for coord in coordinates:
            for axis, size in enumerate(dimensions):
                nxt = list(coord)
                nxt[axis] = (nxt[axis] + 1) % size
                pair = (lookup[coord], lookup[tuple(nxt)])
                if pair[0] != pair[1]:
                    graph.set_weight(*pair, 1.0)
        return graph, {"dimensions": dimensions}
    if family == "complete":
        graph = WeightedGraph(nodes=labels)
        for u, v in itertools.combinations(labels, 2):
            graph.set_weight(u, v, 1.0)
        return graph, {}
    raise ValueError(f"unknown benchmark family {family!r}.")


def materialize_bench297(benchmark_id, base_seed=42):
    plan = resolve_bench297(benchmark_id, base_seed)
    labels = list(range(plan["n_vertices"]))
    graph, extras = _topology(plan["family"], plan["n_vertices"], plan["topology_seed"], labels)
    _assign_bench_weights(graph, plan["weight_seed"], plan["weights"])
    graph._recipe = {**plan, "resolved": {**plan, "hyperparameters": extras}}
    return graph


def materialize_bench16(benchmark_id):
    spec = resolve_bench16(benchmark_id)
    if "dimensions" in spec:
        dimensions = spec["dimensions"]
        n = int(np.prod(dimensions))
        labels = list(range(n))
        coordinates = list(itertools.product(*[range(d) for d in dimensions]))
        lookup = {coord: labels[i] for i, coord in enumerate(coordinates)}
        graph = WeightedGraph(nodes=labels)
        for coord in coordinates:
            for axis, size in enumerate(dimensions):
                nxt = list(coord)
                nxt[axis] = (nxt[axis] + 1) % size
                pair = (lookup[coord], lookup[tuple(nxt)])
                if pair[0] != pair[1]:
                    graph.set_weight(*pair, 1.0)
        rng = np.random.default_rng(benchmark_id)
        for u, v, _ in list(graph.edges):
            graph.set_weight(u, v, round(2 * rng.random(), 4))
        graph._recipe = {**spec, "resolved": {"label": "legacy_bench16_v1", "dimensions": list(dimensions)}}
        return graph
    n = spec["n"]
    labels = list(range(n))
    graph = WeightedGraph(nodes=labels)
    rng = np.random.default_rng(benchmark_id)
    density = float(spec["density"])
    for u, v in itertools.combinations(labels, 2):
        if rng.random() < density:
            graph.set_weight(u, v, 1.0)
    if spec["weights"] == "random":
        for u, v, _ in list(graph.edges):
            graph.set_weight(u, v, round(0.01 + rng.random(), 4))
    else:
        for u, v, _ in list(graph.edges):
            graph.set_weight(u, v, round(rng.choice([-1, 1]) * rng.randint(1, 10), 4))
    graph._recipe = {**spec, "resolved": {"label": "legacy_bench16_v1", "density": density}}
    return graph
