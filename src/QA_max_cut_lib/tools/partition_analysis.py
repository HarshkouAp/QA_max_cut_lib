"""Independent coverage and topology analysis for typed plans.

The analyzer reads a graph snapshot and a plan.  It never calls a builder,
solver, sampler, or plotting backend.
"""

from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
import itertools
import math

from ..core.graph import WeightedGraph
from ..generators.plans import SubproblemPlan, validate_plan, canonical_edge

__all__ = [
    "analyze_plan",
    "topology_descriptors",
    "hop_metrics",
    "plan_to_graph",
]


def _adjacency(graph):
    nodes = list(graph.nodes)
    adjacency = {int(node): set() for node in nodes}
    for u, v, weight in graph.edges:
        if float(weight) != 0.0:
            adjacency[int(u)].add(int(v))
            adjacency[int(v)].add(int(u))
    return adjacency


def _bfs(adjacency, source):
    distance = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for other in sorted(adjacency[node]):
            if other not in distance:
                distance[other] = distance[node] + 1
                queue.append(other)
    return distance


def topology_descriptors(graph):
    n = len(graph)
    edges = [(u, v, w) for u, v, w in graph.edges if float(w) != 0.0]
    m = len(edges)
    adjacency = _adjacency(graph)
    degree = {node: len(adjacency[node]) for node in adjacency}
    mean_degree = 2.0 * m / n if n else None
    density = 2.0 * m / (n * (n - 1)) if n >= 2 else None
    variance = (
        sum((value - mean_degree) ** 2 for value in degree.values()) / n
        if n and mean_degree is not None else None
    )
    triangles = 0
    coefficients = []
    for node in adjacency:
        neighbours = sorted(adjacency[node])
        local = sum(1 for u, v in itertools.combinations(neighbours, 2) if v in adjacency[u])
        triangles += local
        denominator = len(neighbours) * (len(neighbours) - 1) / 2
        coefficients.append(local / denominator if denominator else 0.0)
    return {
        "n": n,
        "m": m,
        "density": density,
        "mean_degree": mean_degree,
        "degree_variance": variance,
        "triangles": triangles // 3,
        "mean_local_clustering": sum(coefficients) / n if n else None,
        "regions": {
            "density": "2m/[n(n-1)]" if n >= 2 else None,
            "mean_degree": "2m/n" if n else None,
            "clustering": "mean local triangle/choose(degree,2), degree<2 -> 0",
        },
    }


def hop_metrics(graph, *, sample_pairs=None, seed=0):
    """Return BFS hop metrics; exact unless an explicit pair sample is given."""
    n = len(graph)
    total_pairs = n * (n - 1) // 2 if n >= 2 else 0
    labels = sorted(int(node) for node in graph.nodes)
    adjacency = _adjacency(graph)
    if sample_pairs is not None:
        pairs = set()
        for pair in sample_pairs:
            u, v = canonical_edge(*pair)
            if u not in labels or v not in labels or u == v:
                raise ValueError("sample pair is not active.")
            pairs.add((u, v))
        sampled = True
    else:
        pairs = set(itertools.combinations(labels, 2))
        sampled = False
    # BFS only for sources that occur in the requested set.
    sources = {pair[0] for pair in pairs}
    distances = {source: _bfs(adjacency, source) for source in sources}
    reachable = 0
    total = 0.0
    inverse = 0.0
    for pair in pairs:
        distance = distances.get(pair[0], {}).get(pair[1])
        if distance is None:
            continue
        reachable += 1
        total += distance
        inverse += 1.0 / distance
    mean = total / reachable if reachable else None
    all_mean = None
    if not sampled and n >= 2 and reachable == total_pairs:
        all_mean = total / total_pairs
    return {
        "mode": "sampled" if sampled else "exact",
        "n_pairs_universe": total_pairs,
        "n_pairs_requested": len(pairs),
        "n_pairs_reachable": reachable,
        "reachability_fraction": reachable / len(pairs) if pairs else None,
        "reachable_mean_hop_distance": mean,
        "all_pairs_mean_hop_distance": all_mean,
        "global_efficiency": inverse / len(pairs) if pairs else None,
        "unreachable_pairs": len(pairs) - reachable,
        "sampled": sampled,
        "sample_seed": seed if sampled else None,
        "all_pairs_mean_reason": None if all_mean is not None else (
            "empty_heap" if n < 2 else "disconnected_or_sampled"
        ),
    }


def _part_graph(graph, spec):
    if spec.edge_domain == "induced":
        return graph.subgraph(spec.nodes)
    return WeightedGraph(
        nodes=spec.nodes,
        edges=[(u, v, weight) for u, v, weight in spec.edges],
    )


def analyze_plan(graph, plan, *, sample_pairs=None, seed=0):
    """Return an independent AnalysisRecord for a graph and typed plan."""
    if not isinstance(graph, WeightedGraph):
        raise TypeError("graph must be a WeightedGraph.")
    if isinstance(plan, dict):
        plan = SubproblemPlan.from_dict(plan)
    validate_plan(graph, plan)
    parent_edges = {canonical_edge(u, v): float(w) for u, v, w in graph.edges}
    covered_vertices = Counter()
    covered_edges = Counter()
    pair_counts = Counter()
    abs_covered = 0.0
    signed_covered = 0.0
    part_records = []
    for spec in plan.parts:
        nodes = set(spec.nodes)
        for node in nodes:
            covered_vertices[node] += 1
        if spec.edge_domain == "induced":
            edges = {
                key: parent_edges[key]
                for key in parent_edges
                if key[0] in nodes and key[1] in nodes
            }
        else:
            edges = {canonical_edge(u, v): float(w) for u, v, w in spec.edges}
        for pair in itertools.combinations(sorted(nodes), 2):
            pair_counts[pair] += 1
        for key, weight in edges.items():
            covered_edges[key] += 1
            abs_covered += abs(weight)
            signed_covered += weight
        local_graph = _part_graph(graph, spec)
        part_records.append(
            {
                "task_id": spec.task_id,
                "nodes": list(spec.nodes),
                "size": spec.size,
                "edge_domain": spec.edge_domain,
                "assigned_edges": len(edges),
                "assigned_abs_weight": sum(abs(value) for value in edges.values()),
                "topology": topology_descriptors(local_graph),
                "hop": hop_metrics(local_graph, seed=seed),
            }
        )
    total_abs = sum(abs(weight) for weight in parent_edges.values())
    n = len(graph)
    total_pairs = n * (n - 1) // 2 if n >= 2 else 0
    return {
        "parent_hash": plan.parent_hash,
        "plan_kind": plan.plan_kind,
        "completion_status": plan.completion_status,
        "vertices": len(covered_vertices),
        "vertices_total": n,
        "vertex_coverage": len(covered_vertices) / n if n else None,
        "edges": len(covered_edges),
        "edges_total": len(parent_edges),
        "edge_coverage": len(covered_edges) / len(parent_edges) if parent_edges else None,
        "pairs": len(pair_counts),
        "pairs_total": total_pairs,
        "pair_coverage": len(pair_counts) / total_pairs if total_pairs else None,
        "uncovered_vertices": sorted(set(graph.nodes) - set(covered_vertices)),
        "uncovered_edges": sorted([list(pair) for pair in set(parent_edges) - set(covered_edges)]),
        "vertex_multiplicity": [[int(node), int(count)] for node, count in sorted(covered_vertices.items())],
        "edge_multiplicity": [[int(pair[0]), int(pair[1]), int(count)] for pair, count in sorted(covered_edges.items())],
        "repeated_vertex_incidences": sum(count - 1 for count in covered_vertices.values()),
        "repeated_edge_assignments": sum(count - 1 for count in covered_edges.values()),
        "abs_weight_coverage": abs_covered / total_abs if total_abs else None,
        "signed_weight_total": signed_covered,
        "abs_weight_total": total_abs,
        "topology": topology_descriptors(graph),
        "hop": hop_metrics(graph, sample_pairs=sample_pairs, seed=seed),
        "parts": part_records,
        "edge_exclusive": all(count <= 1 for count in covered_edges.values()) if plan.plan_kind == "edge_partition" else None,
        "invalid_edges": [],
    }


def plan_to_graph(graph, spec):
    """Return the exact local graph declared by one part."""
    return _part_graph(graph, spec)
