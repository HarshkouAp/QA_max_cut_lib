"""Typed subproblem plans and deterministic builders.

A plan is separate from a graph.  In particular, an edge partition stores
its assigned edge domain; a membership list alone is not enough to rebuild
the local edge-subset Hamiltonian.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import itertools
import math

import numpy as np

from ..core.graph import WeightedGraph

__all__ = [
    "SubproblemSpec",
    "SubproblemPlan",
    "validate_plan",
    "build_random_partition_plan",
    "build_connected_bfs_partition",
    "build_adaptive_dense_induced",
    "build_edge_neighborhood_cover",
    "build_edge_coverage_scan",
    "build_overlap_edge_partition",
    "canonical_edge",
]


def canonical_edge(u, v):
    return (min(int(u), int(v)), max(int(u), int(v)))


def _graph_hash(graph):
    return graph.snapshot()["fingerprint"]


def _part(task_id, nodes, graph, *, edges=None, edge_domain="induced", round_index=0, anchors=(), provenance=None):
    nodes = tuple(sorted(int(node) for node in nodes))
    if edges is None:
        chosen = set(nodes)
        edges = tuple(
            (int(u), int(v), float(weight))
            for u, v, weight in graph.edges
            if u in chosen and v in chosen
        )
    else:
        edges = tuple(sorted((int(u), int(v), float(weight)) for u, v, weight in edges))
    return SubproblemSpec(
        task_id=str(task_id),
        nodes=nodes,
        edges=edges,
        edge_domain=edge_domain,
        anchors=tuple(tuple(anchor) for anchor in anchors),
        round=int(round_index),
        provenance=deepcopy(provenance or {}),
    )


@dataclass
class SubproblemSpec:
    task_id: str
    nodes: tuple
    edges: tuple
    edge_domain: str = "induced"
    anchors: tuple = ()
    round: int = 0
    provenance: dict = field(default_factory=dict)

    @property
    def size(self):
        return len(self.nodes)

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "nodes": [int(node) for node in self.nodes],
            "edge_domain": self.edge_domain,
            "is_induced": self.edge_domain == "induced",
            "edges": [[int(u), int(v), float(w)] for u, v, w in self.edges],
            "size": len(self.nodes),
            "anchors": [list(anchor) for anchor in self.anchors],
            "round": int(self.round),
            "provenance": deepcopy(self.provenance),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            task_id=str(data["task_id"]),
            nodes=tuple(int(node) for node in data["nodes"]),
            edges=tuple((int(u), int(v), float(w)) for u, v, w in data.get("edges", [])),
            edge_domain=data.get("edge_domain", "induced"),
            anchors=tuple(tuple(anchor) for anchor in data.get("anchors", [])),
            round=int(data.get("round", 0)),
            provenance=deepcopy(data.get("provenance", {})),
        )


@dataclass
class SubproblemPlan:
    plan_kind: str
    parent_hash: str
    builder_recipe: dict
    parts: tuple
    completion_status: str = "complete"
    coverage: dict = field(default_factory=dict)
    seed: int | None = None
    limits: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "schema_version": 1,
            "plan_kind": self.plan_kind,
            "parent_hash": self.parent_hash,
            "builder_recipe": deepcopy(self.builder_recipe),
            "seed": self.seed,
            "completion_status": self.completion_status,
            "coverage": deepcopy(self.coverage),
            "limits": deepcopy(self.limits),
            "parts": [part.to_dict() if isinstance(part, SubproblemSpec) else part for part in self.parts],
        }

    @classmethod
    def from_dict(cls, data):
        parts = tuple(
            part if isinstance(part, SubproblemSpec) else SubproblemSpec.from_dict(part)
            for part in data.get("parts", [])
        )
        return cls(
            plan_kind=data["plan_kind"],
            parent_hash=data["parent_hash"],
            builder_recipe=deepcopy(data.get("builder_recipe", {})),
            parts=parts,
            completion_status=data.get("completion_status", "complete"),
            coverage=deepcopy(data.get("coverage", {})),
            seed=data.get("seed"),
            limits=deepcopy(data.get("limits", {})),
        )


def _parent_edges(graph):
    return {(canonical_edge(u, v)): float(w) for u, v, w in graph.edges}


def validate_plan(graph, plan):
    """Validate parent identity, membership, edge weights and completion."""
    if not isinstance(graph, WeightedGraph):
        raise TypeError("graph must be a WeightedGraph.")
    if isinstance(plan, dict):
        plan = SubproblemPlan.from_dict(plan)
    if plan.parent_hash != _graph_hash(graph):
        raise ValueError("plan parent hash does not match the graph.")
    parent_edges = _parent_edges(graph)
    all_nodes = set(int(node) for node in graph.nodes)
    seen_edges = {}
    seen_vertices = {}
    for spec in plan.parts:
        nodes = set(spec.nodes)
        if not nodes <= all_nodes:
            raise ValueError("part contains a label absent from the parent.")
        for node in nodes:
            seen_vertices[node] = seen_vertices.get(node, 0) + 1
        if spec.edge_domain == "induced":
            expected = {
                key: weight
                for key, weight in parent_edges.items()
                if key[0] in nodes and key[1] in nodes
            }
            actual = {(canonical_edge(u, v)): float(w) for u, v, w in spec.edges}
            if actual != expected:
                raise ValueError("induced part edge domain does not equal G[S].")
        elif spec.edge_domain == "explicit":
            actual = {}
            for u, v, weight in spec.edges:
                key = canonical_edge(u, v)
                if key not in parent_edges:
                    raise ValueError("assigned edge is absent from the parent.")
                if not {u, v} <= nodes:
                    raise ValueError("assigned edge endpoint is outside membership.")
                if parent_edges[key] != float(weight):
                    raise ValueError("assigned edge weight differs from the parent.")
                if key in actual:
                    raise ValueError("duplicate assigned edge in one part.")
                if key in seen_edges:
                    raise ValueError("edge is assigned to more than one part.")
                actual[key] = float(weight)
                seen_edges[key] = spec.task_id
        else:
            raise ValueError("unknown edge_domain.")
    if plan.plan_kind == "vertex_partition":
        if any(count != 1 for count in seen_vertices.values()):
            raise ValueError("vertex_partition memberships overlap.")
        if set(seen_vertices) != all_nodes:
            raise ValueError("vertex_partition does not cover every vertex.")
    elif plan.plan_kind == "edge_partition" and plan.completion_status == "complete":
        if set(seen_edges) != set(parent_edges):
            raise ValueError("complete edge_partition does not cover every edge.")
    elif plan.plan_kind not in ("induced_cover", "edge_partition"):
        raise ValueError("unknown plan_kind.")
    return True


def _coverage(graph, specs):
    parent_nodes = set(graph.nodes)
    parent_edges = set(_parent_edges(graph))
    covered_nodes = set()
    covered_edges = set()
    pair_counts = {}
    edge_counts = {}
    for spec in specs:
        nodes = set(spec.nodes)
        covered_nodes.update(nodes)
        for pair in itertools.combinations(sorted(nodes), 2):
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        for u, v, _ in spec.edges:
            key = canonical_edge(u, v)
            covered_edges.add(key)
            edge_counts[key] = edge_counts.get(key, 0) + 1
    total_pairs = len(parent_nodes) * (len(parent_nodes) - 1) // 2 if len(parent_nodes) >= 2 else 0
    return {
        "vertices": len(covered_nodes),
        "vertices_total": len(parent_nodes),
        "vertex_fraction": len(covered_nodes) / len(parent_nodes) if parent_nodes else None,
        "edges": len(covered_edges),
        "edges_total": len(parent_edges),
        "edge_fraction": len(covered_edges) / len(parent_edges) if parent_edges else None,
        "pairs": len(pair_counts),
        "pairs_total": total_pairs,
        "pair_fraction": len(pair_counts) / total_pairs if total_pairs else None,
        "pair_multiplicity_max": max(pair_counts.values(), default=0),
        "edge_multiplicity_max": max(edge_counts.values(), default=0),
        "vertex_complete": len(covered_nodes) == len(parent_nodes),
        "edge_complete": len(covered_edges) == len(parent_edges),
        "pair_complete": len(pair_counts) == total_pairs,
        "uncovered_vertices": sorted(set(parent_nodes) - covered_nodes),
        "uncovered_edges": sorted([list(pair) for pair in set(parent_edges) - covered_edges]),
    }


def _make_plan(graph, plan_kind, builder_recipe, specs, *, seed=None, completion_status="complete", limits=None):
    plan = SubproblemPlan(
        plan_kind=plan_kind,
        parent_hash=_graph_hash(graph),
        builder_recipe=deepcopy(builder_recipe),
        parts=tuple(specs),
        completion_status=completion_status,
        coverage=_coverage(graph, specs),
        seed=seed,
        limits=deepcopy(limits or {}),
    )
    validate_plan(graph, plan)
    return plan


def build_random_partition_plan(graph, *, max_size=6, seed=0, version=0, full_graph_when_fits=True):
    from .subgraphs import build_subgraph_builder

    builder = build_subgraph_builder(
        {"type": "random_partition", "parameters": {"max_size": max_size, "rounds": 1, "full_graph_when_fits": full_graph_when_fits}},
        seed=seed,
    )
    specs = [
        _part(task["task_id"], task["nodes"], graph, round_index=0)
        for task in builder.build(graph, version=version)
    ]
    return _make_plan(graph, "vertex_partition", {"type": "random_partition", "max_size": max_size, "seed": seed}, specs, seed=seed)


def _randomized_bfs_order(graph, remaining, rng):
    start = int(remaining[0])
    block = []
    queue = [start]
    seen = {start}
    while queue and len(block) < len(remaining):
        node = queue.pop(0)
        if node not in remaining:
            continue
        block.append(node)
        neighbours = sorted(
            int(v if u == node else u)
            for u, v, weight in graph.edges
            if float(weight) != 0.0 and (u == node or v == node)
        )
        neighbours = [other for other in neighbours if other in remaining and other not in seen]
        if neighbours:
            neighbours = rng.permutation(np.asarray(neighbours, dtype=np.int64)).tolist()
            for other in neighbours:
                if int(other) not in seen:
                    seen.add(int(other))
                    queue.append(int(other))
    return block


def build_connected_bfs_partition(graph, *, max_size=6, seed=0, version=0, merge_singletons=False):
    rng = np.random.default_rng(seed)
    remaining = set(int(node) for node in graph.nodes)
    order = [int(node) for node in rng.permutation(np.asarray(sorted(remaining), dtype=np.int64))]
    specs = []
    block_index = 0
    while remaining:
        seed_node = next(node for node in order if node in remaining)
        connected = [seed_node]
        frontier = {
            int(v if u == seed_node else u)
            for u, v, weight in graph.edges
            if float(weight) != 0.0 and (u == seed_node or v == seed_node)
        } & remaining
        while len(connected) < max_size and frontier:
            candidates = sorted(frontier - set(connected))
            if not candidates:
                break
            shuffled = [int(node) for node in rng.permutation(np.asarray(candidates, dtype=np.int64))]
            chosen = shuffled[0]
            connected.append(chosen)
            frontier.discard(chosen)
            for u, v, weight in graph.edges:
                if float(weight) == 0.0:
                    continue
                if u == chosen and v in remaining:
                    frontier.add(v)
                elif v == chosen and u in remaining:
                    frontier.add(u)
        for node in connected:
            remaining.discard(node)
        specs.append(_part(f"bfs:{block_index}", connected, graph, round_index=0))
        block_index += 1
    if merge_singletons:
        merged_ids = set()
        for spec in sorted(specs, key=lambda item: item.task_id):
            if spec.size != 1:
                continue
            singleton = spec.nodes[0]
            for target in specs:
                if target.task_id == spec.task_id or target.size >= max_size:
                    continue
                if singleton in target.nodes:
                    continue
                adjacent = any(
                    canonical_edge(singleton, node) in {
                        canonical_edge(u, v) for u, v, _ in graph.edges
                    }
                    for node in target.nodes
                )
                if adjacent:
                    nodes_val = tuple(sorted(set(target.nodes) | {singleton}))
                    target.nodes = nodes_val
                    target.edges = tuple(
                        (u, v, w) for u, v, w in graph.edges if u in nodes_val and v in nodes_val
                    )
                    merged_ids.add(spec.task_id)
                    break
        specs = [spec for spec in specs if spec.task_id not in merged_ids]
    return _make_plan(graph, "vertex_partition", {"type": "connected_bfs_partition_v1", "max_size": max_size, "seed": seed, "merge_singletons": merge_singletons}, specs, seed=seed)


def _topology_metrics(graph):
    nodes = list(graph.nodes)
    adjacency = {node: set() for node in nodes}
    for u, v, weight in graph.edges:
        if float(weight) != 0.0:
            adjacency[u].add(v)
            adjacency[v].add(u)
    degree = {node: len(adjacency[node]) for node in nodes}
    triangles = {}
    clustering = {}
    for node in nodes:
        neighbours = adjacency[node]
        count = sum(1 for u, v in itertools.combinations(sorted(neighbours), 2) if v in adjacency[u])
        triangles[node] = count
        denominator = len(neighbours) * (len(neighbours) - 1) / 2
        clustering[node] = count / denominator if denominator else 0.0
    return adjacency, degree, triangles, clustering


def _mean_distance_to_set(graph, node, members, adjacency):
    distances = []
    queue = [int(member) for member in members]
    seen = {member: 0 for member in members}
    while queue:
        current = queue.pop(0)
        for other in sorted(adjacency[current]):
            if other not in seen:
                seen[other] = seen[current] + 1
                queue.append(other)
    for member in members:
        if node == member:
            distances.append(0.0)
        elif node in seen:
            distances.append(float(seen[node]))
        else:
            distances.append(float(len(graph)))
    return sum(distances) / len(distances) if distances else float(len(graph))


def build_adaptive_dense_induced(
    graph,
    *,
    max_size=6,
    min_size=1,
    target_parts=4,
    overlap_policy="vertex_disjoint",
    overlap_cap=0.3,
    temperature=0.5,
    attempts=50,
    seed=0,
    version=0,
    alpha=1.0,
    beta=0.5,
    gamma=0.5,
    delta=0.5,
):
    if overlap_policy not in ("vertex_disjoint", "union_fraction_cap", "unrestricted"):
        raise ValueError("unknown overlap policy.")
    rng = np.random.default_rng(seed)
    adjacency, degree, triangles, clustering = _topology_metrics(graph)
    labels = sorted(int(node) for node in graph.nodes)
    if not labels:
        return _make_plan(graph, "induced_cover", {"type": "adaptive_dense_induced_v1"}, [], seed=seed, completion_status="empty_parent")
    max_degree = max(degree.values(), default=0)
    max_triangles = max(triangles.values(), default=0)
    used = set()
    specs = []
    for part_index in range(target_parts):
        allowed = [
            node for node in labels
            if overlap_policy == "unrestricted"
            or node not in used
            or (
                overlap_policy == "union_fraction_cap"
                and (1 / 1) <= overlap_cap
            )
        ]
        if not allowed:
            break
        def start_score(node):
            dn = degree[node] / max_degree if max_degree else 0.0
            tn = triangles[node] / max_triangles if max_triangles else 0.0
            return 0.4 * dn + 0.3 * clustering[node] + 0.2 * tn + 0.1 * dn * tn
        start = min(allowed, key=lambda node: (-start_score(node), node))
        if overlap_policy == "union_fraction_cap" and len(set([start]) & used) / 1.0 > overlap_cap:
            break
        members = [start]
        boundary = set(adjacency[start])
        steps = 0
        max_steps = max(1, len(labels) * 2)
        while len(members) < max_size and steps < max_steps:
            steps += 1
            claimed = set(used) if overlap_policy == "vertex_disjoint" else set()
            candidates = sorted(
                node for node in boundary
                if node not in members and node not in claimed
                and (
                    overlap_policy != "union_fraction_cap"
                    or len((set(members) | {node}) & used) / (len(members) + 1) <= overlap_cap
                )
            )
            if not candidates:
                break
            ratings = []
            for node in candidates:
                value = (
                    alpha * len(adjacency[node] & set(members))
                    + beta * len(adjacency[node] & boundary)
                    + gamma * clustering[node]
                    + delta * (
                        (degree[node] / max_degree if max_degree else 0.0)
                        + (triangles[node] / max_triangles if max_triangles else 0.0)
                    )
                    + 0.1 / (1.0 + _mean_distance_to_set(graph, node, members, adjacency))
                )
                ratings.append(value)
            count = min(max_size - len(members), max(1, len(candidates) // 2))
            if temperature <= 0:
                chosen = [
                    node for _, node in sorted(zip(ratings, candidates), key=lambda item: (-item[0], item[1]))[:count]
                ]
            else:
                low, high = min(ratings), max(ratings)
                normalized = [(value - low) / (high - low) if high > low else 0.0 for value in ratings]
                weights = np.exp((np.asarray(normalized) - max(normalized)) / temperature)
                weights /= weights.sum()
                chosen = [
                    int(rng.choice(candidates, p=weights))
                ]
                while len(chosen) < count:
                    remaining = [node for node in candidates if node not in chosen]
                    if not remaining:
                        break
                    local_weights = [weights[candidates.index(node)] for node in remaining]
                    total = sum(local_weights)
                    local_weights = [value / total for value in local_weights]
                    chosen.append(int(rng.choice(remaining, p=local_weights)))
            for node in chosen:
                members.append(node)
                boundary.add(node)
                boundary.update(adjacency[node])
            boundary -= set(members)
        if len(members) < min_size:
            break
        if overlap_policy == "union_fraction_cap" and len(set(members) & used) / len(members) > overlap_cap:
            break
        specs.append(_part(f"adaptive:{part_index}", members, graph, round_index=0, provenance={"temperature": temperature}))
        if overlap_policy != "unrestricted":
            used.update(members)
    status = "complete" if len(specs) == target_parts else "target_count_not_met"
    return _make_plan(
        graph,
        "induced_cover",
        {
            "type": "adaptive_dense_induced_v1",
            "max_size": max_size,
            "min_size": min_size,
            "target_parts": target_parts,
            "overlap_policy": overlap_policy,
        },
        specs,
        seed=seed,
        completion_status=status,
    )


def _edge_gain(graph, node, members, growth_metric):
    weights = [abs(float(graph.weight(node, member))) if growth_metric == "abs_weight" else 1.0 for member in members]
    return sum(weights)


def _neighborhood_block(graph, endpoints, k, growth_metric, rng):
    anchor_pair = canonical_edge(*endpoints)
    members = list(anchor_pair)
    covered = set()
    direct = set()
    for endpoint in anchor_pair:
        for u, v, weight in graph.edges:
            if float(weight) == 0.0:
                continue
            if u == endpoint:
                direct.add(v)
            elif v == endpoint:
                direct.add(u)
    direct -= set(anchor_pair)
    covered.update(direct)
    parts = []
    uncovered = set(direct)
    while uncovered:
        seed = max(uncovered, key=lambda node: (sum(1 for u, v, w in graph.edges if float(w) != 0 and (u == node or v == node)), -int(node)))
        block = [*anchor_pair, int(seed)]
        uncovered.discard(seed)
        while len(block) < k:
            candidates = sorted(uncovered - set(block))
            if not candidates:
                break
            chosen = max(candidates, key=lambda node: (_edge_gain(graph, node, block, growth_metric), -int(node)))
            block.append(chosen)
            uncovered.discard(chosen)
        if len(block) < k:
            two_hop = set()
            for member in block:
                for u, v, weight in graph.edges:
                    if float(weight) == 0.0:
                        continue
                    if u == member:
                        two_hop.add(v)
                    elif v == member:
                        two_hop.add(u)
            candidates = sorted(two_hop - set(block) - set(uncovered))
            while len(block) < k and candidates:
                chosen = max(candidates, key=lambda node: (_edge_gain(graph, node, block, growth_metric), -int(node)))
                if _edge_gain(graph, chosen, block, growth_metric) <= 0:
                    break
                block.append(chosen)
                candidates.remove(chosen)
        parts.append(tuple(sorted(block)))
    if not parts:
        parts = [tuple(sorted(anchor_pair))]
    return parts


def build_edge_neighborhood_cover(graph, anchor, *, max_size=3, goal="cover_neighbors", growth_metric="count", seed=0, version=0):
    endpoints = canonical_edge(*anchor)
    if endpoints[0] not in graph.nodes or endpoints[1] not in graph.nodes:
        raise ValueError("anchor endpoints must be active.")
    if float(graph.weight(*endpoints)) == 0.0:
        raise ValueError("anchor must be a nonzero parent edge.")
    if goal not in ("cover_neighbors", "anchor_only"):
        raise ValueError("unknown edge-neighborhood goal.")
    if goal == "cover_neighbors" and max_size < 3:
        direct = set()
        for endpoint in endpoints:
            for u, v, weight in graph.edges:
                if float(weight) != 0.0 and (u == endpoint or v == endpoint):
                    direct.add(v if u == endpoint else u)
        direct -= set(endpoints)
        if direct:
            raise ValueError("cover_neighbors needs max_size >= 3 when neighbors exist.")
    if goal == "anchor_only":
        blocks = [tuple(sorted(endpoints))]
        if max_size > 2:
            members = list(endpoints)
            candidates = sorted(
                {v if u in endpoints else u for u, v, weight in graph.edges if float(weight) != 0.0 and (u in endpoints or v in endpoints)}
                - set(endpoints)
            )
            while len(members) < max_size and candidates:
                chosen = max(candidates, key=lambda node: (_edge_gain(graph, node, members, growth_metric), -int(node)))
                members.append(chosen)
                candidates.remove(chosen)
            blocks = [tuple(sorted(members))]
    else:
        blocks = _neighborhood_block(graph, endpoints, max_size, growth_metric, np.random.default_rng(seed))
    specs = [
        _part(f"edge-neighborhood:{index}", block, graph, anchors=[endpoints], round_index=0)
        for index, block in enumerate(blocks)
    ]
    return _make_plan(graph, "induced_cover", {"type": "edge_neighborhood_cover_v1", "goal": goal, "max_size": max_size}, specs, seed=seed)


def build_overlap_edge_partition(graph, *, max_size=3, seed=0, version=0, growth_metric="abs_weight", lambda_internal=1.0, lambda_overlap=1.0, seed_strategy="max_overlap", max_tasks=None):
    if max_size < 2:
        raise ValueError("max_size must be at least 2.")
    if growth_metric not in ("count", "abs_weight"):
        raise ValueError("unknown growth metric.")
    if seed_strategy not in ("max_overlap", "random"):
        raise ValueError("unknown seed strategy.")
    rng = np.random.default_rng(seed)
    unassigned = {canonical_edge(u, v): float(w) for u, v, w in graph.edges}
    history = {int(node): 0 for node in graph.nodes}
    specs = []
    part_index = 0
    while unassigned:
        if max_tasks is not None and len(specs) >= max_tasks:
            return _make_plan(graph, "edge_partition", {"type": "overlap_edge_partition_abs_v1"}, specs, seed=seed, completion_status="max_tasks")
        active = sorted({node for pair in unassigned for node in pair})
        candidates = [node for node in active if history[node] > 0] if seed_strategy == "max_overlap" else []
        if not candidates:
            candidates = active
        def seed_score(node):
            incident = sum(
                (1.0 if growth_metric == "count" else abs(weight))
                for pair, weight in unassigned.items()
                if node in pair
            )
            return lambda_internal * incident + lambda_overlap * history[node]
        if seed_strategy == "random":
            seed_node = int(rng.choice(candidates))
        else:
            seed_node = min(candidates, key=lambda node: (-seed_score(node), node))
        incident_edges = sorted(
            (pair for pair in unassigned if seed_node in pair),
            key=lambda pair: (-((1.0 if growth_metric == "count" else abs(unassigned[pair])) + lambda_overlap * history[pair[0] if pair[1] == seed_node else pair[1]]), pair),
        )
        if not incident_edges:
            raise RuntimeError("active vertex has no unassigned edge.")
        other = incident_edges[0][0] if incident_edges[0][1] == seed_node else incident_edges[0][1]
        members = {seed_node, int(other)}
        while len(members) < max_size:
            outside = sorted(set(active) - members)
            scored = []
            for node in outside:
                if not any(node in pair for pair in unassigned):
                    continue
                internal = sum(
                    (1.0 if growth_metric == "count" else abs(weight))
                    for pair, weight in unassigned.items()
                    if pair[0] in members and pair[1] == node or pair[1] in members and pair[0] == node
                )
                if internal <= 0:
                    continue
                scored.append((-(lambda_internal * internal + lambda_overlap * history[node]), node))
            if not scored:
                break
            members.add(min(scored)[1])
            active = sorted({node for pair in unassigned for node in pair})
        assigned = [
            (u, v, weight) for (u, v), weight in unassigned.items()
            if u in members and v in members
        ]
        if not assigned:
            raise RuntimeError("edge-partition growth produced no assigned edges.")
        specs.append(_part(f"overlap-edge:{part_index}", members, graph, edges=assigned, edge_domain="explicit", round_index=0, provenance={"history": {str(node): history[node] for node in members}}))
        for u, v, _ in assigned:
            del unassigned[(u, v)]
        for node in members:
            history[node] += 1
        part_index += 1
    for node in sorted(graph.nodes):
        if history[node] == 0:
            specs.append(_part(f"overlap-edge:{part_index}", [node], graph, edges=[], edge_domain="explicit", round_index=0))
            part_index += 1
    return _make_plan(graph, "edge_partition", {"type": "overlap_edge_partition_abs_v1", "max_size": max_size, "growth_metric": growth_metric, "seed_strategy": seed_strategy}, specs, seed=seed)


def build_edge_coverage_scan(
    graph,
    *,
    target_edge_coverage=1.0,
    max_tasks=64,
    max_episodes=32,
    cover_isolates=True,
    anchor_policy="random",
    local_builder=None,
    seed=0,
    version=0,
):
    """Scan parent edges with atomic edge-neighborhood episodes."""
    if not 0.0 <= float(target_edge_coverage) <= 1.0:
        raise ValueError("target_edge_coverage must be in [0,1].")
    if anchor_policy not in ("random", "max_abs_weight"):
        raise ValueError("unknown anchor policy.")
    local_builder = deepcopy(local_builder or {"type": "edge_neighborhood_cover_v1", "max_size": 3, "goal": "cover_neighbors", "growth_metric": "count"})
    k = int(local_builder.get("max_size", 3))
    growth = local_builder.get("growth_metric", "count")
    goal = local_builder.get("goal", "cover_neighbors")
    rng = np.random.default_rng(seed)
    remaining = {canonical_edge(u, v): float(w) for u, v, w in graph.edges}
    total = len(remaining)
    assigned_specs = []
    used_edges = set()
    episodes = 0
    while remaining and episodes < max_episodes:
        if total and len(used_edges) / total >= float(target_edge_coverage):
            break
        keys = sorted(remaining)
        if anchor_policy == "random":
            anchor = keys[int(rng.integers(0, len(keys)))]
        else:
            anchor = min(keys, key=lambda pair: (-abs(remaining[pair]), pair))
        blocks = _neighborhood_block(graph, anchor, k, growth, rng)
        block = sorted(blocks[0], key=lambda node: (0 if node in anchor else 1, node))
        # Keep the anchor at the front conceptually; membership is sorted.
        candidates = [
            (u, v, weight) for (u, v), weight in remaining.items()
            if u in block and v in block
        ]
        if not candidates:
            raise RuntimeError("edge episode assigned no edge.")
        if len(assigned_specs) + 1 > max_tasks:
            return _make_plan(
                graph,
                "edge_partition",
                {"type": "edge_coverage_scan_v1", "anchor_policy": anchor_policy, "max_size": k},
                assigned_specs,
                seed=seed,
                completion_status="max_tasks",
                limits={"max_tasks": int(max_tasks), "max_episodes": int(max_episodes)},
            )
        assigned_specs.append(
            _part(
                f"scan:{episodes}",
                block,
                graph,
                edges=candidates,
                edge_domain="explicit",
                round_index=0,
                anchors=[anchor],
                provenance={"anchor": list(anchor), "goal": goal},
            )
        )
        for u, v, _ in candidates:
            remaining.pop((u, v), None)
            used_edges.add((u, v))
        episodes += 1
    if cover_isolates:
        touched = {node for spec in assigned_specs for node in spec.nodes}
        for node in sorted(graph.nodes):
            if node not in touched:
                if len(assigned_specs) + 1 > max_tasks:
                    return _make_plan(
                        graph, "edge_partition",
                        {"type": "edge_coverage_scan_v1"}, assigned_specs,
                        seed=seed, completion_status="max_tasks", limits={"max_tasks": max_tasks},
                    )
                assigned_specs.append(
                    _part(f"scan:isolate:{node}", [node], graph, edges=[], edge_domain="explicit", round_index=0)
                )
    complete = not remaining
    return _make_plan(
        graph,
        "edge_partition",
        {"type": "edge_coverage_scan_v1", "anchor_policy": anchor_policy, "max_size": k, "growth_metric": growth, "goal": goal},
        assigned_specs,
        seed=seed,
        completion_status="complete" if complete else "coverage_shortfall",
        limits={"max_tasks": int(max_tasks), "max_episodes": int(max_episodes)},
    )
