"""Block mean-field Ising problem and compact moment helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math

import numpy as np

from ..core.graph import WeightedGraph

__all__ = [
    "MeanFieldError",
    "MeanFieldBlockProblem",
    "build_block_problem",
    "canonical_edge",
]


class MeanFieldError(ValueError):
    """Raised for an invalid partition, message, or block problem."""


def canonical_edge(u, v):
    """Return the canonical undirected edge key."""
    return (int(u), int(v)) if int(u) <= int(v) else (int(v), int(u))


@dataclass
class MeanFieldBlockProblem:
    """One block Ising problem with fields from external mean spins.

    Parameters
    ----------
    graph : WeightedGraph
        Immutable parent graph.
    nodes : tuple
        Block node order.  Bit ``q`` of an active index follows this tuple.
    block_id : int
        Partition block index.
    couplings : dict
        Internal ``(u, v) -> w_uv / 2`` coefficients.
    fields : dict
        ``i -> (1/2) sum_j w_ij m_j`` over nodes outside the block.
    constant : float
        Local affected constant ``c_aff``.
    """

    graph: WeightedGraph
    nodes: tuple
    block_id: int
    couplings: dict
    fields: dict
    constant: float
    internal_edges: tuple = field(default_factory=tuple)
    crossing_edges: tuple = field(default_factory=tuple)
    field_fingerprint: str = ""
    affected_cuts: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        n = len(self.nodes)
        if not n:
            raise MeanFieldError("A local block must contain at least one node.")
        if len(set(self.nodes)) != n or not set(self.nodes) <= set(self.graph.nodes):
            raise MeanFieldError("Block nodes must be unique graph labels.")
        if not math.isfinite(self.constant):
            raise MeanFieldError("The affected constant must be finite.")
        if any(not math.isfinite(float(value)) for value in self.couplings.values()):
            raise MeanFieldError("Local couplings must be finite.")
        if any(not math.isfinite(float(value)) for value in self.fields.values()):
            raise MeanFieldError("Local fields must be finite.")
        object.__setattr__(self, "affected_cuts", self._build_affected_cuts())
        if not self.field_fingerprint:
            object.__setattr__(self, "field_fingerprint", self._fingerprint())

    def _build_affected_cuts(self):
        n = len(self.nodes)
        size = 1 << n
        indices = np.arange(size, dtype=np.uint64)
        spins = np.empty((size, n), dtype=float)
        for i in range(n):
            spins[:, i] = 1.0 - 2.0 * ((indices >> np.uint64(i)) & np.uint64(1))
        # A_S^sur = -c_aff - sum h_i x_i - sum J_ik x_i x_k.
        cuts = np.full(size, -float(self.constant), dtype=float)
        for node, value in self.fields.items():
            i = self.nodes.index(node)
            cuts -= float(value) * spins[:, i]
        for (u, v), value in self.couplings.items():
            i = self.nodes.index(u)
            j = self.nodes.index(v)
            cuts -= float(value) * spins[:, i] * spins[:, j]
        if not np.isfinite(cuts).all():
            raise MeanFieldError("The local affected diagonal is not finite.")
        return cuts

    @property
    def energy_diagonal(self):
        """Return the local Ising energy diagonal ``-A_S``."""
        return -self.affected_cuts

    @property
    def active_nodes(self):
        """Return block nodes in little-endian state order."""
        return self.nodes

    def _fingerprint(self):
        payload = {
            "graph": self.graph.snapshot().get("fingerprint"),
            "block_id": int(self.block_id),
            "nodes": [int(node) for node in self.nodes],
            "constant": float(self.constant),
            "couplings": [
                [int(u), int(v), float(value)]
                for (u, v), value in sorted(self.couplings.items())
            ],
            "fields": [
                [int(node), float(value)] for node, value in sorted(self.fields.items())
            ],
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(text.encode("ascii")).hexdigest()

    def to_dict(self):
        """Return compact JSON-compatible local coefficients and provenance."""
        return {
            "schema_version": 1,
            "block_id": int(self.block_id),
            "nodes": [int(node) for node in self.nodes],
            "couplings": [
                [int(u), int(v), float(value)]
                for (u, v), value in sorted(self.couplings.items())
            ],
            "fields": [
                [int(node), float(value)] for node, value in sorted(self.fields.items())
            ],
            "constant": float(self.constant),
            "internal_edges": [
                [int(u), int(v), float(weight)] for u, v, weight in self.internal_edges
            ],
            "crossing_edges": [
                [int(u), int(v), float(weight)] for u, v, weight in self.crossing_edges
            ],
            "field_fingerprint": self.field_fingerprint,
            "graph_fingerprint": self.graph.snapshot()["fingerprint"],
        }


def build_block_problem(graph, nodes, means, *, block_id=0):
    """Build one local problem from the current external mean snapshot.

    The active block contributes ``J=w/2`` but never contributes to its own
    fields.  Every crossing edge contributes once with the external mean.
    """
    if not isinstance(graph, WeightedGraph):
        raise TypeError("graph must be a WeightedGraph.")
    if not isinstance(means, dict):
        raise MeanFieldError("means must be a mapping.")
    if set(means) != set(graph.nodes):
        raise MeanFieldError("means must cover exactly all graph nodes.")
    for node, value in means.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not -1 <= float(value) <= 1
        ):
            raise MeanFieldError(f"Mean for node {node} must be finite in [-1, 1].")
    block = []
    for value in nodes:
        if isinstance(value, bool) or not isinstance(value, int):
            raise MeanFieldError("Block labels must be integers.")
        if value not in graph.nodes:
            raise MeanFieldError(f"Block label is not in the graph: {value}")
        if value in block:
            raise MeanFieldError(f"Duplicate block label: {value}")
        block.append(value)
    if not block:
        raise MeanFieldError("A block must not be empty.")
    active = set(block)
    couplings = {}
    fields = {node: 0.0 for node in block}
    internal_edges = []
    crossing_edges = []
    internal_sum = 0.0
    crossing_sum = 0.0
    for u, v, weight in graph.edges:
        weight = float(weight)
        inside_u = u in active
        inside_v = v in active
        if inside_u and inside_v:
            couplings[canonical_edge(u, v)] = weight / 2.0
            internal_edges.append((int(u), int(v), weight))
            internal_sum += weight
        elif inside_u:
            fields[u] += 0.5 * weight * float(means[v])
            crossing_edges.append((int(u), int(v), weight))
            crossing_sum += weight
        elif inside_v:
            fields[v] += 0.5 * weight * float(means[u])
            crossing_edges.append((int(u), int(v), weight))
            crossing_sum += weight
    constant = -0.5 * (internal_sum + crossing_sum)
    if not all(math.isfinite(value) for value in (constant, *fields.values())):
        raise MeanFieldError("Local coefficients must be finite.")
    return MeanFieldBlockProblem(
        graph=graph,
        nodes=tuple(block),
        block_id=int(block_id),
        couplings=couplings,
        fields=fields,
        constant=constant,
        internal_edges=tuple(internal_edges),
        crossing_edges=tuple(crossing_edges),
    )


def damped_moments(old_m, old_q, raw_m, raw_q, damping):
    """Return one convex moment mixture with the same damping for m and q."""
    lam = float(damping)
    if isinstance(damping, bool) or not math.isfinite(lam) or not 0 <= lam <= 1:
        raise MeanFieldError("moment damping must be finite in [0, 1].")
    result_m = {
        int(node): (1.0 - lam) * float(old_m[node]) + lam * float(raw_m[node])
        for node in old_m
    }
    result_q = {
        tuple(edge): (1.0 - lam) * float(old_q[edge]) + lam * float(raw_q[edge])
        for edge in old_q
    }
    for value in [*result_m.values(), *result_q.values()]:
        if not math.isfinite(value) or abs(value) > 1.0 + 1e-9:
            raise MeanFieldError("Damped moments are outside [-1, 1].")
    return result_m, result_q


def pair_probabilities(m_i, m_j, q_ij):
    """Return the four pair probabilities in (+,-) bit order."""
    values = [float(m_i), float(m_j), float(q_ij)]
    if any(not math.isfinite(value) or abs(value) > 1.0 + 1e-9 for value in values):
        raise MeanFieldError("Pair moments are outside [-1, 1].")
    return {
        (1, 1): (1 + m_i + m_j + q_ij) / 4.0,
        (1, -1): (1 + m_i - m_j - q_ij) / 4.0,
        (-1, 1): (1 - m_i + m_j - q_ij) / 4.0,
        (-1, -1): (1 - m_i - m_j + q_ij) / 4.0,
    }
