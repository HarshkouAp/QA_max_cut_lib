"""Structured Wishart and Hebbian weighted-graph constructions.

These constructors return a graph plus an evidence record.  A candidate
pattern is never silently promoted to a certified optimum; floating
construction, numerical checks, and exact stored-graph verification are
separate statuses.
"""

from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from ..core.graph import WeightedGraph

__all__ = [
    "build_wishart_projected",
    "build_wishart_from_factor",
    "build_hebbian_patterns",
    "default_hebbian_coefficients",
    "sample_hebbian_patterns",
]


def _labels(n, labels):
    if labels is None:
        labels = list(range(n))
    labels = [int(label) for label in labels]
    if len(labels) != n or len(set(labels)) != n:
        raise ValueError("labels must contain n unique integer values.")
    return labels


def _spins(values, n, name):
    array = np.asarray(values, dtype=float)
    if array.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},).")
    if not np.isfinite(array).all() or not np.all(np.abs(array) == 1):
        raise ValueError(f"{name} must contain only +1/-1.")
    return array


def _cut(graph, spins, labels):
    solution = {label: int(spins[index]) for index, label in enumerate(labels)}
    return float(graph.cut_value(solution))


def build_wishart_from_factor(factor, target, *, labels=None, construction_label="provided_factor"):
    """Build the off-diagonal Gram graph from an explicit factor.

    The factor must be ``N x M``.  No projection or rescaling is applied
    to a provided factor; pass an already projected matrix.
    """
    matrix = np.asarray(factor, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError("factor must be a nonempty N x M matrix.")
    n = matrix.shape[0]
    if not np.isfinite(matrix).all():
        raise ValueError("factor must be finite.")
    labels = _labels(n, labels)
    target = _spins(target, n, "target")
    gram = matrix @ matrix.T
    graph = WeightedGraph(nodes=labels)
    for i in range(n):
        for j in range(i + 1, n):
            graph.set_weight(labels[i], labels[j], float(gram[i, j]))
    candidate_cut = _cut(graph, target, labels)
    residual = float(np.linalg.norm(matrix.T @ target))
    all_cut = _cut(graph, np.ones(n), labels)
    record = {
        "type": "wishart_projected_v1",
        "construction": construction_label,
        "node_order": labels,
        "factor_shape": [int(n), int(matrix.shape[1])],
        "factor": [[float(value) for value in row] for row in matrix],
        "target": [int(value) for value in target],
        "candidate_cut": candidate_cut,
        "candidate_energy": -candidate_cut,
        "all_plus_cut": all_cut,
        "residual_norm": residual,
        "theoretical_optimum_cut": float(np.linalg.norm(matrix.T @ np.ones(n)) ** 2 / 4.0),
        "construction_theorem": "candidate_optimal_if_W^T_target=0",
        "numerically_checked": True,
        "stored_graph_optimum_verified": False,
        "status": "candidate_not_certified",
    }
    graph._recipe = {"schema_version": 2, "type": "wishart_projected_v1", "resolved": record}
    return graph, record


def build_wishart_projected(
    n_vertices,
    n_factors,
    *,
    target=None,
    factor=None,
    seed=None,
    labels=None,
    dtype="float64",
):
    """Build W=sqrt(N/(N-1))(I-tt^T/N)Z and its off-diagonal Gram graph."""
    n = int(n_vertices)
    m = int(n_factors)
    if n < 2 or m < 1:
        raise ValueError("Wishart needs N>=2 and M>=1.")
    labels = _labels(n, labels)
    if target is None:
        target = np.ones(n, dtype=float)
    target = _spins(target, n, "target")
    if factor is None:
        rng = np.random.default_rng(seed)
        z = rng.normal(size=(n, m))
        projection = np.eye(n) - np.outer(target, target) / n
        matrix = math.sqrt(n / (n - 1)) * projection @ z
    else:
        matrix = np.asarray(factor, dtype=float)
        if matrix.shape != (n, m):
            raise ValueError("provided factor must have shape (N,M).")
    return build_wishart_from_factor(matrix, target, labels=labels, construction_label="projected_v1")


def default_hebbian_coefficients(pattern_count, n_vertices):
    count = int(pattern_count)
    n_vertices = int(n_vertices)
    if count < 1 or n_vertices < 1:
        raise ValueError("pattern_count and n_vertices must be positive.")
    values = np.linspace(1.0, float(n_vertices), count).tolist()
    values[-1] = 2.0 * n_vertices
    return values


def build_hebbian_patterns(
    patterns,
    *,
    coefficients=None,
    labels=None,
    profile="canonical_v1",
):
    """Build w_ij=-(1/N)sum_mu a_mu xi_i xi_j for i!=j."""
    array = np.asarray(patterns, dtype=float)
    if array.ndim != 2 or array.shape[1] < 1:
        raise ValueError("patterns must be M x N.")
    m, n = array.shape
    if not np.isfinite(array).all() or not np.all(np.abs(array) == 1):
        raise ValueError("patterns must contain only +1/-1.")
    labels = _labels(n, labels)
    if coefficients is None:
        coefficients = default_hebbian_coefficients(m, n)
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != (m,) or not np.isfinite(coefficients).all() or np.any(coefficients <= 0):
        raise ValueError("coefficients must be positive finite values of length M.")
    gram = (array.T * coefficients) @ array / n
    graph = WeightedGraph(nodes=labels)
    for i in range(n):
        for j in range(i + 1, n):
            graph.set_weight(labels[i], labels[j], float(-gram[i, j]))
    candidates = []
    for index, pattern in enumerate(array):
        cut = _cut(graph, pattern, labels)
        candidates.append(
            {
                "pattern_id": index,
                "coefficient": float(coefficients[index]),
                "candidate_cut": cut,
                "candidate_energy": -cut,
                "optimality_status": "not_certified",
            }
        )
    record = {
        "type": "hebbian_patterns_v1",
        "profile": profile,
        "node_order": labels,
        "patterns": [[int(value) for value in row] for row in array],
        "coefficients": [float(value) for value in coefficients],
        "candidates": candidates,
        "construction_theorem": "none",
        "stored_graph_optimum_verified": False,
        "status": "candidates_not_certified",
    }
    graph._recipe = {"schema_version": 2, "type": "hebbian_patterns_v1", "resolved": record}
    return graph, record


def sample_hebbian_patterns(n_vertices, n_patterns, *, seed=None, exclude_all_minus=True):
    """Sample explicit +/-1 patterns without integer 2^N overflow."""
    n = int(n_vertices)
    m = int(n_patterns)
    if n < 1 or m < 1:
        raise ValueError("n_vertices and n_patterns must be positive.")
    rng = np.random.default_rng(seed)
    patterns = []
    for _ in range(m):
        while True:
            row = rng.integers(0, 2, size=n).astype(int) * 2 - 1
            if not exclude_all_minus or np.any(row == 1):
                patterns.append(row.tolist())
                break
    return patterns
