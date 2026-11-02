"""Independent numerical summary for the analytic p=1 implementation."""

from __future__ import annotations

from itertools import combinations
import json
import math

import numpy as np

from QA_max_cut_lib import AnalyticP1Evaluator, WeightedGraph
from QA_max_cut_lib.solvers.qaoa_state import cut_diagonal, state_at_angles


def make_graph(seed, n):
    rng = np.random.default_rng(seed)
    nodes = [3 * i - 7 for i in range(n)]
    edges = []
    for u, v in combinations(nodes, 2):
        if rng.random() < 0.55:
            weight = float(rng.choice([-2.0, -1.0, -0.3, 0.0, 0.2, 0.7, 1.5, 2.0]))
            edges.append((u, v, weight))
    return WeightedGraph(edges, nodes=nodes)


def state_correlations(graph, gamma, beta):
    probabilities = state_at_angles(cut_diagonal(graph), [gamma, beta], 1)[1]
    n = len(graph)
    indices = np.arange(1 << n, dtype=np.uint64)
    bits = ((indices[:, None] >> np.arange(n, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(float)
    signs = 1.0 - 2.0 * bits
    return (signs * probabilities[:, None]).T @ signs, probabilities


def main():
    max_correlation_error = 0.0
    max_energy_error = 0.0
    max_beta_gap = 0.0
    cases = 0
    for seed in range(20):
        graph = make_graph(seed, 2 + seed % 7)
        evaluator = AnalyticP1Evaluator(graph)
        for gamma, beta in [
            (0.0, 0.0),
            (0.37, -0.61),
            (-1.23, 0.41),
            (math.pi / 2, -math.pi / 8),
            (2.7, 1.1),
        ]:
            reference, probabilities = state_correlations(graph, gamma, beta)
            values = evaluator.correlations(gamma, beta, pair_domain="all_pairs")
            for (u, v), value in values.items():
                i, j = evaluator.node_order.index(u), evaluator.node_order.index(v)
                max_correlation_error = max(max_correlation_error, abs(value - reference[i, j]))
            energy = float(-(cut_diagonal(graph) @ probabilities))
            max_energy_error = max(
                max_energy_error,
                abs(evaluator.expected_energy(gamma, beta) - energy),
            )
            cases += 1
        for gamma in (-1.7, -0.4, 0.8, 2.2):
            profile = evaluator.energy_min_beta(gamma)
            beta_grid = np.linspace(-math.pi / 2, math.pi / 2, 2001)
            grid_min = min(
                evaluator.expected_energy(gamma, float(beta)) for beta in beta_grid
            )
            max_beta_gap = max(max_beta_gap, grid_min - profile["energy_min"])
    print(json.dumps({
        "graph_angle_cases": cases,
        "max_correlation_error": max_correlation_error,
        "max_energy_error": max_energy_error,
        "max_beta_grid_minus_exact": max_beta_gap,
    }, indent=2))


if __name__ == "__main__":
    main()
