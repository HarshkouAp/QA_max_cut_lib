"""Independent checks for analytic p=1 moments and source integration."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
import math

import numpy as np
import pytest

from QA_max_cut_lib import (
    AnalyticP1Evaluator,
    AnalyticP1CorrelationSource,
    GraphReducer,
    WeightedGraph,
    SolverFactory,
    SolverFactory as _SolverFactory,
    RQAOACPU,
)
from QA_max_cut_lib.core.budget import ShotBudget
from QA_max_cut_lib.solvers.components import (
    QAOAStatevectorCorrelationSource,
    validate_correlation_source_config,
)
from QA_max_cut_lib.solvers.qaoa_state import cut_diagonal, state_at_angles


def statevector_correlations(graph, gamma, beta):
    probabilities = state_at_angles(
        cut_diagonal(graph), [gamma, beta], 1, "complex128"
    )[1]
    n = len(graph)
    indices = np.arange(1 << n, dtype=np.uint64)
    bits = (
        (indices[:, None] >> np.arange(n, dtype=np.uint64)[None, :])
        & np.uint64(1)
    ).astype(float)
    signs = 1.0 - 2.0 * bits
    matrix = (signs * probabilities[:, None]).T @ signs
    return matrix


def direct_state_energy(graph, gamma, beta):
    cuts = cut_diagonal(graph)
    probabilities = state_at_angles(cuts, [gamma, beta], 1)[1]
    return float(-(cuts @ probabilities))


def analytic_config(**overrides):
    config = {
        "type": "qaoa_analytic_p1_cpu",
        "parameters": {"layers": 1, "dtype": "float64", "pair_batch_size": 7},
        "objective": {"type": "mean_cut", "parameters": {}},
        "objective_reference": "working_graph",
        "estimation": {"mode": "exact"},
        "optimizer": {
            "type": "gamma_grid_exact_beta",
            "parameters": {
                "gamma_bounds": [-math.pi, math.pi],
                "points": 9,
                "gamma_batch_size": 2,
            },
        },
        "correlations": {"mode": "exact"},
    }
    config.update(overrides)
    return config


def root_config(**overrides):
    config = {
        "type": "rqaoa_cpu",
        "seed": 17,
        "num_treads": 1,
        "parameters": {"finish_at": 2, "max_reductions": None},
        "correlation_source": analytic_config(),
        "selection": {
            "type": "max_abs_correlation",
            "parameters": {"pairs": "edges", "zero_sign": 1},
        },
        "finisher": {
            "type": "exact_cpu",
            "parameters": {"max_vertices": 16, "batch_size": 65536},
        },
        "limits": {"shots_total": 0},
        "collect": {
            "reduction_history": True,
            "parameter_trajectory": True,
            "local_optimizer_history": True,
            "correlation_tables": False,
            "final_reduction_state": False,
        },
        "output": {"report": 0, "logs": 0, "color": "never"},
    }
    config.update(overrides)
    return config


def random_graph(seed, n):
    rng = np.random.default_rng(seed)
    nodes = [10 + 3 * i for i in range(n)]
    edges = []
    for u, v in combinations(nodes, 2):
        if rng.random() < 0.55:
            weight = float(rng.choice([-2.0, -1.0, -0.25, 0.0, 0.5, 1.5, 2.0]))
            edges.append((u, v, weight))
    return WeightedGraph(edges, nodes=nodes)


def test_analytic_correlations_energy_and_batch_sizes():
    max_error_c = 0.0
    max_error_e = 0.0
    for seed in range(8):
        graph = random_graph(seed, 2 + seed % 4)
        evaluator = AnalyticP1Evaluator(graph, pair_batch_size=3)
        reference = statevector_correlations(graph, 0.37, -0.61)
        correlated = evaluator.correlations(0.37, -0.61, pair_domain="all_pairs")
        reference_energy = direct_state_energy(graph, 0.37, -0.61)
        max_error_e = max(
            max_error_e,
            abs(evaluator.expected_energy(0.37, -0.61) - reference_energy),
        )
        for pair, value in correlated.items():
            i = evaluator.node_order.index(pair[0])
            j = evaluator.node_order.index(pair[1])
            max_error_c = max(max_error_c, abs(value - reference[i, j]))
        small = AnalyticP1Evaluator(graph, pair_batch_size=1)
        np.testing.assert_allclose(
            list(small.correlations(0.37, -0.61).values()),
            list(correlated.values()),
            rtol=0,
            atol=2e-15,
        )
        np.testing.assert_allclose(
            evaluator.correlation_matrix(0.37, -0.61),
            reference,
            rtol=0,
            atol=2e-13,
        )
    assert max_error_c < 2e-12
    assert max_error_e < 2e-12


def test_analytic_one_edge_signs_and_weights():
    for weight in (0.25, 1.0, 2.5, -0.5, -2.0):
        graph = WeightedGraph([(0, 1, weight)], nodes=[0, 1])
        evaluator = AnalyticP1Evaluator(graph)
        gamma = math.pi / (2.0 * abs(weight))
        beta = -math.pi / 8.0
        value = evaluator.correlations(gamma, beta, pair_domain="all_pairs")[(0, 1)]
        assert math.isclose(value, -math.copysign(1.0, weight), abs_tol=2e-12)
        expected = -abs(weight) if weight > 0 else 0.0
        assert math.isclose(evaluator.expected_energy(gamma, beta), expected, abs_tol=2e-12)


def test_analytic_nonedge_correlation_and_zero_dynamics():
    graph = WeightedGraph([(0, 1, 1.0), (1, 2, 1.0)], nodes=[0, 1, 2])
    evaluator = AnalyticP1Evaluator(graph)
    value = evaluator.correlations(math.pi / 2.0, math.pi / 4.0) [(0, 2)]
    assert math.isclose(value, 1.0, abs_tol=2e-12)
    zero = evaluator.correlations(0.0, 0.0)
    assert all(math.isclose(value, 0.0, abs_tol=1e-15) for value in zero.values())
    assert math.isclose(evaluator.expected_energy(0.0, 0.0), evaluator.e0)
    empty = AnalyticP1Evaluator(WeightedGraph([], nodes=[]))
    assert empty.expected_energy(0.4, -0.2) == 0.0
    one = AnalyticP1Evaluator(WeightedGraph([], nodes=[42]))
    assert one.correlations(0.4, -0.2) == {}


def test_exact_beta_minimum_coefficients_and_dense_beta_grid():
    graph = random_graph(11, 5)
    evaluator = AnalyticP1Evaluator(graph)
    for gamma in (-2.1, -0.7, 0.0, 0.9):
        profile = evaluator.energy_min_beta(gamma)
        beta_grid = np.linspace(-math.pi / 2, math.pi / 2, 401)
        energies = [evaluator.expected_energy(gamma, float(beta)) for beta in beta_grid]
        assert min(energies) >= profile["energy_min"] - 2e-12
        assert math.isclose(
            evaluator.expected_energy(gamma, profile["beta_star"]),
            profile["energy_min"],
            abs_tol=2e-12,
        )
        y0 = evaluator.expected_energy(gamma, 3 * math.pi / 4)
        y1 = evaluator.expected_energy(gamma, 3 * math.pi / 8)
        f0 = (profile["E0"] + y0) / 2
        fc = (profile["E0"] - y0) / 2
        fs = f0 - y1
        assert math.isclose(f0, profile["F0"], abs_tol=2e-12)
        assert math.isclose(fc, profile["Fc"], abs_tol=2e-12)
        assert math.isclose(fs, profile["Fs"], abs_tol=2e-12)


def test_gamma_grid_uses_minimum_and_first_tie():
    graph = WeightedGraph([(0, 1, 1.0)], nodes=[0, 1])
    evaluator = AnalyticP1Evaluator(graph)
    scan = evaluator.scan_gamma(
        gamma_bounds=[-math.pi, math.pi], points=9, gamma_batch_size=2
    )
    assert len(scan["gamma_values"]) == 9
    assert scan["gamma_values"][0] == -math.pi
    assert scan["gamma_values"][-1] == math.pi
    manual = [
        evaluator.energy_min_beta(float(gamma))["energy_min"]
        for gamma in scan["gamma_values"]
    ]
    np.testing.assert_allclose(scan["profile_energy"], manual, atol=1e-15)
    assert scan["best_index"] == int(np.argmin(manual))
    assert math.isclose(scan["best_gamma"], float(scan["gamma_values"][scan["best_index"]]))


def test_source_result_coverage_profile_and_zero_shots():
    graph = WeightedGraph(
        [(10, 20, 1.5), (20, 30, -0.5), (10, 30, 0.0)],
        nodes=[10, 20, 30, 101],
    )
    state = GraphReducer().start(graph)
    source = AnalyticP1CorrelationSource(analytic_config(), seed=1)
    result = source.estimate(state, budget=ShotBudget(0), pair_domain="all_pairs")
    assert result.node_order == state.current_nodes
    assert result.task_version == 0
    assert result.mode == "exact"
    assert len(result.values) == 6
    assert result.resources["training_shots"] == 0
    assert result.resources["correlation_shots"] == 0
    assert result.resources["gamma_points"] == 9
    assert result.resources["pair_correlators"] == 6
    assert result.training["profile_type"] == "analytic_gamma_grid"
    assert len(result.training["gamma_profile"]["profile_energy"]) == 9
    assert result.training["objective_value"] > 0
    evaluator = AnalyticP1Evaluator(graph)
    assert math.isclose(
        result.training["expected_cut_working"],
        evaluator.expected_cut(float(result.angles[0]), float(result.angles[1])),
        abs_tol=1e-12,
    )
    edges = source.estimate(state, budget=ShotBudget(0), pair_domain="edges")
    assert len(edges.values) == 2


def test_analytic_source_above_statevector_vertex_limit():
    rng = np.random.default_rng(31)
    nodes = [100 + 2 * i for i in range(30)]
    edges = []
    for u, v in combinations(nodes, 2):
        if rng.random() < 0.15:
            edges.append((u, v, float(rng.choice([-1.5, -0.5, 0.75, 1.25]))))
    graph = WeightedGraph(edges, nodes=nodes)
    state = GraphReducer().start(graph)
    config = analytic_config()
    config["optimizer"]["parameters"]["points"] = 5
    result = AnalyticP1CorrelationSource(config).estimate(
        state, budget=ShotBudget(0), pair_domain="edges"
    )
    assert result.node_order == tuple(nodes)
    assert len(result.values) == len(graph.edges)
    assert result.resources["gamma_points"] == 5
    assert all(-1.0 <= value <= 1.0 for value in result.values.values())


def test_source_result_matches_statevector_at_selected_angles():
    graph = random_graph(4, 5)
    state = GraphReducer().start(graph)
    source = AnalyticP1CorrelationSource(analytic_config())
    result = source.estimate(state, budget=ShotBudget(0), pair_domain="all_pairs")
    gamma, beta = (float(value) for value in result.angles)
    reference = statevector_correlations(graph, gamma, beta)
    evaluator = AnalyticP1Evaluator(graph)
    for pair, value in result.values.items():
        i = evaluator.node_order.index(pair[0])
        j = evaluator.node_order.index(pair[1])
        assert math.isclose(value, reference[i, j], abs_tol=2e-11)


def test_source_original_graph_shift_and_fresh_state():
    graph = WeightedGraph([(10, 30, 2.0), (10, 80, 3.0), (30, 80, 5.0)], nodes=[10, 30, 80])
    reducer = GraphReducer()
    state = reducer.start(graph)
    reducer.reduce(state, 10, 30, -1, correlation=-0.6)
    config = analytic_config(objective_reference="original_graph")
    source = AnalyticP1CorrelationSource(config)
    result = source.estimate(state, budget=ShotBudget(0))
    assert math.isclose(result.training["objective_shift"], state.cut_offset)
    assert math.isclose(
        result.training["expected_cut_original"],
        result.training["expected_cut_working"] + state.cut_offset,
    )
    profile = result.training["gamma_profile"]
    assert all(
        math.isclose(original, working - state.cut_offset)
        for original, working in zip(
            profile["profile_energy_original"], profile["profile_energy"]
        )
    )


def test_analytic_config_rejections():
    bad = analytic_config()
    bad["parameters"]["layers"] = 2
    with pytest.raises(ValueError):
        validate_correlation_source_config(bad)
    bad = analytic_config()
    bad["estimation"] = {"mode": "shots", "shots_per_evaluation": 8}
    with pytest.raises(ValueError):
        validate_correlation_source_config(bad)
    bad = analytic_config()
    bad["correlations"] = {"mode": "shots", "shots": 8}
    with pytest.raises(ValueError):
        validate_correlation_source_config(bad)
    bad = analytic_config()
    bad["objective"] = {"type": "cvar", "parameters": {"alpha": 0.5}}
    with pytest.raises(ValueError):
        validate_correlation_source_config(bad)
    bad = analytic_config()
    bad["optimizer"]["parameters"]["points"] = 1
    with pytest.raises(ValueError):
        validate_correlation_source_config(bad)
    bad = analytic_config()
    bad["optimizer"]["parameters"]["gamma_bounds"] = [1.0, 1.0]
    with pytest.raises(ValueError):
        validate_correlation_source_config(bad)
    bad = analytic_config()
    bad["initialization"] = {"type": "fixed_angles", "parameters": {"values": [0.1, 0.2]}}
    with pytest.raises(ValueError):
        validate_correlation_source_config(bad)


def test_analytic_rqaoa_end_to_end_with_zero_shot_budget():
    graph = WeightedGraph(
        [(10, 20, 1.5), (20, 30, -0.5), (10, 30, 2.0)],
        nodes=[10, 20, 30, 101],
    )
    solver = SolverFactory.from_config(root_config())
    solution = solver.solve(graph)
    assert set(solution) == set(graph.nodes)
    assert solver.status == "completed"
    assert solver.resources["shots_used"] == 0
    assert solver.resources["gamma_points"] > 0
    assert solver.reductions_completed >= 1
    assert math.isclose(solver.cut_value, graph.cut_value(solution), abs_tol=1e-12)
    assert math.isclose(solver.cut_value, solver.cut_offset + solver.reduced_cut_value)
    assert solver.parameter_trajectory[0]["source_type"] == "qaoa_analytic_p1_cpu"
    assert solver.local_optimizer_history[0]["profile_type"] == "analytic_gamma_grid"
    figure = __import__("QA_max_cut_lib.tools", fromlist=["plot_analytic_gamma_profile"]).plot_analytic_gamma_profile(solver)
    assert len(figure.axes) == 2


def test_analytic_source_does_not_construct_statevector(monkeypatch):
    import QA_max_cut_lib.solvers.components as components

    def forbidden(*args, **kwargs):
        raise AssertionError("statevector source must not be constructed")

    monkeypatch.setattr(components, "QAOAStatevectorCorrelationSource", forbidden)
    graph = WeightedGraph([(0, 1, 1.0), (1, 2, -0.5)], nodes=[0, 1, 2])
    solver = SolverFactory.from_config(root_config())
    solver.solve(graph)
    assert solver.status == "completed"
