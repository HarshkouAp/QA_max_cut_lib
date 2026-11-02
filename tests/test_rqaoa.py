"""Independent checks for RQAOA components, coordinator, and analysis tools."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations, product
import json
import math
import subprocess
import sys

import numpy as np
import pytest
from scipy.linalg import expm

import QA_max_cut_lib
from QA_max_cut_lib import (
    ExactCPU,
    GraphFactory,
    GraphReducer,
    SolverFactory,
    WeightedGraph,
)
from QA_max_cut_lib.core.budget import ShotBudget, ShotBudgetExceeded
from QA_max_cut_lib.core.records import RunRecord
from QA_max_cut_lib.solvers.components import (
    CorrelationResult,
    MaxAbsCorrelationSelector,
    QAOAStatevectorCorrelationSource,
    SelectionResult,
)
from QA_max_cut_lib.solvers.rqaoa_cpu import RQAOACPU
from QA_max_cut_lib import tools


def direct_cut(graph, spins):
    return math.fsum(w for u, v, w in graph.edges if spins[u] != spins[v])


def source_config(layers=1, **overrides):
    values = [0.7, -0.3] if layers == 1 else [0.7, 0.2, -0.3, 0.4]
    config = {
        "type": "qaoa_statevector_cpu",
        "parameters": {"layers": layers, "max_vertices": 12, "dtype": "complex128"},
        "objective": {"type": "mean_cut", "parameters": {}},
        "objective_reference": "working_graph",
        "estimation": {"mode": "exact", "shots_per_evaluation": 16},
        "optimizer": {
            "type": "adam",
            "parameters": {"max_steps": 0},
            "gradient": {
                "type": "central_difference",
                "parameters": {"step": 0.01},
            },
        },
        "initialization": {
            "type": "fixed_angles",
            "parameters": {"values": values},
        },
        "step_initialization": "restart",
        "correlations": {"mode": "exact"},
    }
    config.update(overrides)
    return config


def root_config(**overrides):
    config = {
        "type": "rqaoa_cpu",
        "seed": 11,
        "num_treads": 1,
        "parameters": {"finish_at": 3, "max_reductions": None},
        "correlation_source": source_config(),
        "selection": {
            "type": "max_abs_correlation",
            "parameters": {"pairs": "edges", "zero_sign": 1},
        },
        "finisher": {
            "type": "exact_cpu",
            "parameters": {"max_vertices": 16, "batch_size": 65536},
        },
        "limits": {"shots_total": None},
        "collect": {
            "reduction_history": True,
            "parameter_trajectory": True,
            "local_optimizer_history": False,
            "correlation_tables": False,
            "final_reduction_state": False,
        },
        "output": {"report": 0, "logs": 0, "color": "never"},
    }
    config.update(overrides)
    return config


def reference_state(graph, angles, layers):
    n = len(graph)
    size = 1 << n
    costs = []
    for state in range(size):
        spins = {
            node: 1 - 2 * ((state >> i) & 1) for i, node in enumerate(graph.nodes)
        }
        costs.append(-direct_cut(graph, spins))
    mixer = np.zeros((size, size))
    for state in range(size):
        for qubit in range(n):
            mixer[state ^ (1 << qubit), state] += 1
    state = np.ones(size, dtype=complex) / np.sqrt(size)
    for layer in range(layers):
        state = expm(-1j * angles[layer] * np.diag(costs)) @ state
        state = expm(-1j * angles[layers + layer] * mixer) @ state
    probabilities = np.abs(state) ** 2
    correlations = np.eye(n)
    for i in range(n):
        for j in range(i):
            different = 0.0
            for index, mass in enumerate(probabilities):
                different += mass * (((index >> i) ^ (index >> j)) & 1)
            correlations[i, j] = correlations[j, i] = 1.0 - 2.0 * different
    return probabilities, correlations


@pytest.mark.parametrize("layers", [1, 2])
def test_source_correlations_match_dense_matrix_evolution(layers):
    graph = WeightedGraph(
        edges=[(7, 2, -1.5), (7, 9, 0.75), (2, 9, 2.0), (2, 4, 0.0)],
        nodes=[7, 2, 9, 4],
    )
    config = source_config(layers)
    angles = np.asarray(config["initialization"]["parameters"]["values"])
    source = QAOAStatevectorCorrelationSource(config, seed=3)
    state = GraphReducer().start(graph)
    result = source.estimate(state, budget=ShotBudget(None))
    probabilities, reference = reference_state(graph, angles, layers)
    assert result.node_order == state.current_nodes
    assert result.task_version == 0
    assert result.mode == "exact"
    assert len(result.values) == 6
    for (u, v), value in result.values.items():
        i = result.node_order.index(u)
        j = result.node_order.index(v)
        assert math.isclose(value, reference[i, j], abs_tol=2e-12)
    assert math.isclose(
        result.training["expected_cut_working"],
        float(sum(c * p for c, p in zip(
            [direct_cut(graph, {n: 1 - 2 * ((s >> i) & 1) for i, n in enumerate(graph.nodes)})
             for s in range(1 << len(graph))],
            probabilities,
        ))),
        abs_tol=2e-12,
    )


def test_source_shots_are_reproducible_and_charged_once():
    graph = WeightedGraph([(0, 1, 1.0), (1, 2, -0.5)], nodes=[0, 1, 2])
    state = GraphReducer().start(graph)
    config = source_config()
    config["correlations"] = {"mode": "shots", "shots": 1000}
    first = QAOAStatevectorCorrelationSource(config, seed=8).estimate(
        state, budget=ShotBudget(None)
    )
    second = QAOAStatevectorCorrelationSource(config, seed=8).estimate(
        state, budget=ShotBudget(None)
    )
    assert first.values == second.values
    assert all(-1 <= value <= 1 for value in first.values.values())
    budget = ShotBudget(1000)
    QAOAStatevectorCorrelationSource(config, seed=8).estimate(state, budget=budget)
    assert budget.used == 1000
    with pytest.raises(ShotBudgetExceeded):
        QAOAStatevectorCorrelationSource(config, seed=8).estimate(
            state, budget=ShotBudget(999)
        )


def correlation_table(nodes, values, version=0):
    pairs = tuple(combinations(sorted(nodes), 2))
    return CorrelationResult(
        tuple(nodes),
        version,
        {pair: values[pair] for pair in pairs},
        pairs,
        "exact",
        np.array([0.0, 0.0]),
        1,
        {},
        {},
        "working_graph",
    )


def test_selector_domains_ties_zero_sign_and_errors():
    graph = WeightedGraph([(0, 1, 1.0), (1, 2, 1.0)], nodes=[0, 1, 2])
    state = GraphReducer().start(graph)
    table = correlation_table(state.current_nodes, {(0, 1): 0.5, (0, 2): -0.5, (1, 2): 0.0})
    selector = MaxAbsCorrelationSelector(
        {"parameters": {"pairs": "all_pairs", "zero_sign": 1}}
    )
    result = selector.select(state, table)
    assert (result.removed, result.retained, result.sign) == (1, 0, 1)
    assert result.score == 0.5
    edge_selector = MaxAbsCorrelationSelector(
        {"parameters": {"pairs": "edges", "zero_sign": -1}}
    )
    edge_result = edge_selector.select(state, table)
    assert (edge_result.removed, edge_result.retained) == (1, 0)
    zero_table = correlation_table(state.current_nodes, {(0, 1): 0.0, (0, 2): 0.0, (1, 2): 0.0})
    zero = MaxAbsCorrelationSelector(
        {"parameters": {"pairs": "all_pairs", "zero_sign": -1}}
    ).select(state, zero_table)
    assert (zero.removed, zero.retained, zero.sign) == (1, 0, -1)

    stale = correlation_table(state.current_nodes, {(0, 1): 0.5, (0, 2): 0.5, (1, 2): 0.5}, 1)
    with pytest.raises(ValueError):
        selector.select(state, stale)
    bad_order = correlation_table([0, 2, 1], {(0, 1): 0.5, (0, 2): 0.5, (1, 2): 0.5})
    with pytest.raises(ValueError):
        selector.select(state, bad_order)
    missing = correlation_table(state.current_nodes, {(0, 1): 0.5, (0, 2): 0.5, (1, 2): 0.5})
    object.__setattr__(missing, "values", {(0, 1): 0.5})
    with pytest.raises(ValueError):
        selector.select(state, missing)


class FixedSource:
    type = "fixed_source"

    def __init__(self, table):
        self.table = table
        self.calls = 0

    def estimate(self, state, *, budget, pair_domain="all_pairs", **kwargs):
        self.calls += 1
        return self.table


class FixedSelector:
    type = "fixed_selector"

    def __init__(self, result):
        self.result = result

    def select(self, state, correlations):
        return self.result


def control_graph():
    return WeightedGraph(
        edges=[(10, 30, 2.0), (10, 80, 3.0), (30, 80, 5.0)],
        nodes=[10, 30, 80, 101],
    )


def test_rqaoa_control_with_injected_components_and_no_global_optimum_claim():
    graph = control_graph()
    nodes = tuple(graph.nodes)
    table = correlation_table(
        nodes,
        {(10, 30): -0.6, (10, 80): 0.1, (10, 101): 0.0,
         (30, 80): 0.2, (30, 101): 0.0, (80, 101): 0.0},
    )
    config = root_config(parameters={"finish_at": 3, "max_reductions": None})
    solver = RQAOACPU(
        config,
        correlation_source=FixedSource(table),
        selector=FixedSelector(SelectionResult(10, 30, -1, 0.6, -0.6)),
    )
    solution = solver.solve(graph)
    assert solution == {10: -1, 30: 1, 80: -1, 101: 1}
    assert solver.cut_value == 7.0
    assert solver.solution_energy == -7.0
    assert solver.cut_offset == 5.0
    assert solver.ising_constant == -6.0
    assert solver.reduced_solution == {30: 1, 80: -1, 101: 1}
    assert solver.reduced_cut_value == 2.0
    assert solver.reductions_completed == 1
    assert solver.solution_source == "exact_finisher"
    event = solver.reduction_history[0]
    assert event["fixed_energy"] == -2.0
    assert math.isclose(event["expected_fixed_energy"], -1.6)
    assert math.isclose(event["cut_offset_delta"], 5.0)
    assert event["ising_constant_delta"] == -1.0
    assert solver.cut_value != 8.0


def test_rqaoa_exact_finisher_does_not_call_source():
    graph = WeightedGraph([(2, 8, 1.0), (2, 9, -2.0), (8, 9, 0.5)])
    table = correlation_table(tuple(graph.nodes), {(2, 8): 0.0, (2, 9): 0.0, (8, 9): 0.0})
    source = FixedSource(table)
    config = root_config(parameters={"finish_at": 3, "max_reductions": None})
    solver = RQAOACPU(config, correlation_source=source)
    solution = solver.solve(graph)
    exact = ExactCPU()
    exact.solve(graph)
    assert source.calls == 0
    assert solver.solution_source == "exact_finisher"
    assert solver.reductions_completed == 0
    assert solver.cut_value == exact.cut_value


def test_rqaoa_full_actual_run_completion_and_energy():
    graph = GraphFactory.from_recipe(
        {
            "type": "complete",
            "parameters": {"n_vertices": 6},
            "seed": 15,
            "weights": {
                "type": "signed_discrete",
                "parameters": {"values": [-2, -1, 1, 2]},
                "seed": 16,
            },
        }
    )
    config = root_config(
        seed=21,
        parameters={"finish_at": 2, "max_reductions": None},
        correlation_source=source_config(),
    )
    solver = RQAOACPU(config)
    solution = solver.solve(graph)
    assert set(solution) == set(graph.nodes)
    assert all(type(spin) is int and spin in (-1, 1) for spin in solution.values())
    assert math.isclose(solver.cut_value, direct_cut(graph, solution), abs_tol=1e-12)
    assert math.isclose(
        solver.cut_value,
        solver.cut_offset + solver.reduced_cut_value,
        abs_tol=1e-9,
    )
    assert solver.solution_energy == -solver.cut_value
    assert len(solver.reduction_history) == solver.reductions_completed
    assert len(solver.parameter_trajectory) == solver.reductions_completed
    assert len(solver.remaining_node_order) <= 2


def test_rqaoa_max_reductions_fallback_is_full_and_stopped():
    graph = GraphFactory.from_recipe(
        {
            "type": "complete",
            "parameters": {"n_vertices": 5},
            "seed": 4,
            "weights": {"type": "constant", "parameters": {"value": 1.0}},
        }
    )
    source = FixedSource(correlation_table(tuple(graph.nodes), {
        (u, v): 0.0 for u, v in combinations(sorted(graph.nodes), 2)
    }))
    config = root_config(parameters={"finish_at": 1, "max_reductions": 0})
    solver = RQAOACPU(config, correlation_source=source)
    solution = solver.solve(graph)
    assert source.calls == 0
    assert solver.status == "stopped"
    assert solver.termination_reason == "max_reductions"
    assert solver.solution_source == "all_plus_reduced_fallback"
    assert solver.reductions_completed == 0
    assert set(solution) == set(graph.nodes)
    assert solver.reduction_history == []


def test_rqaoa_budget_zero_returns_full_fallback_for_shot_mode():
    graph = GraphFactory.from_recipe(
        {
            "type": "complete",
            "parameters": {"n_vertices": 4},
            "seed": 5,
            "weights": {"type": "constant", "parameters": {"value": 1.0}},
        }
    )
    shots_source = source_config()
    shots_source["estimation"] = {"mode": "shots", "shots_per_evaluation": 8}
    shots_source["correlations"] = {"mode": "shots", "shots": 16}
    config = root_config(
        parameters={"finish_at": 1, "max_reductions": None},
        correlation_source=shots_source,
        limits={"shots_total": 0},
    )
    solver = RQAOACPU(config)
    solution = solver.solve(graph)
    assert solver.status == "stopped"
    assert solver.termination_reason == "shot_budget"
    assert solver.solution_source == "all_plus_reduced_fallback"
    assert solver.resources["shots_used"] == 0
    assert set(solution) == set(graph.nodes)

    exact_config = root_config(
        parameters={"finish_at": 1, "max_reductions": None},
        correlation_source=source_config(),
        limits={"shots_total": 0},
    )
    exact = RQAOACPU(exact_config)
    exact.solve(graph)
    assert exact.status == "completed"
    assert exact.resources["shots_used"] == 0


def test_rqaoa_empty_one_vertex_zero_interaction_and_isolates():
    solver = RQAOACPU(root_config(parameters={"finish_at": 2, "max_reductions": None}))
    empty = WeightedGraph([], nodes=[])
    assert solver.solve(empty) == {}
    assert solver.cut_value == 0.0
    assert solver.solution_source == "empty_graph"

    one = WeightedGraph([], nodes=[42])
    assert solver.solve(one) == {42: 1}
    assert solver.cut_value == 0.0
    assert solver.solution_source == "zero_interaction_initial"

    zero = WeightedGraph([(0, 1, 0.0), (1, 2, 0.0)], nodes=[0, 1, 2])
    assert solver.solve(zero) == {0: 1, 1: 1, 2: 1}
    assert solver.cut_value == 0.0

    isolated = WeightedGraph([(0, 1, 2.0), (1, 2, -1.0)], nodes=[0, 1, 2, 77])
    solution = solver.solve(isolated)
    assert set(solution) == {0, 1, 2, 77}
    assert math.isclose(solver.cut_value, direct_cut(isolated, solution))
    assert solver.status == "completed"


def test_rqaoa_negative_labels_and_tie_direction():
    graph = WeightedGraph([(-10, -20, 1.0), (-20, 5, 1.0)], nodes=[-10, -20, 5])
    state = GraphReducer().start(graph)
    table = correlation_table(
        state.current_nodes,
        {(-20, -10): 0.5, (-20, 5): 0.5, (-10, 5): 0.0},
    )
    selected = MaxAbsCorrelationSelector(
        {"parameters": {"pairs": "all_pairs", "zero_sign": 1}}
    ).select(state, table)
    assert (selected.removed, selected.retained) == (-10, -20)
    assert selected.correlation == 0.5
    assert selected.score == 0.5


def test_source_threshold_shift_and_previous_step_modes():
    graph = WeightedGraph([(10, 30, 2.0), (10, 80, 3.0), (30, 80, 5.0)], nodes=[10, 30, 80])
    reducer = GraphReducer()
    state = reducer.start(graph)
    reducer.reduce(state, 10, 30, -1, correlation=-0.6)
    offset = state.cut_offset
    config = source_config()
    config["objective"] = {
        "type": "threshold_probability",
        "parameters": {"threshold": 6.0},
    }
    config["objective_reference"] = "original_graph"
    source = QAOAStatevectorCorrelationSource(config, seed=2)
    result = source.estimate(state, budget=ShotBudget(None))
    assert math.isclose(result.training["objective_threshold_requested"], 6.0)
    assert math.isclose(result.training["objective_threshold_effective"], 6.0 - offset)
    assert math.isclose(result.training["objective_shift"], offset)

    previous_config = source_config()
    previous_config["optimizer"]["parameters"]["max_steps"] = 1
    previous_config["step_initialization"] = "previous_step"
    previous = QAOAStatevectorCorrelationSource(previous_config, seed=2)
    first = previous.estimate(state, budget=ShotBudget(None))
    second = previous.estimate(state, budget=ShotBudget(None))
    assert second.training["initialization"] == "previous_step"
    assert not np.allclose(second.angles, first.angles)
    assert not np.allclose(second.angles, [0.7, -0.3])

    restart_config = source_config()
    restart_config["optimizer"]["parameters"]["max_steps"] = 1
    restart_config["step_initialization"] = "restart"
    restart = QAOAStatevectorCorrelationSource(restart_config, seed=2)
    first_restart = restart.estimate(state, budget=ShotBudget(None))
    second_restart = restart.estimate(state, budget=ShotBudget(None))
    assert second_restart.training["initialization"] == "fixed_angles"
    assert np.allclose(second_restart.angles, first_restart.angles)


def test_rqaoa_exact_correlation_budget_boundary():
    graph = WeightedGraph([(0, 1, 1.0), (1, 2, 1.0)], nodes=[0, 1, 2])
    config = root_config(parameters={"finish_at": 2, "max_reductions": None})
    config["limits"] = {"shots_total": 16}
    config["correlation_source"] = source_config()
    config["correlation_source"]["correlations"] = {"mode": "shots", "shots": 16}
    solver = RQAOACPU(config)
    solver.solve(graph)
    assert solver.status == "completed"
    assert solver.resources["shots_used"] == 16

    tight = root_config(parameters={"finish_at": 2, "max_reductions": None})
    tight["limits"] = {"shots_total": 15}
    tight["correlation_source"] = source_config()
    tight["correlation_source"]["correlations"] = {"mode": "shots", "shots": 16}
    stopped = RQAOACPU(tight)
    stopped.solve(graph)
    assert stopped.status == "stopped"
    assert stopped.termination_reason == "shot_budget"
    assert stopped.resources["shots_used"] == 0


def test_rqaoa_one_edge_sign_control():
    for weight in (1.0, -1.0):
        graph = WeightedGraph([(0, 1, weight)], nodes=[0, 1])
        source = source_config()
        source["initialization"] = {
            "type": "fixed_angles",
            "parameters": {"values": [math.pi / (2 * abs(weight)), -math.pi / 8]},
        }
        config = root_config(
            parameters={"finish_at": 1, "max_reductions": 1},
            correlation_source=source,
        )
        solver = RQAOACPU(config)
        solution = solver.solve(graph)
        event = solver.reduction_history[0]
        assert event["physical_correlation"] == (-1.0 if weight > 0 else 1.0)
        assert event["sign"] == (-1 if weight > 0 else 1)
        assert solution == {0: 1, 1: -1} if weight > 0 else solution == {0: 1, 1: 1}
        assert solver.cut_value == (weight if weight > 0 else 0.0)


def test_rqaoa_rejects_parallel_threads_and_has_no_global_angles():
    graph = WeightedGraph([(0, 1, 1.0)], nodes=[0, 1])
    with pytest.raises(ValueError):
        RQAOACPU(root_config(num_treads=2)).solve(graph)
    solver = RQAOACPU(root_config(parameters={"finish_at": 1}))
    solver.solve(graph)
    assert solver.optimal_angles is None
    assert "optimal_angles" in solver.result


def test_rqaoa_training_plus_correlation_budget_is_exact():
    graph = GraphFactory.from_recipe(
        {
            "type": "complete",
            "parameters": {"n_vertices": 4},
            "seed": 9,
            "weights": {"type": "constant", "parameters": {"value": 1.0}},
        }
    )
    config = root_config(parameters={"finish_at": 3, "max_reductions": None})
    source = source_config()
    source["estimation"] = {"mode": "shots", "shots_per_evaluation": 8}
    source["correlations"] = {"mode": "shots", "shots": 16}
    source["optimizer"]["parameters"]["max_steps"] = 1
    config["correlation_source"] = source
    config["limits"] = {"shots_total": 56}
    solver = RQAOACPU(config)
    solver.solve(graph)
    assert solver.status == "completed"
    assert solver.resources["shots_used"] == 56
    assert solver.resources["training_shots"] == 40
    assert solver.resources["correlation_shots"] == 16


def test_rqaoa_repeat_solver_keeps_previous_result_independent():
    graph_a = WeightedGraph([(0, 1, 1.0), (1, 2, 2.0)], nodes=[0, 1, 2])
    graph_b = WeightedGraph([(5, 6, -1.0)], nodes=[5, 6])
    solver = RQAOACPU(root_config(parameters={"finish_at": 1, "max_reductions": None}))
    solver.solve(graph_a)
    old_result = solver.result
    old_solution = solver.solution
    solver.solve(graph_b)
    assert set(old_solution) == set(graph_a.nodes)
    assert old_result["solution"] == {str(k): int(v) for k, v in old_solution.items()}
    assert set(solver.solution) == set(graph_b.nodes)


def test_rqaoa_output_levels_do_not_change_scientific_result(capsys):
    graph = WeightedGraph([(0, 1, 1.0), (1, 2, 2.0)], nodes=[0, 1, 2])
    quiet = RQAOACPU(root_config(parameters={"finish_at": 2}))
    quiet_solution = quiet.solve(graph, report=0, logs=0)
    capsys.readouterr()
    verbose = RQAOACPU(root_config(parameters={"finish_at": 2}))
    verbose_solution = verbose.solve(graph, report=2, logs=2)
    output = capsys.readouterr().out
    assert quiet_solution == verbose_solution
    assert quiet.resources == verbose.resources
    assert "RQAOA" in output


def test_rqaoa_save_and_json_roundtrip(tmp_path):
    graph = control_graph()
    config = root_config(parameters={"finish_at": 2, "max_reductions": None})
    solver = RQAOACPU(config)
    solver.solve(graph)
    files = tools.save_result(
        solver,
        {
            "path": str(tmp_path),
            "filename": "rqaoa",
            "atts": ["solution", "reduction_history", "parameter_trajectory"],
            "savedir": True,
            "makedir": True,
        },
    )
    loaded = tools.load_result(files.primary)
    assert loaded.solution == solver.solution
    assert loaded.reduction_history == solver.reduction_history
    assert loaded.parameter_trajectory == solver.parameter_trajectory
    with pytest.raises(FileExistsError):
        tools.save_result(
            solver,
            {
                "path": str(tmp_path),
                "filename": "rqaoa",
                "atts": ["solution"],
                "savedir": True,
            },
        )


def test_factory_and_cli_registration():
    assert "rqaoa_cpu" in SolverFactory.available()
    solver = SolverFactory.from_config(root_config())
    assert isinstance(solver, RQAOACPU)
    result = subprocess.run(
        [sys.executable, "-m", "QA_max_cut_lib.solvers.rqaoa_cpu", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "RQAOACPU" in result.stdout + result.stderr


def _trajectory_record(layers):
    steps = 2
    trajectory = []
    history = []
    for step in range(1, steps + 1):
        angles = [0.1 * step + 0.01 * index for index in range(2 * layers)]
        trajectory.append(
            {
                "reduction_step": step,
                "task_version": step - 1,
                "layers": layers,
                "angles": angles,
                "angle_names": [f"a{index}" for index in range(2 * layers)],
                "initialization": "test",
                "source_type": "test",
                "objective_reference": "working_graph",
            }
        )
        history.append(
            {
                "step": step,
                "selected_weight": float(step),
                "mean_abs_weight": None if step == 2 else 1.5,
                "fixed_energy": -1.0 * step,
                "expected_fixed_energy": None if step == 2 else -0.5 * step,
                "cut_offset": float(step),
                "ising_constant": -float(step),
            }
        )
    return RunRecord(
        {
            "schema_version": 1,
            "algorithm_type": "rqaoa_cpu",
            "package_version": QA_max_cut_lib.__version__,
            "status": "completed",
            "config": {},
            "graph": None,
            "reduction_history": history,
            "parameter_trajectory": trajectory,
        }
    )


@pytest.mark.parametrize("layers,pairs", [(1, 1), (2, 6), (3, 15)])
def test_parameter_plot_pair_counts_and_empty_history(layers, pairs):
    record = _trajectory_record(layers)
    figsize = [8, 6] if layers < 3 else [14, 10]
    fig = tools.plot_parameter_trajectories(record, {"style": {"figsize": figsize}})
    assert len(fig.axes) == pairs
    plotly_fig = tools.plot_parameter_trajectories_3d(record)
    assert len(plotly_fig.data) == pairs
    assert all(list(trace.z) == [1, 2] for trace in plotly_fig.data)
    if layers == 3:
        pages = tools.plot_parameter_trajectories(
            record, {"layout": {"panels_per_page": 6}}
        )
        assert len(pages) == 3
        assert sum(len(page.axes) for page in pages) == 15
    with pytest.raises(ValueError):
        tools.plot_parameter_trajectories(
            RunRecord({"parameter_trajectory": [], "reduction_history": []})
        )


def test_reduction_plots_and_autonomous_html(tmp_path):
    record = _trajectory_record(1)
    weights = tools.plot_reduction_weights(
        record, {"style": {"figsize": [8, 4]}}
    )
    energy = tools.plot_reduction_energy(record, {"style": {"figsize": [8, 6]}})
    assert len(weights.axes) == 2
    assert len(energy.axes) == 4
    fig = tools.plot_parameter_trajectories_3d(
        record,
        {
            "save": {
                "path": str(tmp_path),
                "filename": "traj",
                "format": "html",
                "makedir": True,
                "savedir": True,
                "metadata": True,
            }
        },
    )
    path = tmp_path / "traj" / "traj.html"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "Plotly.newPlot" in text
    assert "<script src=" not in text
    assert "cdn.plot.ly" not in text.split("topojsonURL")[0]
