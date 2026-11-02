"""Tests for fixed-spin conditional problems and Laboratory assemblies."""

import math

import numpy as np
import pytest

from QA_max_cut_lib import QAOACPU, WeightedGraph
from QA_max_cut_lib.laboratory import ConditionalQAOACPU, EnvironmentLSRQAOACPU
from QA_max_cut_lib.solvers.conditional import (
    ConditionalProblemError,
    build_conditional_problem,
)


def graph_a():
    return WeightedGraph(
        [(10, 30, 2), (10, 80, 3), (30, 80, -4), (30, 101, 5), (80, 101, -2)],
        nodes=[-7, 10, 30, 80, 101],
    )


def entries(spins):
    return [{"node": node, "spin": spin} for node, spin in spins.items()]


def standalone_config(**parameters):
    values = {"active_vertices": [10, 30], "layers": 1, "max_active_vertices": 18, "dtype": "complex128"}
    values.update(parameters)
    return {
        "schema_version": 1,
        "type": "laboratory.conditional_qaoa_cpu",
        "seed": 9,
        "parameters": {
            "active_vertices": [10, 30],
            "layers": 1,
            "max_active_vertices": 18,
            "dtype": "complex128",
            **values,
        },
        "environment": {
            "type": "fixed_spins",
            "source": {
                "type": "provided_spins",
                "parameters": {
                    "entries": entries({-7: -1, 10: 1, 30: 1, 80: 1, 101: -1})
                },
            },
        },
        "boundary": {"type": "fixed_spins_conditional"},
        "objective": {"type": "mean_cut", "parameters": {}},
        "objective_reference": "original_graph",
        "estimation": {"mode": "exact"},
        "optimizer": {
            "type": "adam",
            "parameters": {"max_steps": 0, "learning_rate": 0.05},
            "gradient": {"type": "central_difference", "parameters": {"step": 0.01}},
        },
        "initialization": {
            "type": "fixed_angles",
            "parameters": {"values": [0.4, 0.2]},
        },
        "readout": {"mode": "most_probable"},
        "limits": {"shots_total": 0},
        "collect": {"probabilities": True},
        "output": {"report": 0, "logs": 0},
    }


def ls_config():
    return {
        "schema_version": 1,
        "type": "laboratory.environment_ls_rqaoa_cpu",
        "seed": 20261102,
        "num_treads": 1,
        "parameters": {"finish_at": 3, "max_reductions": 2},
        "environment": {
            "type": "fixed_spins",
            "source": {"type": "all_plus"},
            "update_policy": "retain_survivor",
            "epoch_policy": "frozen_per_reduction",
        },
        "boundary": {"type": "fixed_spins_conditional"},
        "subgraph_builder": {
            "type": "random_partition",
            "parameters": {"max_size": 3, "rounds": 1, "full_graph_when_fits": True},
        },
        "correlation_source": {
            "type": "qaoa_ising_statevector_cpu",
            "parameters": {"layers": 1, "max_vertices": 18, "dtype": "complex128"},
            "objective": {"type": "mean_cut", "parameters": {}},
            "objective_reference": "conditional_affected_cut",
            "estimation": {"mode": "shots", "shots_per_evaluation": 8},
            "optimizer": {
                "type": "adam",
                "parameters": {"max_steps": 1, "learning_rate": 0.05},
                "gradient": {"type": "central_difference", "parameters": {"step": 0.01}},
            },
            "initialization": {
                "type": "fixed_angles",
                "parameters": {"values": [0.4, 0.2]},
            },
            "step_initialization": "restart",
            "correlations": {"mode": "shots", "shots": 16},
        },
        "aggregation": {"type": "mean", "parameters": {}},
        "selection": {
            "type": "max_abs_correlation",
            "parameters": {"pairs": "edges", "zero_sign": 1},
        },
        "coverage": {"require_full": False, "empty_policy": "anchor_pair"},
        "update": {"type": "refresh_all"},
        "fallback": {"type": "reference_assignment"},
        "execution": {"backend": "process", "start_method": "spawn", "native_threads": 1},
        "budget_policy": {"type": "step_round_robin"},
        "finisher": {"type": "exact_cpu", "parameters": {"max_vertices": 16}},
        "limits": {"shots_total": 384, "max_local_tasks_total": None},
        "collect": {
            "reduction_history": True,
            "local_task_history": True,
            "correlation_tables": True,
        },
        "output": {"report": 0, "logs": 0},
    }


def test_affected_fields_and_constants():
    problem = build_conditional_problem(
        graph_a(), [10, 30], {-7: -1, 10: 1, 30: 1, 80: 1, 101: -1}
    )
    assert problem.couplings[(10, 30)] == 1.0
    assert problem.fields == {10: 1.5, 30: -4.5}
    assert problem.constant == -3.0
    assert problem.outside_cut == -2.0
    assert problem.affected_cuts.tolist() == [5.0, 10.0, -2.0, -1.0]
    assert problem.original_cuts.tolist() == [3.0, 8.0, -4.0, -3.0]


def test_singleton_field_and_sign():
    graph = WeightedGraph([(10, 80, 2)], nodes=[10, 80])
    config = standalone_config()
    config["parameters"]["active_vertices"] = [10]
    config["initialization"]["parameters"]["values"] = [-math.pi / 4, math.pi / 4]
    config["environment"]["source"]["parameters"]["entries"] = entries({10: 1, 80: 1})
    solver = ConditionalQAOACPU(config)
    solution = solver.solve(graph)
    assert solution == {10: -1, 80: 1}
    assert solver.cut_value == 2.0
    assert solver.probabilities[-1] == pytest.approx(1.0, abs=1e-12)


def test_full_graph_matches_ordinary_qaoa():
    graph = WeightedGraph([(0, 1, 2), (1, 2, -1), (0, 2, 3)], nodes=[0, 1, 2])
    base = {
        "schema_version": 1,
        "seed": 4,
        "parameters": {"layers": 2},
        "objective": {"type": "mean_cut"},
        "estimation": {"mode": "exact"},
        "optimizer": {
            "type": "adam",
            "parameters": {"max_steps": 0},
            "gradient": {"type": "central_difference", "parameters": {"step": 0.01}},
        },
        "initialization": {"type": "fixed_angles", "parameters": {"values": [0.2, 0.3, 0.4, 0.5]}},
        "readout": {"mode": "most_probable"},
        "limits": {"shots_total": 0},
        "collect": {"probabilities": True},
        "output": {"report": 0, "logs": 0},
    }
    ordinary = QAOACPU({**base, "type": "qaoa_cpu"})
    ordinary.solve(graph)
    config = standalone_config()
    config["parameters"].update(active_vertices=[0, 1, 2], layers=2)
    config["initialization"]["parameters"]["values"] = [0.2, 0.3, 0.4, 0.5]
    config["environment"] = {"type": "fixed_spins", "source": {"type": "all_plus"}}
    conditional = ConditionalQAOACPU(config)
    conditional.solve(graph)
    assert np.allclose(conditional.probabilities, ordinary.probabilities, atol=1e-12)


def test_shifted_threshold_keeps_outside_constant():
    config = standalone_config()
    config["parameters"]["cut_offset"] = 7.0
    config["initialization"]["parameters"]["values"] = [0.0, 0.0]
    config["objective"] = {"type": "threshold_probability", "parameters": {"threshold": 12}}
    solver = ConditionalQAOACPU(config)
    solver.solve(graph_a())
    assert solver.objective_value == pytest.approx(0.25)


def test_invalid_reference_is_rejected():
    graph = graph_a()
    with pytest.raises(ConditionalProblemError):
        build_conditional_problem(graph, [10], {10: 1})


def test_environment_ls_commits_reference_and_reduces():
    graph = WeightedGraph([(10, 30, 2), (10, 80, 3), (30, 80, 5)], nodes=[10, 30, 80, 101])
    solver = EnvironmentLSRQAOACPU(ls_config())
    solution = solver.solve(graph)
    assert set(solution) == {10, 30, 80, 101}
    assert solver.result["reductions_completed"] >= 1
    assert solver.result["solution_source"] in (
        "exact_finisher",
        "all_plus_reduced_fallback",
        "reference_assignment",
    )
    assert {int(k) for k in solver.result["reference"]} == set(solver.result["final_nodes"])
    assert solver.result["cut_value"] == graph.cut_value(solution)


def test_retain_survivor_rebuilds_fields_and_original_offset():
    from QA_max_cut_lib.core.reduction import GraphReducer

    graph = WeightedGraph([(10, 30, 2), (10, 80, 3), (30, 80, 5)], nodes=[10, 30, 80, 101])
    reference = {10: 1, 30: 1, 80: 1, 101: 1}
    before = build_conditional_problem(graph, [10], reference)
    assert before.fields == {10: 2.5}
    reducer = GraphReducer()
    state = reducer.start(graph)
    reducer.reduce(state, removed=80, retained=30, sign=-1, correlation=-0.6)
    del reference[80]
    after = build_conditional_problem(
        state.current_graph(), [10], reference, cut_offset=state.cut_offset
    )
    assert after.fields == {10: -0.5}
    assert after.constant == 0.5
    assert after.original_cuts.tolist() == [8.0, 7.0]
    assert state.restore({10: 1, 30: 1, 101: 1}) == {10: 1, 30: 1, 80: -1, 101: 1}
