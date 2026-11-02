"""Tests for block mean-field QAOA and message updates."""

import math

import numpy as np
import pytest

from QA_max_cut_lib import WeightedGraph
from QA_max_cut_lib.laboratory import BlockMeanFieldQAOACPU
from QA_max_cut_lib.solvers.block_mean_field import (
    MeanFieldError,
    build_block_problem,
    damped_moments,
    pair_probabilities,
)


def base_config(**changes):
    config = {
        "schema_version": 1,
        "type": "laboratory.block_mean_field_qaoa_cpu",
        "seed": 7,
        "num_treads": 1,
        "parameters": {
            "layers": 1,
            "max_sweeps": 0,
            "max_block_vertices": 8,
            "dtype": "complex128",
        },
        "partition": {
            "type": "explicit_partition",
            "parameters": {"blocks": [[0], [1]]},
        },
        "initial_messages": {
            "type": "provided_means",
            "parameters": {
                "entries": [
                    {"node": 0, "mean": 0.2},
                    {"node": 1, "mean": -0.4},
                ]
            },
        },
        "initialization": {
            "type": "fixed_angles",
            "parameters": {"values": [0.0, 0.0]},
        },
        "schedule": {"type": "jacobi", "block_order": "as_given"},
        "updates": {
            "moment_damping": 0.5,
            "angle_damping": 1.0,
            "acceptance": "always",
            "optimizer_state_policy": "restart_each_update",
        },
        "local_solver": {
            "type": "qaoa_ising_statevector_cpu",
            "objective": {"type": "mean_cut", "parameters": {}},
            "objective_reference": "local_surrogate",
            "estimation": {"mode": "exact"},
            "optimizer": {
                "type": "adam",
                "parameters": {"max_steps": 0},
                "gradient": {"type": "central_difference", "parameters": {"step": 0.01}},
            },
        },
        "moments": {
            "mode": "exact",
            "statistics": ["z_all", "zz_internal_edges"],
            "shared_sample": True,
        },
        "rounding": {"type": "sign_means", "zero_sign": 1},
        "stopping": {"type": "max_sweeps"},
        "execution": {
            "backend": "process",
            "start_method": "spawn",
            "native_threads": 1,
        },
        "budget_policy": {"type": "sweep_round_robin"},
        "limits": {"shots_total": 0, "max_block_updates_total": 0},
        "checkpoint": {"enabled": False},
        "collect": {},
        "output": {"report": 0, "logs": 0},
    }
    config.update(changes)
    return config


def test_model_cut_vs_rounding_and_double_counting():
    graph = WeightedGraph([(0, 1, 2)], nodes=[0, 1])
    solver = BlockMeanFieldQAOACPU(base_config())
    solution = solver.solve(graph)
    assert solution == {0: 1, 1: -1}
    assert solver.result["model_mean_cut"] == pytest.approx(1.08)
    assert solver.cut_value == 2.0
    # Two local affected expectations are not the global model cut.
    problem0 = build_block_problem(graph, [0], {0: 0.2, 1: -0.4})
    problem1 = build_block_problem(graph, [1], {0: 0.2, 1: -0.4})
    local_mass0 = np.array([(1 + 0.2) / 2, (1 - 0.2) / 2])
    local_mass1 = np.array([(1 - 0.4) / 2, (1 + 0.4) / 2])
    assert float(problem0.affected_cuts @ local_mass0) == pytest.approx(1.08)
    assert float(problem1.affected_cuts @ local_mass1) == pytest.approx(1.08)


def test_consistent_moment_damping_and_pair_check():
    means, q = damped_moments(
        {0: 1.0, 1: 1.0},
        {(0, 1): 1.0},
        {0: -1.0, 1: 1.0},
        {(0, 1): -1.0},
        0.5,
    )
    assert means == {0: 0.0, 1: 1.0}
    assert q[(0, 1)] == 0.0
    probs = pair_probabilities(means[0], means[1], q[(0, 1)])
    assert probs[(1, 1)] == pytest.approx(0.5)
    assert probs[(-1, 1)] == pytest.approx(0.5)
    # Taking raw q with damped m is not a physical pair distribution.
    invalid = pair_probabilities(0.0, 1.0, -1.0)
    assert min(invalid.values()) < 0


def test_applied_angle_moments_and_single_spin_field():
    graph = WeightedGraph([(0, 1, 2)], nodes=[0, 1])
    problem = build_block_problem(graph, [0], {0: 0.0, 1: 1.0})
    solver = BlockMeanFieldQAOACPU.__new__(BlockMeanFieldQAOACPU)
    z, q, probabilities, origin = solver._sample_moments(
        problem,
        [-math.pi / 16, math.pi / 16],
        {"parameters": {"layers": 1, "dtype": "complex128"}},
        np.random.default_rng(0),
        shots=0,
    )
    expected = -(2.0 - math.sqrt(2.0)) / 4.0
    assert origin == "exact"
    assert z[0] == pytest.approx(expected, abs=1e-12)
    assert q == {}
    assert probabilities.sum() == pytest.approx(1.0)


def test_zero_sweeps_and_full_rounding():
    graph = WeightedGraph([(0, 1, 2)], nodes=[0, 1])
    solver = BlockMeanFieldQAOACPU(base_config())
    solution = solver.solve(graph)
    assert solver.result["completed_sweeps"] == 0
    assert solver.result["accepted_updates"] == 0
    assert solver.result["termination_reason"] == "zero_sweeps"
    assert solver.result["model_mean_cut"] == pytest.approx(1.08)
    assert all(spin in (-1, 1) for spin in solution.values())


def test_empty_graph_returns_empty_mapping():
    graph = WeightedGraph([], nodes=[])
    config = base_config()
    config["partition"]["parameters"]["blocks"] = []
    config["initial_messages"] = {"type": "zero", "parameters": {}}
    solver = BlockMeanFieldQAOACPU(config)
    assert solver.solve(graph) == {}
    assert solver.result["termination_reason"] == "empty_graph"
    assert solver.result["resources"]["shots_used"] == 0


def test_partition_overlap_is_rejected():
    graph = WeightedGraph([(0, 1, 1)], nodes=[0, 1])
    config = base_config()
    config["partition"]["parameters"]["blocks"] = [[0, 1], [1]]
    solver = BlockMeanFieldQAOACPU(config)
    with pytest.raises(ValueError):
        solver.solve(graph)


def test_mandatory_moment_reserve_blocks_sweep():
    graph = WeightedGraph([(0, 1, 2), (1, 2, 1)], nodes=[0, 1, 2])
    config = base_config()
    config["partition"]["parameters"]["blocks"] = [[0], [1], [2]]
    config["initial_messages"]["parameters"]["entries"] = [
        {"node": 0, "mean": 0.2},
        {"node": 1, "mean": -0.4},
        {"node": 2, "mean": 0.0},
    ]
    config["parameters"]["max_sweeps"] = 1
    config["moments"] = {
        "mode": "shots",
        "shots": 16,
        "statistics": ["z_all", "zz_internal_edges"],
        "shared_sample": True,
    }
    config["local_solver"]["estimation"] = {"mode": "shots", "shots_per_evaluation": 8}
    config["local_solver"]["optimizer"]["parameters"]["max_steps"] = 2
    config["limits"] = {"shots_total": 47, "max_block_updates_total": 9}
    config["collect"] = {"message_history": True}
    solver = BlockMeanFieldQAOACPU(config)
    solver.solve(graph)
    assert solver.result["status"] == "stopped"
    assert solver.result["termination_reason"] == "moment_reserve_unavailable"
    assert solver.result["resources"]["shots_used"] == 0

    config["limits"]["shots_total"] = 48
    solver = BlockMeanFieldQAOACPU(config)
    solver.solve(graph)
    assert solver.result["completed_sweeps"] == 1
    assert solver.result["completed_local_jobs"] == 3
    assert solver.result["resources"]["shots_used"] == 48
    assert solver.result["resources"]["training_shots"] == 0


def test_jacobi_does_not_commit_partial_batch():
    graph = WeightedGraph([(0, 1, 2), (1, 2, 1)], nodes=[0, 1, 2])
    config = base_config()
    config["partition"]["parameters"]["blocks"] = [[0], [1], [2]]
    config["initial_messages"]["parameters"]["entries"] = [
        {"node": 0, "mean": 0.2},
        {"node": 1, "mean": -0.4},
        {"node": 2, "mean": 0.0},
    ]
    config["parameters"]["max_sweeps"] = 1
    config["moments"] = {
        "mode": "shots",
        "shots": 16,
        "statistics": ["z_all", "zz_internal_edges"],
        "shared_sample": True,
    }
    config["local_solver"]["estimation"] = {"mode": "shots", "shots_per_evaluation": 8}
    config["local_solver"]["optimizer"]["parameters"]["max_steps"] = 2
    config["limits"] = {"shots_total": 128, "max_block_updates_total": 2}
    solver = BlockMeanFieldQAOACPU(config)
    solver.solve(graph)
    # The remaining allowance cannot cover a complete logical sweep.
    assert solver.result["status"] == "stopped"
    assert solver.result["termination_reason"] == "block_update_limit"
    assert solver.result["completed_sweeps"] == 0
    assert solver.result["partial_sweep"] is False
    assert solver.result["pending_batch"] is False
    assert solver.result["means"] == {"0": 0.2, "1": -0.4, "2": 0.0}


def test_checkpoint_resume_at_sweep_boundary(tmp_path):
    from QA_max_cut_lib import restore

    graph = WeightedGraph([(0, 1, 2), (1, 2, 1)], nodes=[0, 1, 2])
    config = base_config()
    config["partition"]["parameters"]["blocks"] = [[0, 1], [2]]
    config["initial_messages"] = {
        "type": "biased_spins",
        "parameters": {"amplitude": 0.1, "spins": {"type": "all_plus"}},
    }
    config["parameters"]["max_sweeps"] = 2
    config["limits"] = {"shots_total": 192, "max_block_updates_total": 4}
    config["checkpoint"] = {
        "enabled": True,
        "path": str(tmp_path / "cp"),
        "every": {"optimizer_iterations": 1},
        "durable_resource_journal": True,
    }
    config["collect"] = {"message_history": True, "model_history": True}
    solver = BlockMeanFieldQAOACPU(config)
    solver.solve(graph)
    assert solver.result["completed_sweeps"] == 2
    restored = restore(solver.last_checkpoint.path)
    solution = restored.resume()
    assert set(solution) == {0, 1, 2}
    assert restored.result["completed_sweeps"] == 2
    assert restored.result["status"] == "completed"


def test_zero_start_symmetry_keeps_first_moments():
    graph = WeightedGraph([(0, 1, 2)], nodes=[0, 1])
    config = base_config()
    config["partition"]["parameters"]["blocks"] = [[0, 1]]
    config["initial_messages"] = {"type": "zero", "parameters": {}}
    config["parameters"]["max_sweeps"] = 1
    config["local_solver"]["estimation"] = {"mode": "exact"}
    config["moments"] = {
        "mode": "exact",
        "statistics": ["z_all", "zz_internal_edges"],
        "shared_sample": True,
    }
    config["limits"] = {"shots_total": 0, "max_block_updates_total": 1}
    config["initialization"]["parameters"]["values"] = [0.3, 0.2]
    solver = BlockMeanFieldQAOACPU(config)
    solution = solver.solve(graph)
    assert solver.result["means"] == {"0": 0.0, "1": 0.0}
    assert solution == {0: 1, 1: 1}
    assert solver.result["completed_sweeps"] == 1


def test_single_full_block_uses_full_local_qaoa():
    graph = WeightedGraph([(0, 1, 2), (1, 2, -1)], nodes=[0, 1, 2])
    config = base_config()
    config["partition"]["parameters"]["blocks"] = [[0, 1, 2]]
    config["initial_messages"]["parameters"]["entries"].append({"node": 2, "mean": 0.0})
    config["parameters"]["max_sweeps"] = 1
    config["local_solver"]["estimation"] = {"mode": "exact"}
    config["moments"] = {
        "mode": "exact",
        "statistics": ["z_all", "zz_internal_edges"],
        "shared_sample": True,
    }
    config["limits"] = {"shots_total": 0, "max_block_updates_total": 1}
    solver = BlockMeanFieldQAOACPU(config)
    solution = solver.solve(graph)
    assert set(solution) == {0, 1, 2}
    assert solver.result["completed_sweeps"] == 1
    assert solver.cut_value == graph.cut_value(solution)
    assert solver.result["model_mean_cut"] != pytest.approx(
        graph.cut_value(solution), abs=1e-9
    )
