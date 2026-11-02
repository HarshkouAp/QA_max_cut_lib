"""Real GPU numerical checks; skipped on CPU-only installations."""

import math

import os

import numpy as np
import pytest

REQUIRE_GPU = os.environ.get("QA_MAX_CUT_REQUIRE_GPU") == "1"
if REQUIRE_GPU:
    import cupy  # noqa: F401
else:
    pytest.importorskip("cupy")

from QA_max_cut_lib import WeightedGraph
from QA_max_cut_lib.solvers.components import QAOAStatevectorCorrelationSource
from QA_max_cut_lib.solvers.gpu_state import (
    adjoint_mean_cut_gradient_gpu,
    correlations_gpu,
    cut_diagonal_gpu,
    memory_preflight,
    objective_value_gpu,
    restore_full_gpu,
    sample_full_indices_gpu,
    state_at_angles_gpu,
)
from QA_max_cut_lib.solvers.qaoa_gpu import QAOAGPU, QAOAGPUCorrelationSource
from QA_max_cut_lib.solvers.qaoa_state import cut_diagonal, state_at_angles
from QA_max_cut_lib.core.budget import ShotBudget
from QA_max_cut_lib.core.reduction import GraphReducer

pytestmark = pytest.mark.gpu


def graph():
    return WeightedGraph(
        [(0, 1, 2), (1, 2, -1), (2, 3, 3), (0, 3, 1)],
        nodes=[0, 1, 2, 3],
    )


def gpu_section():
    return {
        "device_id": 0,
        "diagonal_dtype": "float64",
        "accumulation_dtype": "float64",
        "memory": {"fraction_of_available": 0.8, "reserve_bytes": 268435456},
        "kernels": {"mixer": "pairwise", "cuda_graphs": False},
    }


def solver_config(representation="full", **overrides):
    config = {
        "schema_version": 1,
        "type": "qaoa_gpu",
        "seed": 4,
        "num_treads": 1,
        "parameters": {
            "layers": 2,
            "max_vertices": 10,
            "dtype": "complex128",
            "representation": representation,
        },
        "gpu": gpu_section(),
        "objective": {"type": "mean_cut", "parameters": {}},
        "estimation": {"mode": "exact"},
        "optimizer": {
            "type": "adam",
            "parameters": {"max_steps": 2},
            "gradient": {"type": "central_difference", "parameters": {"step": 0.01}},
        },
        "initialization": {
            "type": "fixed_angles",
            "parameters": {"values": [0.3, 0.2, 0.1, 0.4]},
        },
        "readout": {"mode": "most_probable"},
        "limits": {"shots_total": 0},
        "collect": {"correlations": True, "optimizer_history": True},
    }
    config.update(overrides)
    return config


def test_gpu_full_and_pairs_match_cpu():
    import cupy as cp

    graph_ = graph()
    cuts_cpu = cut_diagonal(graph_)
    for representation in ("full", "spin_flip_pairs"):
        cuts_gpu = cut_diagonal_gpu(graph_, representation)
        state, masses = state_at_angles_gpu(
            cuts_gpu,
            [0.3, 0.2, 0.1, 0.4],
            2,
            n_vertices=4,
            representation=representation,
            dtype="complex128",
        )
        cpu_state, cpu_masses = state_at_angles(
            cuts_cpu, [0.3, 0.2, 0.1, 0.4], 2, "complex128"
        )
        if representation == "full":
            assert float(cp.max(cp.abs(state - cp.asarray(cpu_state)))) < 1e-11
            assert float(cp.max(cp.abs(masses - cp.asarray(cpu_masses)))) < 1e-11
        else:
            assert float(cp.abs(masses.sum() - 1)) < 1e-11
            restored = restore_full_gpu(state, 4)
            assert float(cp.max(cp.abs(restored - cp.asarray(cpu_state)))) < 1e-11


def test_gpu_correlations_match_cpu_probabilities():
    import cupy as cp

    graph_ = graph()
    cuts = cut_diagonal_gpu(graph_, "spin_flip_pairs")
    state, masses = state_at_angles_gpu(
        cuts,
        [0.3, 0.2],
        1,
        n_vertices=4,
        representation="spin_flip_pairs",
        dtype="complex128",
    )
    matrix = correlations_gpu(masses, 4, "spin_flip_pairs")
    cpu_state, cpu_prob = state_at_angles(cut_diagonal(graph_), [0.3, 0.2], 1)
    indices = np.arange(2**4)
    for i in range(4):
        for j in range(4):
            if i == j:
                expected = 1.0
            else:
                bits_i = (indices >> i) & 1
                bits_j = (indices >> j) & 1
                expected = float(
                    (((1 - 2 * bits_i) * (1 - 2 * bits_j)) @ cpu_prob)
                )
            assert abs(float(cp.asnumpy(matrix[i, j])) - expected) < 1e-11


def test_gpu_all_five_criteria_match_cpu():
    import cupy as cp

    from QA_max_cut_lib.solvers.objectives import objective_value, validate_objective

    cuts = np.array([-2.0, 0.0, 1.0, 3.0])
    masses = np.array([0.1, 0.2, 0.3, 0.4])
    for raw in (
        {"type": "mean_cut", "parameters": {}},
        {"type": "cvar", "parameters": {"alpha": 0.5}},
        {"type": "gibbs", "parameters": {"eta": 1.3}},
        {"type": "best_of_k", "parameters": {"k": 3}},
        {"type": "threshold_probability", "parameters": {"threshold": 1.0}},
    ):
        config = validate_objective(raw)
        expected = objective_value(cuts, masses, config)
        actual = objective_value_gpu(cp.asarray(cuts), cp.asarray(masses), config)
        assert abs(actual - expected) < 1e-12


def test_gpu_sampler_preserves_both_flip_orientations():
    import cupy as cp

    graph_ = graph()
    cuts = cut_diagonal_gpu(graph_, "spin_flip_pairs")
    _, masses = state_at_angles_gpu(
        cuts, [0.3, 0.2], 1, n_vertices=4, representation="spin_flip_pairs"
    )
    rng = cp.random.RandomState(7)
    sampled = sample_full_indices_gpu(
        masses, 20000, n_vertices=4, representation="spin_flip_pairs", rng=rng
    )
    counts = cp.bincount(sampled, minlength=16).astype(cp.float64) / 20000
    expected = cp.concatenate([masses / 2, masses[::-1] / 2])
    assert float(cp.max(cp.abs(counts - expected))) < 0.02


def test_adjoint_matches_central_difference():
    import cupy as cp

    graph_ = graph()
    theta = np.array([0.3, 0.2, 0.1, 0.4])
    for representation in ("full", "spin_flip_pairs"):
        cuts_gpu = cut_diagonal_gpu(graph_, representation)
        cuts_cpu = cut_diagonal(graph_)
        cuts = cuts_cpu if representation == "full" else cuts_cpu[: len(cuts_cpu) // 2]
        analytic = adjoint_mean_cut_gradient_gpu(
            cuts_gpu,
            theta,
            2,
            n_vertices=4,
            representation=representation,
            dtype="complex128",
        )
        numeric = []
        for index in range(4):
            delta = np.zeros(4)
            delta[index] = 1e-6
            def value(angles):
                state, masses = state_at_angles_gpu(
                    cuts_gpu,
                    angles,
                    2,
                    n_vertices=4,
                    representation=representation,
                    dtype="complex128",
                )
                return float(cp.asnumpy(cp.sum(cp.asarray(cuts) * masses)))
            numeric.append((value(theta + delta) - value(theta - delta)) / 2e-6)
        assert np.max(np.abs(analytic + np.asarray(numeric))) < 1e-8


def test_qaoa_gpu_solver_and_readout():
    graph_ = graph()
    result = QAOAGPU(solver_config()).solve(graph_, report=0, logs=0)
    solver = QAOAGPU(solver_config())
    solution = solver.solve(graph_, report=0, logs=0)
    assert set(solution) == set(graph_.nodes)
    assert solver.status == "completed"
    assert solver.resources["shots_used"] == 0
    assert solver.state_representation["type"] == "full"
    assert solver.correlations.shape == (4, 4)

    shots = solver_config(
        readout={"mode": "best_of_shots", "shots": 32},
        limits={"shots_total": 32},
        collect={"correlations": False, "optimizer_history": True},
        representation="spin_flip_pairs",
    )
    shot_solver = QAOAGPU(shots)
    shot_solution = shot_solver.solve(graph_, report=0, logs=0)
    assert set(shot_solution) == set(graph_.nodes)
    assert shot_solver.resources["shots_used"] == 32


def test_gpu_source_and_composite_solvers():
    graph_ = graph()
    source = {
        "type": "qaoa_statevector_gpu",
        "parameters": {
            "layers": 1,
            "max_vertices": 8,
            "dtype": "complex128",
            "representation": "spin_flip_pairs",
        },
        "gpu": gpu_section(),
        "objective": {"type": "mean_cut", "parameters": {}},
        "objective_reference": "working_graph",
        "estimation": {"mode": "exact"},
        "optimizer": {
            "type": "adam",
            "parameters": {"max_steps": 1},
            "gradient": {"type": "central_difference", "parameters": {"step": 0.01}},
        },
        "initialization": {
            "type": "fixed_angles",
            "parameters": {"values": [0.3, 0.2]},
        },
        "correlations": {"mode": "exact"},
    }
    state = GraphReducer().start(graph_)
    result = QAOAGPUCorrelationSource(source).estimate(
        state, budget=ShotBudget(0), pair_domain="all_pairs"
    )
    assert result.node_order == tuple(graph_.nodes)
    assert result.resources["shots_used"] == 0
    assert len(result.values) == 6

    from QA_max_cut_lib.solvers.rqaoa_gpu import RQAOAGPU

    rq = RQAOAGPU(
        {
            "type": "rqaoa_gpu",
            "seed": 4,
            "parameters": {"finish_at": 2},
            "correlation_source": source,
            "selection": {"type": "max_abs_correlation", "parameters": {"pairs": "edges"}},
            "finisher": {"type": "exact_cpu", "parameters": {"max_vertices": 4}},
            "limits": {"shots_total": 0},
            "collect": {"reduction_history": True},
        }
    )
    solution = rq.solve(graph_, report=0, logs=0)
    assert set(solution) == set(graph_.nodes)
    assert rq.status == "completed"


def test_ls_gpu_owner_uses_one_sequence():
    graph_ = graph()
    source = {
        "type": "qaoa_statevector_gpu",
        "parameters": {
            "layers": 1,
            "max_vertices": 6,
            "dtype": "complex64",
            "representation": "spin_flip_pairs",
        },
        "gpu": gpu_section(),
        "objective": {"type": "mean_cut", "parameters": {}},
        "objective_reference": "working_graph",
        "estimation": {"mode": "exact"},
        "optimizer": {"type": "adam", "parameters": {"max_steps": 0}},
        "initialization": {"type": "fixed_angles", "parameters": {"values": [0.3, 0.2]}},
        "correlations": {"mode": "exact"},
    }
    from QA_max_cut_lib.solvers.ls_rqaoa_gpu import LSRQAOAGPU

    config = {
        "type": "ls_rqaoa_gpu",
        "seed": 4,
        "num_treads": 2,
        "parameters": {"finish_at": 2},
        "subgraph_builder": {
            "type": "greedy_abs_weight_partition",
            "parameters": {"max_size": 3, "rounds": 1, "full_graph_when_fits": False},
        },
        "correlation_source": source,
        "aggregation": {"type": "mean", "parameters": {}},
        "selection": {"type": "max_abs_correlation", "parameters": {"pairs": "edges"}},
        "coverage": {"require_full": False, "empty_policy": "anchor_pair"},
        "update": {"type": "refresh_all"},
        "execution": {"backend": "gpu_owner", "start_method": "spawn", "native_threads": 1},
        "budget_policy": {"type": "step_round_robin"},
        "finisher": {"type": "exact_cpu", "parameters": {"max_vertices": 4}},
        "limits": {"shots_total": 0},
        "collect": {"local_task_history": True, "reduction_history": True},
    }
    solver = LSRQAOAGPU(config)
    solution = solver.solve(graph_, report=0, logs=0)
    assert set(solution) == set(graph_.nodes)
    assert solver.resources["num_treads_applied"] == 1
    assert solver.resources["backend"] == "gpu_owner"
    assert solver.resources["local_tasks_total"] > 0


def test_gpu_hierarchy_uses_one_owner():
    graph_ = WeightedGraph(
        [
            (0, 1, 1),
            (1, 2, -2),
            (2, 3, 2),
            (3, 4, 1),
            (4, 5, -1),
            (5, 6, 2),
            (6, 7, 1),
            (0, 7, -2),
            (0, 3, 1),
        ],
        nodes=list(range(8)),
    )
    local = {
        "type": "qaoa_gpu",
        "parameters": {
            "layers": 1,
            "max_vertices": 2,
            "dtype": "complex128",
            "representation": "spin_flip_pairs",
        },
        "gpu": gpu_section(),
        "objective": {"type": "mean_cut", "parameters": {}},
        "estimation": {"mode": "exact"},
        "optimizer": {
            "type": "adam",
            "parameters": {"max_steps": 1},
            "gradient": {"type": "central_difference", "parameters": {"step": 0.01}},
        },
        "initialization": {
            "type": "fixed_angles",
            "parameters": {"values": [0.7, -0.3]},
        },
        "readout": {"mode": "most_probable"},
    }
    from QA_max_cut_lib.solvers.qaoa_in_qaoa_gpu import QAOAInQAOAGPU

    config = {
        "type": "qaoa_in_qaoa_gpu",
        "seed": 5,
        "num_treads": 2,
        "parameters": {"finish_at": 2, "max_levels": 4},
        "partitioner": {
            "type": "greedy_abs_weight_partition",
            "parameters": {
                "max_size": 2,
                "rounds": 1,
                "full_graph_when_fits": False,
            },
        },
        "local_solver": local,
        "execution": {
            "backend": "gpu_owner",
            "start_method": "spawn",
            "native_threads": 1,
        },
        "budget_policy": {"type": "level_round_robin"},
        "finisher": {"type": "exact_cpu", "parameters": {"max_vertices": 4}},
        "limits": {"shots_total": 0},
        "collect": {"hierarchy_history": True},
    }
    solver = QAOAInQAOAGPU(config)
    solution = solver.solve(graph_, report=0, logs=0)
    assert set(solution) == set(graph_.nodes)
    assert solver.levels_completed >= 1
    assert solver.resources["backend"] == "cupy"
    assert solver.resources["num_treads_applied"] == 1
    assert solver.resources["local_solver_calls"] > 0


def test_memory_preflight_rejects_impossible_full_state():
    with pytest.raises(MemoryError):
        memory_preflight(gpu_section(), 30, "full", dtype="complex128")
