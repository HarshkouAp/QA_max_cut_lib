"""Real classical-baseline and refinement checks."""

from __future__ import annotations

import math

import numpy as np
import pytest

from QA_max_cut_lib import (
    GoemansWilliamsonCPU,
    ExactBlockRefiner,
    RefinedSolver,
    SimulatedAnnealingCPU,
    WeightedGraph,
    tools,
)


def sa_config(**overrides):
    config = {
        "type": "simulated_annealing_cpu",
        "seed": 42,
        "num_treads": 1,
        "parameters": {
            "num_reads": 3,
            "num_sweeps": 16,
            "beta_schedule_type": "geometric",
            "beta_range": [0.1, 4.0],
        },
        "limits": {"anneal_reads_total": 3, "anneal_sweeps_total": 48},
        "collect": {"candidate_history": True},
    }
    config.update(overrides)
    return config


def refiner_config(blocks, **params):
    p = {
        "max_size": 8,
        "batch_size": 4,
        "max_passes": 1,
        "max_blocks_total": None,
        "improvement_atol": 1e-10,
        "improvement_rtol": 1e-12,
    }
    p.update(params)
    return {
        "type": "exact_block",
        "parameters": p,
        "selector": {
            "type": "explicit_blocks",
            "parameters": {"blocks": blocks},
        },
    }


def test_SA_sign_offset_isolate_and_complete_solution():
    positive = WeightedGraph([(10, 30, 2)], nodes=[10, 30, 80])
    solver = SimulatedAnnealingCPU(sa_config())
    solution = solver.solve(positive, report=0, logs=0)
    assert set(solution) == {10, 30, 80}
    assert solver.bqm_offset == -1.0
    assert solution[80] in (-1, 1)
    assert solver.cut_value == positive.cut_value(solution)
    assert solver.num_reads_completed == 3
    assert len(solver.seed_plan) == 3
    assert solver.candidate_history is not None
    assert all(item["bqm_energy"] == -item["cut_value"] for item in solver.candidate_history)

    negative = WeightedGraph([(10, 30, -2)], nodes=[10, 30, 80])
    solver = SimulatedAnnealingCPU(sa_config(seed=43))
    solution = solver.solve(negative, report=0, logs=0)
    assert solver.bqm_offset == 1.0
    assert set(solution) == {10, 30, 80}
    assert solver.cut_value == negative.cut_value(solution)


def test_SA_read_and_sweep_limits_are_real():
    graph = WeightedGraph([(0, 1, 1)], nodes=[0, 1, 2])
    solver = SimulatedAnnealingCPU(
        sa_config(
            parameters={
                "num_reads": 10,
                "num_sweeps": 64,
                "beta_schedule_type": "linear",
            },
            limits={"anneal_reads_total": 10, "anneal_sweeps_total": 192},
        )
    )
    solution = solver.solve(graph, report=0, logs=0)
    assert set(solution) == {0, 1, 2}
    assert solver.num_reads_completed == 3
    assert solver.num_sweeps_accounted == 192
    assert solver.status == "stopped"
    assert solver.termination_reason == "anneal_sweeps_total"


def test_SA_threads_use_same_seed_plan_and_same_logical_reads():
    graph = WeightedGraph([(0, 1, 1), (1, 2, -2)], nodes=[0, 1, 2])
    first = SimulatedAnnealingCPU(sa_config(num_treads=1))
    second = SimulatedAnnealingCPU(sa_config(num_treads=2))
    first.solve(graph, report=0, logs=0)
    second.solve(graph, report=0, logs=0)
    assert first.seed_plan == second.seed_plan
    assert first.cut_value == second.cut_value
    assert first.num_reads_completed == second.num_reads_completed == 3


def test_GW_real_sdp_triangle_and_isolated_node():
    graph = WeightedGraph(
        [(10, 30, 1), (10, 80, 1), (30, 80, 1)],
        nodes=[10, 30, 80],
    )
    solver = GoemansWilliamsonCPU(
        {
            "type": "gw_cpu",
            "seed": 7,
            "parameters": {"num_hyperplanes": 8},
            "collect": {"candidate_history": True},
        }
    )
    solution = solver.solve(graph, report=0, logs=0)
    assert set(solution) == {10, 30, 80}
    assert solver.sdp_status == "optimal"
    assert math.isclose(solver.sdp_objective_raw, 2.25, abs_tol=1e-4)
    assert solver.sdp_upper_bound is None
    assert solver.candidate_history is not None
    assert len(solver.candidate_history) == 8
    assert solver.cut_value == graph.cut_value(solution)
    assert 0.0 <= solver.cut_value <= 2.0


def test_refiner_control_D_E_F_and_gauge():
    graph = WeightedGraph(
        [(10, 30, 1), (10, 80, 4), (30, 80, 3)],
        nodes=[10, 30, 80],
    )
    refiner = ExactBlockRefiner(refiner_config([[10, 30]]))
    input_solution = {10: 1, 30: 1, 80: 1}
    output = refiner.refine(graph, input_solution)
    assert input_solution == {10: 1, 30: 1, 80: 1}
    assert output == {10: -1, 30: -1, 80: 1}
    assert graph.cut_value(output) == 7.0
    assert refiner.last_refinement["gain"] == 7.0
    assert refiner.last_refinement["blocks_accepted"] == 1

    no_internal = WeightedGraph([(10, 30, 3)], nodes=[10, 30])
    refiner = ExactBlockRefiner(refiner_config([[10]]))
    output = refiner.refine(no_internal, {10: 1, 30: 1})
    assert output[10] == -1
    assert no_internal.cut_value(output) == 3.0

    # F: two independent old-environment gains must not be added at once.
    graph_f = WeightedGraph(
        [(10, 30, 3), (10, 80, -2), (30, 80, -2)],
        nodes=[10, 30, 80],
    )
    refiner = ExactBlockRefiner(refiner_config([[10], [30]], max_passes=2))
    output = refiner.refine(graph_f, {10: 1, 30: 1, 80: 1})
    assert output == {10: -1, 30: 1, 80: 1}
    assert graph_f.cut_value(output) == 1.0


def test_refiner_state_limit_stops_before_block_and_preserves_input():
    graph = WeightedGraph(
        [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1)],
        nodes=list(range(5)),
    )
    refiner = ExactBlockRefiner(
        config=refiner_config([[0, 1, 2], [3, 4]], max_passes=1)
    )
    original = {node: 1 for node in graph.nodes}
    output = refiner.refine(
        graph, original, context={"refinement_states_total": 4}
    )
    assert output == original
    record = refiner.last_refinement
    assert record["states_evaluated"] == 0
    assert record["blocks_started"] == 0
    assert record["status"] == "stopped"
    assert record["stop_reason"] == "refinement_states_total"

    refiner = ExactBlockRefiner(
        config=refiner_config([[0, 1, 2], [3, 4]], max_passes=1)
    )
    refiner.refine(graph, original, context={"refinement_states_total": 10})
    record = refiner.last_refinement
    assert record["states_evaluated"] == 8
    assert record["blocks_started"] == 1
    assert record["status"] == "stopped"
    assert record["stop_reason"] == "refinement_states_total"


def test_refiner_pair_selector_and_ranked_priority():
    graph = WeightedGraph(
        [(10, 30, 5), (30, 80, 3), (10, 80, 1), (80, 101, 2)],
        nodes=[10, 30, 80, 101],
    )
    refiner = ExactBlockRefiner(
        {
            "type": "exact_block",
            "parameters": {"max_size": 3, "batch_size": 4},
            "selector": {
                "type": "pair_greedy_abs_weight",
                "parameters": {"max_seed_pairs": 2},
            },
        }
    )
    output = refiner.refine(graph, {node: 1 for node in graph.nodes})
    assert set(output) == set(graph.nodes)
    assert refiner.last_refinement["blocks_started"] >= 1
    assert refiner.last_refinement["selector"] == "pair_greedy_abs_weight"


def test_refined_solver_exact_and_qaoa_base():
    graph = WeightedGraph(
        [(10, 30, 1), (10, 80, 4), (30, 80, 3)],
        nodes=[10, 30, 80],
    )
    base = {
        "type": "exact_cpu",
        "parameters": {"max_vertices": 8},
    }
    config = {
        "type": "refined",
        "seed": 3,
        "num_treads": 1,
        "base_solver": base,
        "refiner": refiner_config([[10, 30]]),
        "limits": {"refinement_states_total": 16},
        "collect": {"base_run": True, "refinement_history": True},
    }
    solver = RefinedSolver(config)
    solution = solver.solve(graph, report=0, logs=0)
    assert set(solution) == set(graph.nodes)
    assert solver.cut_value >= solver.base_cut_value
    assert solver.gain >= 0
    assert solver.base_run["algorithm_type"] == "exact_cpu"
    assert solver.refinement_run is not None
    assert solver.solution_energy == -solver.cut_value

    qaoa_base = {
        "type": "qaoa_cpu",
        "parameters": {"layers": 1, "max_vertices": 8, "dtype": "complex128"},
        "objective": {"type": "mean_cut", "parameters": {}},
        "estimation": {"mode": "exact"},
        "optimizer": {
            "type": "adam",
            "parameters": {"max_steps": 1},
            "gradient": {"type": "central_difference", "parameters": {"step": 0.01}},
        },
        "initialization": {"type": "fixed_angles", "parameters": {"values": [0.7, -0.3]}},
        "readout": {"mode": "most_probable"},
    }
    solver = RefinedSolver({**config, "base_solver": qaoa_base})
    solution = solver.solve(graph, report=0, logs=0)
    assert set(solution) == set(graph.nodes)
    assert solver.cut_value >= solver.base_cut_value
    assert solver.base_run["algorithm_type"] == "qaoa_cpu"


def test_unknown_classical_fields_are_rejected():
    graph = WeightedGraph([(0, 1, 1)], nodes=[0, 1])
    with pytest.raises(ValueError):
        GoemansWilliamsonCPU(
            {"type": "gw_cpu", "parameters": {"algorithm": "bad"}}
        ).solve(graph, report=0, logs=0)
    with pytest.raises(ValueError):
        SimulatedAnnealingCPU(
            {"type": "simulated_annealing_cpu", "parameters": {"shots": 10}}
        ).solve(graph, report=0, logs=0)


def test_classical_metrics_and_plots_do_not_run_solver():
    graph = WeightedGraph([(10, 30, 1), (10, 80, 4), (30, 80, 3)], nodes=[10, 30, 80])
    metrics = tools.analyze_solution(graph, {10: 1, 30: 1, 80: -1}, reference_cut=7)
    assert metrics["cut_value"] == 7.0
    assert metrics["solution_energy"] == -7.0
    assert metrics["separated_edge_count"] == 2
    assert metrics["optimality_gap"] == 0.0
    assert metrics["ratio"] == 1.0

    gw = GoemansWilliamsonCPU(
        {"type": "gw_cpu", "seed": 1, "parameters": {"num_hyperplanes": 4}, "collect": {"candidate_history": True}}
    )
    gw.solve(graph, report=0, logs=0)
    assert tools.plot_gw_convergence(gw, {"style": {"figsize": [5, 4]}}) is not None

    sa = SimulatedAnnealingCPU(
        sa_config(
            parameters={
                "num_reads": 2,
                "num_sweeps": 8,
                "beta_schedule_type": "geometric",
                "beta_range": [0.1, 4.0],
            }
        )
    )
    sa.solve(graph, report=0, logs=0)
    assert tools.plot_sa_reads(sa, {"style": {"figsize": [5, 4]}}) is not None
