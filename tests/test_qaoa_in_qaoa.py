"""Tests for the hierarchical QAOA-in-QAOA coordinator."""

from itertools import product
import math

import numpy as np
import pytest

from QA_max_cut_lib import GraphFactory, WeightedGraph, ExactCPU, tools
from QA_max_cut_lib.core.reduction import GraphReducer
from QA_max_cut_lib.solvers.hierarchy_components import project_blocks
from QA_max_cut_lib.solvers.qaoa_in_qaoa_cpu import QAOAInQAOACPU
from QA_max_cut_lib.solvers.qaoa_in_qaoa_gpu import QAOAInQAOAGPU


def edge_graph(n=8):
    edges = [
        (0, 1, 1),
        (1, 2, -2),
        (2, 3, 2),
        (3, 4, 1),
        (4, 5, -1),
        (5, 6, 2),
        (6, 7, 1),
        (0, 7, -2),
        (0, 3, 1),
        (1, 4, 2),
        (2, 5, -1),
        (3, 6, 1),
        (4, 7, -2),
    ]
    return WeightedGraph(edges, nodes=list(range(n)))


def control_graph():
    return WeightedGraph(
        [
            (10, 30, 2),
            (80, 101, 3),
            (10, 80, 5),
            (10, 101, 1),
            (30, 80, 4),
            (30, 101, -2),
        ],
        nodes=[10, 30, 80, 101],
    )


def local_solver(max_steps=1, readout="most_probable", shots=None):
    config = {
        "type": "qaoa_cpu",
        "parameters": {"layers": 1, "max_vertices": 3, "dtype": "complex128"},
        "objective": {"type": "mean_cut", "parameters": {}},
        "estimation": {"mode": "exact"},
        "optimizer": {
            "type": "adam",
            "parameters": {"max_steps": max_steps},
            "gradient": {"type": "central_difference", "parameters": {"step": 0.01}},
        },
        "initialization": {"type": "fixed_angles", "parameters": {"values": [0.7, -0.3]}},
        "readout": {"mode": readout},
    }
    if shots is not None:
        config["readout"]["shots"] = shots
    return config


def config(**overrides):
    base = {
        "type": "qaoa_in_qaoa_cpu",
        "seed": 5,
        "num_treads": 1,
        "parameters": {"finish_at": 2, "max_levels": None},
        "partitioner": {
            "type": "greedy_abs_weight_partition",
            "parameters": {
                "max_size": 2,
                "rounds": 1,
                "full_graph_when_fits": False,
            },
        },
        "local_solver": local_solver(),
        "execution": {"backend": "process", "start_method": "spawn", "native_threads": 1},
        "budget_policy": {"type": "level_round_robin"},
        "finisher": {"type": "exact_cpu", "parameters": {"max_vertices": 4}},
        "limits": {"shots_total": 0},
        "collect": {
            "hierarchy_history": True,
            "block_patterns": True,
            "parameter_trajectory": True,
        },
    }
    base.update(overrides)
    return base


def test_projection_control_weights_offset_and_energy():
    graph = control_graph()
    blocks = [(10, 30), (80, 101)]
    patterns = {
        10: {10: 1, 30: -1},
        80: {80: 1, 101: -1},
    }
    meta, delta_b, records = project_blocks(graph, blocks, patterns)
    assert meta.nodes == (10, 80)
    assert meta.edges == ((10, 80, -2.0),)
    assert delta_b == 10.0
    assert records[0]["anchor"] == 10
    assert records[1]["normalized_pattern"] == {80: 1, 101: -1}
    for tau_10, tau_80 in product((-1, 1), repeat=2):
        solution = {
            10: patterns[10][10] * tau_10,
            30: patterns[10][30] * tau_10,
            80: patterns[80][80] * tau_80,
            101: patterns[80][101] * tau_80,
        }
        assert graph.cut_value(solution) == delta_b + meta.cut_value(
            {10: tau_10, 80: tau_80}
        )


def test_projection_matches_core_constraints():
    graph = control_graph()
    blocks = [(10, 30), (80, 101)]
    patterns = {
        10: {10: 1, 30: -1},
        80: {80: 1, 101: -1},
    }
    meta, delta_b, _ = project_blocks(graph, blocks, patterns)
    state = GraphReducer().start(graph)
    for block in blocks:
        anchor = block[0]
        for node in block[1:]:
            GraphReducer().reduce(state, node, anchor, patterns[anchor][node])
    assert state.current_nodes == meta.nodes
    assert state.current_graph().edges == meta.edges
    assert math.isclose(state.cut_offset, delta_b, abs_tol=1e-12)
    assert math.isclose(
        state.ising.constant,
        -delta_b - 0.5 * sum(weight for _, _, weight in meta.edges),
        abs_tol=1e-12,
    )
    assert math.isclose(state.ising.constant, -9.0, abs_tol=1e-12)


def test_internal_gauge_and_constant_ising():
    graph = control_graph()
    blocks = [(10, 30), (80, 101)]
    for sign in (-1, 1):
        # Raw patterns r and -r normalize to the same canonical pattern.
        raw = {
            10: {10: sign, 30: -sign},
            80: {80: sign, 101: -sign},
        }
        normalized = {
            anchor: {node: value // sign for node, value in values.items()}
            for anchor, values in raw.items()
        }
        meta, delta_b, _ = project_blocks(graph, blocks, normalized)
        assert delta_b == 10.0
        assert meta.edges == ((10, 80, -2.0),)


def test_hierarchy_multiple_levels_and_complete_restore():
    solver = QAOAInQAOACPU(config())
    solution = solver.solve(edge_graph(), report=0, logs=0)
    assert set(solution) == set(range(8))
    assert all(spin in (-1, 1) for spin in solution.values())
    assert solver.levels_completed >= 2
    assert solver.primitive_reductions_completed > 0
    assert solver.local_solver_calls >= 2
    assert solver.solution_source == "exact_finisher"
    assert math.isclose(solver.cut_value, edge_graph().cut_value(solution))
    assert solver.hierarchy_tree["root_level"] == solver.levels_completed - 1
    assert all("id" in root for root in solver.hierarchy_tree["roots"])


def test_parallel_and_sequential_have_same_plan_and_result():
    sequential = QAOAInQAOACPU(config(num_treads=1))
    parallel = QAOAInQAOACPU(config(num_treads=2))
    graph = edge_graph()
    first = sequential.solve(graph, report=0, logs=0)
    second = parallel.solve(graph, report=0, logs=0)
    assert first == second
    assert sequential.cut_value == parallel.cut_value
    assert sequential.resources["num_treads_requested"] == 1
    assert parallel.resources["num_treads_requested"] == 2
    assert parallel.resources["num_treads_applied"] == 2
    assert len(sequential.hierarchy_history) == sequential.levels_completed
    assert len(parallel.hierarchy_history) == parallel.levels_completed
    assert [
        [(block["anchor"], block["nodes"]) for block in level["blocks"]]
        for level in sequential.hierarchy_history
    ] == [
        [(block["anchor"], block["nodes"]) for block in level["blocks"]]
        for level in parallel.hierarchy_history
    ]
    parallel_pids = {
        pid
        for level in parallel.hierarchy_history
        for pid in level.get("task_pids", [])
        if pid is not None
    }
    assert len(parallel_pids) > 1


def test_best_of_shots_uses_real_readout_and_respects_budget():
    local = local_solver(max_steps=0, readout="best_of_shots", shots=8)
    graph = edge_graph(8)
    solver = QAOAInQAOACPU(config(local_solver=local, limits={"shots_total": 32}))
    solution = solver.solve(graph, report=0, logs=0)
    assert set(solution) == set(range(8))
    assert solver.resources["shots_used"] <= 32
    assert solver.resources["readout_shots"] > 0
    assert solver.local_solver_calls >= 1


def test_insufficient_readout_budget_does_not_start_level():
    local = local_solver(max_steps=0, readout="best_of_shots", shots=32)
    solver = QAOAInQAOACPU(
        config(local_solver=local, limits={"shots_total": 6}, parameters={"finish_at": 2})
    )
    solution = solver.solve(edge_graph(8), report=0, logs=0)
    assert set(solution) == set(range(8))
    assert solver.status == "stopped"
    assert solver.termination_reason == "shot_budget"
    assert solver.solution_source == "hierarchy_all_plus_fallback"
    assert solver.resources["shots_used"] == 0
    assert solver.local_solver_calls == 0


def test_max_levels_and_task_limits_produce_fallback():
    limited = QAOAInQAOACPU(config(parameters={"finish_at": 2, "max_levels": 1}))
    solution = limited.solve(edge_graph(), report=0, logs=0)
    assert set(solution) == set(range(8))
    assert limited.levels_completed == 1
    assert limited.status == "stopped"
    assert limited.termination_reason == "max_levels"

    task_limited = QAOAInQAOACPU(
        config(limits={"shots_total": 0, "max_local_tasks_total": 1})
    )
    solution = task_limited.solve(edge_graph(), report=0, logs=0)
    assert set(solution) == set(range(8))
    assert task_limited.termination_reason == "max_local_tasks_total"
    assert task_limited.local_solver_calls == 0


def test_max_levels_zero_and_optional_history_collection():
    zero = QAOAInQAOACPU(config(parameters={"finish_at": 2, "max_levels": 0}))
    solution = zero.solve(edge_graph(), report=0, logs=0)
    assert set(solution) == set(range(8))
    assert zero.levels_completed == 0
    assert zero.termination_reason == "max_levels"

    collected = QAOAInQAOACPU(
        config(
            collect={
                "hierarchy_history": True,
                "block_patterns": True,
                "parameter_trajectory": True,
                "coarse_graphs": True,
                "final_reduction_state": True,
            }
        )
    )
    collected.solve(edge_graph(8), report=0, logs=0)
    assert collected.coarse_graphs
    assert collected.final_reduction_state["schema_version"] == 1
    assert collected.hierarchy_history[0]["coarse_graph_before"] is not None


def test_local_solver_rejects_root_owned_fields():
    with pytest.raises(ValueError):
        QAOAInQAOACPU(
            config(local_solver={**local_solver(), "limits": {"shots_total": 10}})
        ).solve(edge_graph(8), report=0, logs=0)


def test_finish_at_and_empty_or_zero_interaction_cases():
    small = WeightedGraph([(10, 20, 2), (20, 30, -1)], nodes=[10, 20, 30])
    solver = QAOAInQAOACPU(config(parameters={"finish_at": 3}))
    solution = solver.solve(small, report=0, logs=0)
    assert set(solution) == {10, 20, 30}
    assert solver.levels_completed == 0
    assert solver.solution_source == "exact_finisher"

    empty = WeightedGraph([], nodes=[])
    empty_solver = QAOAInQAOACPU(config())
    assert empty_solver.solve(empty, report=0, logs=0) == {}
    assert empty_solver.solution_source == "empty_graph"

    no_edges = WeightedGraph([], nodes=[1, 2, 3])
    no_edge_solver = QAOAInQAOACPU(config())
    solution = no_edge_solver.solve(no_edges, report=0, logs=0)
    assert set(solution) == {1, 2, 3}
    assert no_edge_solver.solution_source == "zero_interaction_initial"


def test_singleton_and_zero_interaction_nontrivial_block_are_real_readout():
    # Each block is a singleton, so no QAOA call is made.  A singleton is
    # not a partition failure, but a hierarchy that cannot reduce is stopped.
    class SingletonPartitioner:
        def build(self, graph, *, version=0):
            return [{"nodes": (int(node),)} for node in graph.nodes]

    solver = QAOAInQAOACPU(
        config(),
        partitioner=SingletonPartitioner(),
        local_solver=lambda graph, payload: {node: 1 for node in graph.nodes},
    )
    solution = solver.solve(edge_graph(8), report=0, logs=0)
    assert set(solution) == set(range(8))
    assert solver.termination_reason == "no_partition_progress"
    assert solver.levels_completed == 0


def test_invalid_partition_and_backend_are_rejected_before_projection():
    invalid = QAOAInQAOACPU(config(partitioner={
        "type": "greedy_abs_weight_partition",
        "parameters": {"max_size": 2, "rounds": 2},
    }))
    with pytest.raises(ValueError):
        invalid.solve(edge_graph(4), report=0, logs=0)
    with pytest.raises(ValueError):
        QAOAInQAOAGPU(config())


def test_old_result_is_independent_from_next_solve():
    solver = QAOAInQAOACPU(config())
    first = solver.solve(edge_graph(8), report=0, logs=0)
    first_result = solver.result
    second = solver.solve(WeightedGraph([(0, 1, 1)], nodes=[0, 1, 2]), report=0, logs=0)
    assert set(first) == set(range(8))
    assert set(second) == {0, 1, 2}
    assert first_result["graph"]["nodes"] != solver.result["graph"]["nodes"]
    assert first_result["solution"] is not solver.result["solution"]


def test_hierarchy_tools_read_result_without_new_computation():
    solver = QAOAInQAOACPU(config())
    solver.solve(edge_graph(8), report=0, logs=0)
    for name in (
        "plot_hierarchy_overview",
        "plot_hierarchy_parameters",
        "plot_hierarchy_parameters_3d",
        "plot_hierarchy_tree",
    ):
        figure = getattr(tools, name)(solver, {"style": {"figsize": [5, 4]}})
        assert figure.qa_record.algorithm_type.startswith("qaoa_in_qaoa_")
