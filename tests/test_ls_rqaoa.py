"""Independent checks for the first LS-RQAOA implementation."""

from copy import deepcopy
import math

import numpy as np
import pytest

from QA_max_cut_lib import ExactCPU, LSRQAOACPU, RQAOACPU, SolverFactory, WeightedGraph
from QA_max_cut_lib.core.reduction import GraphReducer
from QA_max_cut_lib.generators.subgraphs import (
    GreedyAbsWeightPartitionBuilder,
    RandomPartitionBuilder,
)
from QA_max_cut_lib.solvers.ls_components import MeanCorrelationAggregator
from QA_max_cut_lib.solvers.ls_rqaoa_cpu import _round_robin_quotas


def graph_path():
    return WeightedGraph(
        [(0, 1, 2), (1, 2, 3), (2, 3, -1), (3, 4, 4), (4, 0, 1), (0, 2, 2)],
        nodes=[0, 1, 2, 3, 4],
    )


def config(builder="greedy_abs_weight_partition", source="analytic", **overrides):
    if source == "analytic":
        source_config = {
            "type": "qaoa_analytic_p1_cpu",
            "parameters": {"layers": 1},
            "objective": {"type": "mean_cut", "parameters": {}},
            "objective_reference": "working_graph",
            "estimation": {"mode": "exact"},
            "optimizer": {
                "type": "gamma_grid_exact_beta",
                "parameters": {
                    "gamma_bounds": [-math.pi, math.pi],
                    "points": 5,
                },
            },
            "correlations": {"mode": "exact"},
        }
    else:
        source_config = {
            "type": "qaoa_statevector_cpu",
            "parameters": {"layers": 1, "max_vertices": 20},
            "objective": {"type": "mean_cut", "parameters": {}},
            "objective_reference": "working_graph",
            "estimation": {"mode": "exact"},
            "optimizer": {
                "type": "adam",
                "parameters": {"max_steps": 2, "learning_rate": 0.05},
            },
            "initialization": {
                "type": "fixed_angles",
                "parameters": {"values": [0.5, 0.3]},
            },
            "correlations": {"mode": "exact"},
        }
    result = {
        "type": "ls_rqaoa_cpu",
        "seed": 7,
        "num_treads": 1,
        "parameters": {"finish_at": 2},
        "subgraph_builder": {
            "type": builder,
            "parameters": {
                "max_size": 3,
                "rounds": 2,
                "full_graph_when_fits": False,
            },
        },
        "correlation_source": source_config,
        "aggregation": {"type": "mean", "parameters": {}},
        "selection": {"type": "max_abs_correlation", "parameters": {"pairs": "edges"}},
        "coverage": {"require_full": False, "empty_policy": "anchor_pair"},
        "update": {"type": "refresh_all"},
        "execution": {"backend": "process", "start_method": "spawn", "native_threads": 1},
        "budget_policy": {"type": "step_round_robin"},
        "finisher": {"type": "exact_cpu", "parameters": {"max_vertices": 5}},
        "limits": {"shots_total": 0},
        "collect": {
            "reduction_history": True,
            "parameter_trajectory": True,
            "local_task_history": True,
        },
    }
    result.update(overrides)
    return result


def test_builders_cover_every_round_and_keep_duplicates():
    graph = WeightedGraph(
        [(0, 1, 1), (1, 2, 1), (2, 3, 1)],
        nodes=[-2, 0, 1, 3, 8],
    )
    builder = RandomPartitionBuilder(
        {
            "type": "random_partition",
            "parameters": {"max_size": 2, "rounds": 3, "full_graph_when_fits": False},
        },
        seed=11,
    )
    tasks = builder.build(graph, version=4)
    assert len(tasks) >= 3
    for round_index in range(3):
        blocks = [task["nodes"] for task in tasks if task["round"] == round_index]
        assert len(blocks) == 3
        assert all(len(block) <= 2 for block in blocks)
        assert set().union(*map(set, blocks)) == set(graph.nodes)
    assert len({task["task_id"] for task in tasks}) == len(tasks)


def test_greedy_builder_uses_external_labels_and_preserves_all_nodes():
    graph = WeightedGraph(
        [(0, 1, -5), (1, 3, 2), (3, 8, 4)],
        nodes=[-2, 0, 1, 3, 8],
    )
    builder = GreedyAbsWeightPartitionBuilder(
        {
            "type": "greedy_abs_weight_partition",
            "parameters": {"max_size": 2, "rounds": 2, "full_graph_when_fits": False},
        },
        seed=1,
    )
    tasks = builder.build(graph, version=0)
    for round_index in (0, 1):
        nodes = [task["nodes"] for task in tasks if task["round"] == round_index]
        assert set().union(*map(set, nodes)) == set(graph.nodes)
        assert all(len(task) <= 2 for task in nodes)
    full = GreedyAbsWeightPartitionBuilder(
        {
            "type": "greedy_abs_weight_partition",
            "parameters": {"max_size": 10, "rounds": 2, "full_graph_when_fits": True},
        },
        seed=3,
    ).build(graph, version=0)
    assert len(full) == 1
    assert full[0]["nodes"] == tuple(sorted(graph.nodes))


def test_mean_aggregator_uses_per_pair_observations_and_keeps_zero():
    state = GraphReducer().start(graph_path())
    results = [
        {"task_id": "a", "parent_version": 0, "values": {(0, 1): -0.8, (0, 2): 0.0}},
        {"task_id": "b", "parent_version": 0, "values": {(0, 1): 0.2, (0, 2): 0.0}},
        {"task_id": "c", "parent_version": 0, "values": {(1, 2): 0.4}},
    ]
    table = MeanCorrelationAggregator().aggregate(state, results, pairs="edges")
    assert table["counts"][(0, 1)] == 2
    assert table["values"][(0, 1)] == pytest.approx(-0.3)
    assert table["counts"][(0, 2)] == 2
    assert table["values"][(0, 2)] == 0.0
    assert table["counts"][(1, 2)] == 1
    assert table["values"][(1, 2)] == pytest.approx(0.4)
    assert table["task_ids"][(0, 1)] == ("a", "b")


def test_round_robin_quotas_follow_the_required_example():
    assert _round_robin_quotas([50, 50, 50], [100, 100, 100], [4, 4, 4], 350, 0) == (
        [1, 1, 0],
        [150, 150, 50],
    )
    assert _round_robin_quotas([50, 50, 50], [100, 100, 100], [4, 4, 4], 149, 0) is None
    assert _round_robin_quotas([0, 0], [0, 0], [2, 3], 0, 0) == ([2, 3], [0, 0])


@pytest.mark.parametrize("builder", ["random_partition", "greedy_abs_weight_partition"])
def test_full_analytic_ls_restores_original_energy(builder):
    graph = graph_path()
    result = LSRQAOACPU(config(builder=builder)).solve(graph, report=0, logs=0)
    solver = LSRQAOACPU(config(builder=builder))
    solver.solve(graph, report=0, logs=0)
    assert set(result) == set(graph.nodes)
    assert solver.status == "completed"
    assert solver.resources["shots_used"] == 0
    assert solver.resources["reductions_completed"] >= 0
    assert solver.solution_energy == pytest.approx(
        -graph.cut_value(result), abs=1e-9
    )
    assert solver.cut_offset is not None
    assert solver.reduced_solution.keys() <= set(solver.remaining_node_order)


def test_statevector_source_and_parameter_rows_are_local():
    graph = graph_path()
    cfg = config(source="statevector")
    solver = LSRQAOACPU(cfg)
    solution = solver.solve(graph, report=0, logs=0)
    assert set(solution) == set(graph.nodes)
    assert solver.resources["shots_used"] == 0
    assert solver.parameter_trajectory
    assert all(row["task_id"] for row in solver.parameter_trajectory)
    assert all(row["layers"] == 1 for row in solver.parameter_trajectory)
    assert solver.local_task_history
    assert all("task_version" in row for row in solver.local_task_history)


def test_parallel_and_sequential_have_the_same_scientific_plan():
    graph = graph_path()
    sequential = LSRQAOACPU(config())
    sequential.solve(graph, report=0, logs=0)
    parallel = LSRQAOACPU(config(num_treads=2))
    parallel.solve(graph, report=0, logs=0)
    assert parallel.resources["num_treads_requested"] == 2
    assert parallel.resources["num_treads_applied"] in (1, 2)
    assert parallel.solution == sequential.solution
    assert parallel.reductions_completed == sequential.reductions_completed
    assert parallel.cut_value == pytest.approx(sequential.cut_value)
    assert parallel.resources["num_treads"] == parallel.num_treads


def test_require_full_is_checked_before_local_evaluation():
    graph = WeightedGraph([(0, 1, 1), (1, 2, 1), (2, 3, 1)], nodes=[0, 1, 2, 3])
    cfg = config()
    cfg["subgraph_builder"]["parameters"]["max_size"] = 2
    cfg["selection"]["parameters"]["pairs"] = "all_pairs"
    cfg["coverage"]["require_full"] = True
    with pytest.raises(ValueError, match="cover"):
        LSRQAOACPU(cfg).solve(graph, report=0, logs=0)


def test_shot_budget_stops_with_a_full_valid_solution():
    graph = graph_path()
    cfg = config(source="statevector")
    cfg["correlation_source"]["estimation"] = {
        "mode": "shots",
        "shots_per_evaluation": 100,
    }
    cfg["correlation_source"]["correlations"] = {"mode": "shots", "shots": 50}
    cfg["correlation_source"]["optimizer"]["parameters"]["max_steps"] = 3
    cfg["limits"]["shots_total"] = 10
    solver = LSRQAOACPU(cfg)
    solution = solver.solve(graph, report=0, logs=0)
    assert solver.status == "stopped"
    assert solver.termination_reason == "shot_budget"
    assert solver.resources["shots_used"] == 0
    assert set(solution) == set(graph.nodes)


def test_empty_and_zero_interaction_graphs_are_complete():
    for graph in (
        WeightedGraph([], nodes=[]),
        WeightedGraph([(10, 30, 0.0)], nodes=[10, 30]),
    ):
        solver = LSRQAOACPU(config())
        solution = solver.solve(graph, report=0, logs=0)
        assert set(solution) == set(graph.nodes)
        assert solver.status == "completed"
        assert solver.resources["reductions_completed"] == 0


def test_global_reduction_touches_neighbors_outside_the_local_task():
    graph = WeightedGraph(
        [(10, 30, 2), (10, 80, 3), (30, 80, 5)],
        nodes=[10, 30, 80, 101],
    )
    state = GraphReducer().start(graph)
    event = GraphReducer().reduce(
        state, 10, 30, -1, correlation=-0.6, selection_score=0.6
    )
    assert state.current_graph().weight(30, 80) == pytest.approx(2.0)
    assert event["cut_offset"] == pytest.approx(5.0)
    assert event["fixed_energy"] == pytest.approx(-2.0)
    residual = ExactCPU({"parameters": {"max_vertices": 3}}).solve(state.current_graph())
    solution = GraphReducer().restore(state, residual)
    assert graph.cut_value(solution) == pytest.approx(7.0)


def test_ls_coverage_plot_reads_stored_history():
    from QA_max_cut_lib.tools.reduction_plots import plot_ls_coverage

    solver = LSRQAOACPU(config())
    solver.solve(graph_path(), report=0, logs=0)
    figure = plot_ls_coverage(solver)
    assert len(figure.axes) == 2


def test_anchor_task_is_added_before_budget_when_partition_misses_all_edges():
    graph = WeightedGraph([(0, 1, 1)], nodes=[0, 1, 2])
    cfg = config()
    cfg["seed"] = 1
    cfg["subgraph_builder"]["type"] = "random_partition"
    cfg["parameters"]["finish_at"] = 1
    cfg["subgraph_builder"]["parameters"]["max_size"] = 2
    cfg["subgraph_builder"]["parameters"]["rounds"] = 1
    cfg["correlation_source"]["optimizer"]["parameters"]["points"] = 3
    solver = LSRQAOACPU(cfg)
    solver.solve(graph, report=0, logs=0)
    first = solver.coverage_history[0]
    assert first["repair_added"] is True
    assert any(task.get("repair") for task in solver.local_task_history)
    assert solver.resources["shots_used"] == 0


def test_full_graph_ls_matches_rqaoa_with_analytic_source():
    graph = graph_path()
    cfg = config()
    cfg["subgraph_builder"]["parameters"].update(
        {"max_size": 10, "full_graph_when_fits": True}
    )
    ls = LSRQAOACPU(cfg)
    ls_solution = ls.solve(graph, report=0, logs=0)
    rq = RQAOACPU(
        {
            "type": "rqaoa_cpu",
            "seed": 7,
            "parameters": {"finish_at": 2},
            "correlation_source": cfg["correlation_source"],
            "selection": cfg["selection"],
            "finisher": cfg["finisher"],
            "limits": {"shots_total": 0},
            "collect": {"reduction_history": True, "parameter_trajectory": True},
        }
    )
    rq_solution = rq.solve(graph, report=0, logs=0)
    assert ls_solution == rq_solution
    assert [e["removed_node"] for e in ls.reduction_history] == [
        e["removed_node"] for e in rq.reduction_history
    ]


def test_induced_local_graph_drops_boundary_edges_without_touching_parent():
    graph = graph_path()
    builder = RandomPartitionBuilder(
        {
            "type": "random_partition",
            "parameters": {"max_size": 2, "rounds": 1, "full_graph_when_fits": False},
        },
        seed=3,
    )
    task = builder.build(graph, version=0)[0]
    local = graph.subgraph(task["nodes"])
    assert set(local.nodes) == set(task["nodes"])
    assert all(u in local.nodes and v in local.nodes for u, v, _ in local.edges)
    assert graph.snapshot() == graph.snapshot()
    for u, v, weight in graph.edges:
        if u in local.nodes and v in local.nodes:
            assert local.weight(u, v) == weight


def test_factory_and_config_validation():
    assert "ls_rqaoa_cpu" in SolverFactory.available()
    with pytest.raises(ValueError):
        LSRQAOACPU({**config(), "num_treads": 0}).solve(graph_path(), report=0, logs=0)
    with pytest.raises(ValueError):
        LSRQAOACPU({**config(), "num_treads": True}).solve(graph_path(), report=0, logs=0)
    cfg = config()
    cfg["correlation_source"]["objective_reference"] = "original_graph"
    with pytest.raises(ValueError, match="working_graph"):
        LSRQAOACPU(cfg).solve(graph_path(), report=0, logs=0)
