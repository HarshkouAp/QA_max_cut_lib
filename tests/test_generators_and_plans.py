"""Tests for generators, structured weights, typed plans, and analysis."""

import json
import math

import numpy as np
import pytest

from QA_max_cut_lib import WeightedGraph
from QA_max_cut_lib.generators.benchmarks import (
    bench297_ids,
    materialize_bench16,
    materialize_bench297,
    resolve_bench297,
)
from QA_max_cut_lib.generators.graphs import GraphFactory
from QA_max_cut_lib.generators.plans import (
    SubproblemPlan,
    build_connected_bfs_partition,
    build_edge_coverage_scan,
    build_edge_neighborhood_cover,
    build_overlap_edge_partition,
    build_random_partition_plan,
    validate_plan,
)
from QA_max_cut_lib.generators.structured import build_hebbian_patterns, build_wishart_from_factor
from QA_max_cut_lib.tools.partition_analysis import analyze_plan, hop_metrics


def test_recipe_path_fill_and_grid():
    path = {
        "schema_version": 2,
        "type": "connected_path_fill_v1",
        "seed": 18,
        "parameters": {"n_vertices": 6, "nodes": [-7, 10, 30, 80, 101, 205], "n_edges": 9},
        "weights": {"type": "signed_discrete", "seed": 118, "parameters": {"values": [-5, -3, -1, 1, 3, 5]}},
    }
    graph = GraphFactory.from_recipe(path)
    assert len(graph) == 6
    assert len(graph.edges) == 9
    assert graph.snapshot()["recipe"]["resolved"]["topology_seed"] == 18
    grid = GraphFactory.from_recipe(
        {"schema_version": 2, "type": "periodic_grid_v1", "parameters": {"n_vertices": 9, "dimensions": [3, 3], "periodic": [True, True]}}
    )
    assert len(grid.edges) == 18
    line = GraphFactory.from_recipe(
        {"schema_version": 2, "type": "periodic_grid_v1", "parameters": {"n_vertices": 3, "dimensions": [1, 3], "periodic": [True, True]}}
    )
    assert len(line.edges) == 3


def test_benchmark_decoding():
    assert resolve_bench297(17)["topology_seed"] == 50042
    assert resolve_bench297(17)["weight_seed"] == 50345
    assert resolve_bench297(296)["topology_seed"] == 980042
    assert resolve_bench297(296)["weight_seed"] == 980345
    assert len(bench297_ids()) == 297
    with pytest.raises(ValueError):
        resolve_bench297(297)
    assert len(materialize_bench16(13)) == 120
    assert len(materialize_bench297(17).edges) > 0


def test_wishart_integer_witness():
    graph, record = build_wishart_from_factor(
        np.array([[1.0], [1.0], [2.0]]), [1, 1, -1], labels=[80, -7, 10]
    )
    assert graph.weight(80, -7) == 1.0
    assert graph.weight(80, 10) == 2.0
    assert graph.weight(-7, 10) == 2.0
    assert record["candidate_cut"] == 4.0
    assert record["theoretical_optimum_cut"] == 4.0
    assert record["stored_graph_optimum_verified"] is False


def test_hebbian_counterexample():
    patterns = [[1, 1, 1]] * 5 + [[1, 1, -1]]
    graph, record = build_hebbian_patterns(patterns)
    assert [weight for _, _, weight in graph.edges] == [-5.0, -1.0, -1.0]
    assert record["coefficients"][-1] == 6.0
    assert record["candidates"][5]["candidate_cut"] == -2.0
    assert record["candidates"][0]["candidate_cut"] == 0.0
    assert all(item["optimality_status"] == "not_certified" for item in record["candidates"])


def test_edge_partition_is_exclusive_and_full():
    graph = WeightedGraph(
        [(-7, 10, 2), (10, 30, -3), (-7, 30, 5)],
        nodes=[-7, 10, 30, 80],
    )
    plan = build_overlap_edge_partition(graph, max_size=3, seed=0)
    assert plan.plan_kind == "edge_partition"
    assert plan.completion_status == "complete"
    assert plan.coverage["edges"] == 3
    assert plan.coverage["vertices"] == 4
    assert validate_plan(graph, plan)
    restored = SubproblemPlan.from_dict(plan.to_dict())
    assert restored.plan_kind == plan.plan_kind
    assert [part.edges for part in restored.parts] == [part.edges for part in plan.parts]


def test_vertex_partition_and_coverages():
    graph = WeightedGraph(
        [(-7, 10, 2), (10, 30, -3), (-7, 30, 5)],
        nodes=[-7, 10, 30, 80],
    )
    plan = build_random_partition_plan(graph, max_size=2, seed=1)
    assert plan.plan_kind == "vertex_partition"
    analysis = analyze_plan(graph, plan)
    assert analysis["vertex_coverage"] == 1.0
    assert 0.0 < analysis["edge_coverage"] < 1.0
    cover = build_edge_neighborhood_cover(graph, (-7, 10), max_size=3)
    assert cover.plan_kind == "induced_cover"
    assert all(set((-7, 10)) <= set(part.nodes) for part in cover.parts)


def test_scanner_cap_and_hops():
    graph = WeightedGraph([(0, 1, 1), (1, 2, 1)], nodes=[0, 1, 2])
    assert hop_metrics(graph)["reachable_mean_hop_distance"] == pytest.approx(4 / 3)
    assert hop_metrics(graph)["global_efficiency"] == pytest.approx(5 / 6)
    plan = build_edge_coverage_scan(
        graph,
        max_tasks=2,
        target_edge_coverage=1.0,
        local_builder={"max_size": 3, "goal": "cover_neighbors", "growth_metric": "count"},
    )
    assert plan.plan_kind == "edge_partition"
    assert validate_plan(graph, plan)


def test_edge_subset_adapter_rejects_induced_mode():
    from QA_max_cut_lib.generators.plans import SubproblemSpec
    from QA_max_cut_lib.solvers.local_problem import (
        LocalProblemCapabilityError,
        local_graph_for_spec,
    )

    graph = WeightedGraph([(0, 1, 2), (1, 2, -3)], nodes=[0, 1, 2])
    spec = SubproblemSpec(
        task_id="x",
        nodes=(0, 1, 2),
        edges=((0, 1, 2.0),),
        edge_domain="explicit",
    )
    local = local_graph_for_spec(graph, spec)
    assert local.edges == ((0, 1, 2.0),)
    from QA_max_cut_lib.solvers.local_problem import validate_capability

    with pytest.raises(LocalProblemCapabilityError):
        validate_capability(
            SubproblemPlan("edge_partition", graph.snapshot()["fingerprint"], {}, (spec,)),
            "induced_no_fields",
        )
