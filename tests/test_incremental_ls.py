"""Tests for incremental LS-RQAOA builders, selectors, and one real run."""

from itertools import combinations
import json
import math

import numpy as np
import pytest

from QA_max_cut_lib import WeightedGraph
from QA_max_cut_lib.core.reduction import GraphReducer
from QA_max_cut_lib.generators.graphs import GraphFactory
from QA_max_cut_lib.generators.incremental import (
    DensestEdgeCountBuilder,
    survivor_neighbor_cover,
)
from QA_max_cut_lib.solvers.incremental_ls import IncrementalLSRQAOACPU
from QA_max_cut_lib.solvers.exact_cpu import ExactCPU


def _selector_config(rule, q=1, pairs="edges"):
    config = json.loads(open("configs/solvers/incremental_ls_keep.json").read())
    config["output"] = {"report": 0}
    config["pairs"] = pairs
    config["selection"] = {"type": rule, "weight_power": q}
    return config


def _selection_fixture():
    graph = WeightedGraph(
        [(10, 30, 1), (10, 80, 3), (30, 80, -5), (80, 101, 2)],
        nodes=[10, 30, 80, 101],
    )
    prepared = {
        (10, 30): {"physical_correlation": -0.9, "observed_epoch": 0, "task_ids": ("a",)},
        (10, 80): {"physical_correlation": 0.6, "observed_epoch": 0, "task_ids": ("a",)},
        (30, 80): {"physical_correlation": 0.3, "observed_epoch": 0, "task_ids": ("a",)},
        (80, 101): {"physical_correlation": -0.8, "observed_epoch": 0, "task_ids": ("a",)},
    }
    state = GraphReducer().start(graph)
    return graph, state, prepared


def test_four_selection_rules():
    graph, state, prepared = _selection_fixture()
    expected = {
        "max_abs_correlation": ((10, 30), -1),
        "max_abs_weighted_correlation_q1": ((10, 80), 1),
        "max_abs_weighted_correlation_q2": ((30, 80), 1),
        "min_signed_edge_product": ((80, 101), -1),
    }
    for name, (pair, sign) in expected.items():
        rule = "max_abs_correlation" if name == "max_abs_correlation" else (
            "min_signed_edge_product" if name == "min_signed_edge_product" else "max_abs_weighted_correlation"
        )
        q = 2 if name.endswith("q2") else 1
        solver = IncrementalLSRQAOACPU(_selector_config(rule, q))
        solver._validate(solver._config, graph)
        selected = solver._select(state, prepared, solver._config, 0)
        assert selected.pair == pair
        assert selected.sign == sign
        assert selected.score >= 0 or rule == "min_signed_edge_product"


def test_score_uses_current_weight_and_zero_sign():
    graph, state, prepared = _selection_fixture()
    prepared[(10, 30)]["physical_correlation"] = 0.0
    solver = IncrementalLSRQAOACPU(_selector_config("min_signed_edge_product"))
    solver._validate(solver._config, graph)
    selected = solver._select(state, prepared, solver._config, 2)
    assert selected.pair == (80, 101)
    assert selected.correlation == -0.8
    assert selected.weight == 2.0
    assert selected.age == 2


def test_star_cover_and_full_graph_shortcut():
    graph = WeightedGraph(
        [(80, -7, 1), (80, 10, 1), (80, 30, 1), (80, 101, 1), (80, 205, 1)],
        nodes=[80, -7, 10, 30, 101, 205],
    )
    tasks = survivor_neighbor_cover(graph, 80, max_size=3, growth_metric="count", version=2)
    assert len(tasks) == 3
    assert all(80 in task["nodes"] for task in tasks)
    covered = {node for task in tasks for node in task["nodes"] if node != 80}
    assert covered == {-7, 10, 30, 101, 205}
    full = survivor_neighbor_cover(graph, 80, max_size=6, full_graph_when_fits=True)
    assert len(full) == 1
    assert set(full[0]["nodes"]) == set(graph.nodes)


def test_densest_objective_ignores_weights():
    graph = WeightedGraph(
        [(0, 1, 2), (1, 2, -3), (0, 2, 5), (2, 3, -1)],
        nodes=[0, 1, 2, 3],
    )
    negated = WeightedGraph(
        [(u, v, -w) for u, v, w in graph.edges], nodes=graph.nodes
    )
    builder = DensestEdgeCountBuilder(
        {"parameters": {"max_size": 3, "n_restarts": 10, "local_iter": 20, "candidate_pool": 8}},
        seed=11,
    )
    first = builder.build(graph)[0]
    second = builder.build(negated)[0]
    assert first["internal_edge_count"] == 3
    assert set(first["nodes"]) == {0, 1, 2}
    assert set(second["nodes"]) == {0, 1, 2}


def test_incremental_real_run_and_full_cut():
    labels = [-7, 10, 30, 80, 101, 205]
    pairs = list(combinations(labels, 2))
    graph = WeightedGraph(
        [(u, v, (2 * i + 1) / 8) for i, (u, v) in enumerate(pairs)],
        nodes=labels,
    )
    config = json.loads(open("configs/solvers/incremental_ls_keep.json").read())
    config["output"] = {"report": 0}
    solver = IncrementalLSRQAOACPU(config)
    solution = solver.solve(graph, num_treads=1)
    assert set(solution) == set(labels)
    assert set(solution.values()) == {-1, 1}
    assert solver.result["status"] == "completed"
    assert solver.result["reductions_completed"] == 2
    assert solver.result["resources"]["shots_used"] <= 224
    assert solver.cut_value == graph.cut_value(solution)
    assert solver.result["final_bank"] is not None
    assert solver.result["resources"]["training_shots"] > 0
    exact = ExactCPU({"parameters": {"max_vertices": 6}})
    exact.solve(graph, report=0)
    assert solver.cut_value <= exact.cut_value + 1e-12
