"""Tests for distributed LS-RQAOA composition."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from QA_max_cut_lib import WeightedGraph, restore
from QA_max_cut_lib.laboratory import DistributedLSRQAOACPU


def config():
    return json.loads(
        (Path(__file__).resolve().parents[1] / "configs/solvers/distributed_ls_jacobi_cpu.json").read_text()
    )


def fixture_c_graph():
    return WeightedGraph([(10, 30, 2)], nodes=[10, 30])


def fixture_c_config():
    value = config()
    value["parameters"] = {"finish_at": 1, "max_reductions": 1}
    value["partition_builder"] = {
        "type": "greedy_abs_weight_partition",
        "parameters": {"max_size": 3, "rounds": 1, "full_graph_when_fits": True},
    }
    value["messages"] = {
        "initial": {
            "type": "provided_means",
            "parameters": {
                "entries": [{"node": 10, "mean": 1.0}, {"node": 30, "mean": 1.0}]
            },
        },
        "episode_mean_aggregation": "uniform",
        "transport": {
            "type": "carry_survivor_means",
            "partition": "rebuild",
            "block_moments": "product_reset",
            "angles": "restart",
        },
    }
    source = value["correlation_source"]
    source["mean_field"]["max_sweeps"] = 0
    source["mean_field"]["moment_damping"] = 0.0
    source["mean_field"]["angle_damping"] = 1.0
    source["mean_field"]["schedule"]["type"] = "jacobi"
    source["initialization"]["parameters"]["values"] = [-math.pi / 4, math.pi / 8]
    source["local_solver"]["estimation"] = {"mode": "exact"}
    source["local_solver"]["optimizer"]["parameters"]["max_steps"] = 0
    source["moments"] = {
        "mode": "exact",
        "statistics": ["z_all", "zz_internal_edges"],
        "shared_sample": True,
    }
    source["final_correlations"] = {"mode": "exact"}
    value["limits"] = {"shots_total": 0, "max_block_updates_total": 4}
    value["checkpoint"] = {"enabled": False}
    value["collect"] = {
        "reduction_history": True,
        "message_transport_history": True,
        "episode_history": True,
        "final_correlation_tables": True,
    }
    value["output"] = {"report": 0, "logs": 0, "color": "never"}
    return value


def test_fixture_c_fresh_correlation_beats_bank_q():
    graph = fixture_c_graph()
    solver = DistributedLSRQAOACPU(fixture_c_config())
    solution = solver.solve(graph)
    assert solution == {10: 1, 30: -1}
    assert solver.cut_value == 2.0
    tables = solver.result["final_correlation_tables"]
    assert tables[0]["values"] == [[10, 30, -1.0]]
    assert solver.result["reduction_history"][0]["correlation"] == -1.0
    assert solver.result["reduction_history"][0]["sign"] == -1


def test_target_small_graph_runs_to_finisher():
    graph = WeightedGraph(
        [(10, 30, 5), (10, 80, 1), (30, 80, 1), (80, 101, 2), (101, 200, 3), (30, 200, -2)],
        nodes=[-7, 10, 30, 80, 101, 200],
    )
    value = config()
    value["checkpoint"] = {"enabled": False}
    value["output"] = {"report": 0, "logs": 0, "color": "never"}
    solver = DistributedLSRQAOACPU(value)
    solution = solver.solve(graph)
    assert set(solution) == set(graph.nodes)
    assert solver.result["status"] in ("completed", "stopped")
    assert solver.result["reductions_completed"] == 2
    assert solver.result["solution_source"] == "exact_finisher"
    assert solver.cut_value == graph.cut_value(solution)


def test_checkpoint_resume_preserves_parent_state(tmp_path):
    graph = fixture_c_graph()
    value = fixture_c_config()
    value["parameters"] = {"finish_at": 1, "max_reductions": 1}
    value["checkpoint"] = {
        "enabled": True,
        "path": str(tmp_path / "dist"),
        "every": {"optimizer_iterations": 1},
        "durable_resource_journal": True,
    }
    solver = DistributedLSRQAOACPU(value)
    solver.solve(graph)
    assert solver.last_checkpoint is not None
    restored = restore(solver.last_checkpoint.path)
    solution = restored.resume()
    assert solution == {10: 1, 30: -1}
    assert restored.result["reductions_completed"] == 1


def test_anchor_partition_is_complete_and_separate():
    graph = WeightedGraph([(10, 30, 2)], nodes=[-7, 10, 30, 80])
    value = config()
    value["partition_builder"] = {
        "type": "greedy_abs_weight_partition",
        "parameters": {"max_size": 2, "rounds": 1, "full_graph_when_fits": False},
    }
    solver = DistributedLSRQAOACPU.__new__(DistributedLSRQAOACPU)
    repair = solver._repair_partition(graph, value)
    assert repair is not None
    assert repair["repair"] is True
    flat = [node for block in repair["blocks"] for node in block]
    assert sorted(flat) == sorted(graph.nodes)
    assert len(flat) == len(set(flat))
    assert any(set(block) == {10, 30} for block in repair["blocks"])


def test_means_carry_keeps_survivor_and_resets_product():
    graph = WeightedGraph([(10, 30, 2), (10, 80, 3), (30, 80, 5)], nodes=[10, 30, 80, 101])
    from QA_max_cut_lib.core.reduction import GraphReducer
    from QA_max_cut_lib.solvers.block_mean_field import build_block_problem

    reducer = GraphReducer()
    state = reducer.start(graph)
    reducer.reduce(state, removed=80, retained=30, sign=-1, correlation=-0.6)
    average = {10: 0.2, 30: -0.4, 80: 0.8, 101: 0.6}
    carried = {node: average[node] for node in state.current_nodes}
    assert carried == {10: 0.2, 30: -0.4, 101: 0.6}
    problem = build_block_problem(state.current_graph(), [10, 30], carried)
    assert problem.couplings[(10, 30)] == pytest.approx(-0.5)
    # Product reinitialization q is the product of the carried means.
    assert carried[10] * carried[30] == pytest.approx(-0.08)
