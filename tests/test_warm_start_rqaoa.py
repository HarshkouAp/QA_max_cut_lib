"""Contract tests for the warm-start ensemble RQAOA assembly."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from QA_max_cut_lib import WeightedGraph
from QA_max_cut_lib.core.budget import ShotBudget, ShotBudgetExceeded
from QA_max_cut_lib.laboratory import WarmStartRQAOACPU
from QA_max_cut_lib.solvers.warm_start_ensemble import (
    WarmStartEnsembleCorrelationSource,
    WarmStartEnsembleError,
)


def graph():
    return WeightedGraph(
        [(10, 30, 2), (10, 80, 3), (30, 80, 5)],
        nodes=[10, 30, 80, 101],
    )


def assignments():
    A = {10: 1, 30: 1, 80: -1, 101: 1}
    B = {10: 1, 30: -1, 80: 1, 101: 1}
    return A, B


def config(*, shots=False, max_steps=0, total=None, finish_at=3, max_starts=2):
    A, B = assignments()
    g = graph()
    record = {
        "graph": g.snapshot(),
        "assignments": [
            {"candidate_id": "A", "spins": A},
            {"candidate_id": "-A", "spins": {node: -spin for node, spin in A.items()}},
            {"candidate_id": "B", "spins": B},
            {"candidate_id": "A2", "spins": A},
        ],
    }
    return {
        "schema_version": 1,
        "type": "laboratory.warm_start_rqaoa_cpu",
        "seed": 20261102,
        "num_treads": 1,
        "parameters": {"finish_at": finish_at, "max_reductions": 2},
        "correlation_source": {
            "type": "warm_start_ensemble_cpu",
            "parameters": {"layers": 1, "max_vertices": 18, "dtype": "complex128"},
            "candidate_source": {
                "type": "provided_by_graph",
                "parameters": {"records": [record]},
            },
            "starts": {
                "max_starts": max_starts,
                "deduplication": "global_flip",
                "on_shortage": "use_available",
                "ranking": "current_cut_then_canonical_spins",
            },
            "warm_start": {"epsilon": 0.25, "mixer": {"type": "rounded_modified"}},
            "objective": {"type": "mean_cut", "parameters": {}},
            "objective_reference": "working_graph",
            "estimation": {
                "mode": "shots" if shots else "exact",
                "shots_per_evaluation": 32,
            },
            "optimizer": {
                "type": "adam",
                "parameters": {
                    "max_steps": max_steps,
                    "learning_rate": 0.05,
                    "gradient_tolerance": 0.0,
                },
                "gradient": {"type": "central_difference", "parameters": {"step": 0.01}},
            },
            "initialization": {
                "type": "fixed_angles",
                "parameters": {"values": [0.0, math.pi / 2]},
            },
            "step_initialization": "restart",
            "correlations": {
                "mode": "shots" if shots else "exact",
                **({"shots": 16} if shots else {}),
                "aggregation": "uniform_components",
            },
        },
        "selection": {
            "type": "max_abs_correlation",
            "parameters": {"pairs": "edges", "zero_sign": 1},
        },
        "finisher": {
            "type": "exact_cpu",
            "parameters": {"max_vertices": 16, "batch_size": 65536},
        },
        "limits": {"shots_total": 0 if total is None else total},
        "collect": {
            "reduction_history": True,
            "parameter_trajectory": False,
            "local_optimizer_history": False,
            "correlation_tables": True,
            "final_reduction_state": True,
        },
        "output": {"report": 0, "logs": 0, "color": "never"},
    }


def test_a_two_unique_starts_mixture_negative_edge_and_lift():
    g = graph()
    solver = WarmStartRQAOACPU(config())
    solution = solver.solve(g)
    assert solver.cut_value == 8
    assert solver.cut_offset == 8
    assert solver.ising_constant == -7.5
    assert solver.reductions_completed == 1
    assert solver.solution_source == "exact_finisher"
    assert g.cut_value(solution) == 8
    history = solver.warm_start_history[0]
    assert history["source"]["unique"] == 2
    assert history["source"]["selected_starts"] == 2
    assert history["children"][0]["multiplicity"] == 3
    assert history["children"][1]["multiplicity"] == 1
    assert history["children"][0]["current_cut"] == 8
    assert history["children"][1]["current_cut"] == 7


def test_g_round_robin_quotas_follow_actual_k():
    source = WarmStartEnsembleCorrelationSource(config(shots=True, total=368, max_steps=2)["correlation_source"], seed=1)
    iterations, reserve = source._quotas(ShotBudget(368), 3, source.config)
    assert iterations == [1, 1, 0]
    assert reserve == 48
    iterations, reserve = source._quotas(ShotBudget(368), 2, source.config)
    assert iterations == [1, 1]
    assert reserve == 32
    with pytest.raises(ShotBudgetExceeded):
        source._quotas(ShotBudget(47), 3, source.config)
    iterations, reserve = source._quotas(ShotBudget(48), 3, source.config)
    assert iterations == [0, 0, 0]
    assert reserve == 48


def test_shortage_error_and_wrong_graph_are_explicit():
    g = graph()
    cfg = config(max_starts=3)
    cfg["correlation_source"]["starts"]["on_shortage"] = "error"
    with pytest.raises(WarmStartEnsembleError):
        WarmStartRQAOACPU(cfg).solve(g)
    other = WeightedGraph([(1, 2, 1), (2, 3, 1), (3, 4, 1)], nodes=[1, 2, 3, 4])
    with pytest.raises(WarmStartEnsembleError):
        WarmStartRQAOACPU(config()).solve(other)


def test_h_budget_accounting_for_two_children():
    cfg = config(shots=True, max_steps=3, total=2048, finish_at=3)
    solver = WarmStartRQAOACPU(cfg)
    solver.solve(graph())
    resources = solver.resources
    assert resources["shots_used"] <= 2048
    assert resources["training_shots"] + resources["correlation_shots"] == resources["shots_used"]
    assert solver.warm_start_history[0]["children"][0]["completed_iterations"] <= 3


def test_real_gw_rounds_batch_has_one_sdp_and_bounded_rounds():
    pytest.importorskip("cvxpy")
    cfg = config()
    cfg["correlation_source"]["candidate_source"] = {
        "type": "gw_rounds_cpu",
        "parameters": {"num_hyperplanes": 4, "solver": "SCS", "allow_inaccurate": False},
    }
    solver = WarmStartRQAOACPU(cfg)
    solver.solve(graph())
    source = solver.warm_start_history[0]["source"]
    assert source["source_cost"]["sdp_calls"] == 1
    assert source["rounds_completed"] == 4
    assert source["source_cost"]["hyperplanes"] == 4
