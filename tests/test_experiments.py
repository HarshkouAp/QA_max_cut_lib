"""Planner, runner, profile, and reference contract tests for Stage 09."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from QA_max_cut_lib import (
    BenchmarkSpec,
    ExperimentRunner,
    ParallelismProfiler,
    WeightedGraph,
    tools,
)
from QA_max_cut_lib.core.experiment_ids import graph_content_id, graph_instance_id
from QA_max_cut_lib.core.experiment_spec import PlanError, load_benchmark_spec
from QA_max_cut_lib.core.profile import ProfileSpec, build_profile_plan, fit_amdahl, karp_flatt
from QA_max_cut_lib.core.references import ReferencePolicy
from QA_max_cut_lib.core.series_analysis import annotate_trial_quality, reference_index


def small_spec(*, save=None, mode="isolated", backend="process"):
    data = {
        "schema_version": 1,
        "trial_seeds": [11, 29],
        "graph_cases": [
            {
                "name": "signed_small",
                "recipe": {
                    "schema_version": 1,
                    "type": "complete",
                    "parameters": {"n_vertices": 8},
                    "seed": 1000,
                    "weights": {
                        "type": "signed_discrete",
                        "parameters": {"values": [-3, -1, 1, 3]},
                        "seed": 1001,
                    },
                },
                "grid": [{"path": ["parameters", "n_vertices"], "values": [8, 10]}],
            }
        ],
        "solver_templates": {
            "qaoa_cpu_base": {
                "schema_version": 1,
                "type": "qaoa_cpu",
                "parameters": {"layers": 1},
                "objective": {"type": "mean_cut"},
                "estimation": {"mode": "shots", "shots_per_evaluation": 32},
                "optimizer": {
                    "type": "adam",
                    "parameters": {"max_steps": 2, "learning_rate": 0.05},
                    "gradient": {"type": "central_difference", "parameters": {"step": 0.01}},
                },
                "readout": {"mode": "best_of_shots", "shots": 64},
                "limits": {"shots_total": 2048},
            }
        },
        "solver_cases": [
            {
                "name": "qaoa_mean",
                "template": "qaoa_cpu_base",
                "grid": [
                    {"path": ["parameters", "layers"], "values": [1, 2]},
                    {"path": ["estimation", "shots_per_evaluation"], "values": [32, 64]},
                ],
            },
            {
                "name": "qaoa_cvar",
                "template": "qaoa_cpu_base",
                "overrides": [
                    {"path": ["objective"], "value": {"type": "cvar", "parameters": {"alpha": 0.1}}}
                ],
                "grid": [
                    {"path": ["parameters", "layers"], "values": [1, 2]},
                    {"path": ["estimation", "shots_per_evaluation"], "values": [32, 64]},
                    {"path": ["objective", "parameters", "alpha"], "values": [0.1, 0.5]},
                ],
            },
            {
                "name": "sa",
                "config": {
                    "schema_version": 1,
                    "type": "simulated_annealing_cpu",
                    "parameters": {
                        "num_reads": 3,
                        "num_sweeps": 16,
                        "beta_range": [0.1, 4.0],
                        "beta_schedule_type": "geometric",
                    },
                },
                "grid": [{"path": ["parameters", "num_reads"], "values": [3, 5]}],
            },
        ],
        "references": {
            "type": "tiered",
            "exact": {"max_vertices": 12, "include_minimum": False},
            "gw": {"max_vertices": 64, "config": {"type": "gw_cpu", "parameters": {"num_hyperplanes": 4}}, "seed": 9100},
            "best_known": {"sources": []},
        },
        "execution": {
            "mode": mode,
            "cpu_slots": 4,
            "solver_num_treads": 1,
            "backend": backend,
            "max_concurrent_trials": 2,
        },
        "measurement": {
            "timing_scope": "end_to_end",
            "solver_lifecycle": "fresh",
            "synchronize_gpu": True,
            "trial_output": {"report": 0, "logs": 0},
            "trial_save": False,
        },
        "series_limits": {"max_runs_total": 400, "shots_total": None},
        "metrics": {"success": {"type": "fraction_of_exact_max", "fraction": 0.9}},
        "failure_policy": {"on_trial_error": "continue", "retry_failed": False},
    }
    if save is not None:
        data["save"] = save
    return data


def test_spec_example_counts_and_conditional_branches(tmp_path):
    spec = load_benchmark_spec("experiments/benchmark_small.json")
    plan = spec.plan()
    assert plan.counts["graph_instances"] == 2
    assert plan.counts["cases"] == 28
    assert plan.counts["trials"] == 56
    assert plan.counts["reference_jobs"] == 2
    assert {case.graph_index for case in plan.cases} == {0, 1}
    for case in plan.cases:
        if case.solver_case in ("qaoa_mean", "sa"):
            if case.solver_case == "qaoa_mean":
                assert case.config["objective"] == {"type": "mean_cut"}
                assert "alpha" not in json.dumps(case.config)
        if case.solver_case == "qaoa_cvar":
            assert case.config["objective"]["type"] == "cvar"
    assert {case.graph_case for case in plan.cases} == {"signed_small"}
    assert spec.trial_seeds == [11, 29]


def test_identity_reorders_content_and_order():
    first = WeightedGraph([(-7, 10, 2), (10, 80, -3)], nodes=[-7, 10, 80, 101])
    second = WeightedGraph([(-7, 10, 2), (10, 80, -3)], nodes=[101, -7, 80, 10])
    assert graph_content_id(first) == graph_content_id(second)
    assert graph_instance_id(first) != graph_instance_id(second)
    changed = WeightedGraph([(-7, 10, 2), (10, 80, -3)], nodes=[-7, 10, 80])
    assert graph_content_id(first) != graph_content_id(changed)
    weighted = WeightedGraph([(-7, 10, 2), (10, 80, -3.0000001)], nodes=[-7, 10, 80, 101])
    assert graph_content_id(first) != graph_content_id(weighted)


def test_amdahl_karp_flatt_and_global_recommendation():
    points = [(1, 10.0), (2, 6.0), (4, 4.0), (8, 3.0)]
    fit = fit_amdahl(points)
    assert math.isclose(fit["serial_fraction"], 0.2, abs_tol=1e-12)
    assert math.isclose(fit["parallel_fraction"], 0.8, abs_tol=1e-12)
    assert fit["minimum_coverage"] is True
    assert math.isclose(karp_flatt(10, 12, 2), 1.4)
    assert math.isclose(karp_flatt(10, 4, 2), -0.2)
    assert karp_flatt(10, 10, 1) is None
    curve = [
        {"n": 1, "total_seconds": 26.0, "complete": True},
        {"n": 2, "total_seconds": 24.0, "complete": True},
        {"n": 4, "total_seconds": 29.0, "complete": True},
    ]
    rec = tools.recommend_total_curve(curve, tolerance_rel=0.03)
    assert rec["raw_argmin"] == 2 and rec["recommended_num_treads"] == 2


def test_exact_max_min_range_and_exact_mismatch():
    graph = WeightedGraph([(-7, 10, 2), (10, 80, -3)], nodes=[-7, 10, 80, 101])
    policy = ReferencePolicy({"type": "exact", "exact": {"max_vertices": 4, "include_minimum": True}})
    records = policy.run_all(policy.jobs_for(graph))
    assert [record["kind"] for record in records] == ["exact_max_cut", "exact_min_cut"]
    assert records[0]["cut_value"] == 2.0
    assert records[1]["cut_value"] == -3.0
    index = reference_index(records)
    solution = {-7: 1, 10: -1, 80: 1, 101: 1}
    trial = {
        "graph_content_id": graph_content_id(graph),
        "cut_value": graph.cut_value(solution),
        "solution_energy": -graph.cut_value(solution),
    }
    quality = annotate_trial_quality(trial, index, {"success": {"type": "fraction_of_exact_max", "fraction": 0.9}})
    assert quality["metrics"]["optimality_gap"] == 3.0
    assert math.isclose(quality["metrics"]["q_range"], 0.4)
    assert quality["metrics"]["ratio_to_exact_max"] == -0.5


def test_runner_real_small_series_and_save_reload(tmp_path):
    save = {
        "path": str(tmp_path),
        "filename": "series",
        "makedir": True,
        "savedir": True,
        "atts": ["trials"],
        "save_original_data": [],
    }
    spec = small_spec(save=save)
    series = ExperimentRunner(spec).run()
    assert series.analysis["planned"] == 56
    assert series.analysis["valid"] == 56
    assert series.analysis["failed"] == 0
    folder = Path(series.save_status["folder"])
    assert (folder / "series_summary.json").exists()
    assert len(list(folder.glob("trial_*.json"))) == 56
    assert series.analysis["by_solver"]["qaoa_mean"]["success"] is not None
    # Reuse must not execute another solver call and must preserve the id.
    second = ExperimentRunner(spec, reuse_completed=True).run()
    assert all(record["reuse"] for record in second.trials)
    loaded = tools.load_saved_series(folder)
    assert len(loaded["trials"]) == 56
    assert len(loaded["references"]) == 2


def test_profile_full_grid_shape_and_real_timing():
    spec = small_spec()
    config = {
        "schema_version": 1,
        "num_treads_values": [1, 2],
        "seed_index": 0,
        "warmup_runs": 0,
        "repetitions": 2,
        "execution_mode": "isolated",
        "timing_scope": "benchmark",
        "measurement_order": "fixed",
        "measurement_order_seed": 7,
        "models": ["amdahl", "karp_flatt"],
        "recommendation": {"criterion": "total_time", "tolerance_rel": 0.03},
        "series_limits": {"max_runs_total": 200},
    }
    plan = build_profile_plan(BenchmarkSpec(spec), ProfileSpec(config))
    assert plan.case_count == 28
    assert plan.points == 56
    assert len(plan.warmups) == 0
    assert len(plan.measured) == 112
    profile = ParallelismProfiler(spec, config).run()
    assert len(profile.measured_records) == 112
    assert len(profile.aggregate) == 2
    assert profile.aggregate[1]["complete"] is True
    assert profile.recommended_num_treads in (1, 2)
    assert profile.analysis["errors"] == []


def test_plot_reads_profile_without_solver(monkeypatch, tmp_path):
    spec = small_spec()
    config = {
        "num_treads_values": [1, 2],
        "seed_index": 0,
        "warmup_runs": 0,
        "repetitions": 1,
        "execution_mode": "isolated",
        "timing_scope": "benchmark",
        "measurement_order": "fixed",
        "recommendation": {"criterion": "total_time", "tolerance_rel": 0.03},
        "series_limits": {"max_runs_total": 60},
    }
    profile = ParallelismProfiler(spec, config).run()

    def forbidden(*args, **kwargs):
        raise AssertionError("A plot must not construct a solver.")

    monkeypatch.setattr("QA_max_cut_lib.core.factory.SolverFactory.from_config", forbidden)
    figure = tools.plot_parallelism(
        profile,
        {"style": {"figsize": [12, 7]}, "save": {"path": str(tmp_path), "filename": "plot", "format": "png", "DPI": 60}},
    )
    assert figure is not None
    assert (tmp_path / "plot_parallelism.png").exists()
