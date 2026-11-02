"""Build the small Stage 12 comparison series."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "benchmark_warm_start_rqaoa.json"


def graph_case(name, seed, weight_seed):
    return {
        "name": name,
        "recipe": {
            "schema_version": 1,
            "type": "complete",
            "parameters": {"n_vertices": 4, "nodes": [10, 30, 80, 101]},
            "seed": seed,
            "weights": {
                "type": "signed_discrete",
                "parameters": {"values": [-2, -1, 1, 2]},
                "seed": weight_seed,
            },
        },
    }


def warm_source(max_starts):
    return {
        "type": "warm_start_ensemble_cpu",
        "parameters": {"layers": 1, "max_vertices": 18, "dtype": "complex128"},
        "candidate_source": {
            "type": "gw_rounds_cpu",
            "parameters": {"num_hyperplanes": 8, "solver": "SCS", "allow_inaccurate": False},
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
        "estimation": {"mode": "shots", "shots_per_evaluation": 32},
        "optimizer": {
            "type": "adam",
            "parameters": {"max_steps": 3, "learning_rate": 0.05, "gradient_tolerance": 0.0},
            "gradient": {"type": "central_difference", "parameters": {"step": 0.01}},
        },
        "initialization": {"type": "fixed_angles", "parameters": {"values": [0.4, 0.2]}},
        "step_initialization": "restart",
        "correlations": {"mode": "shots", "shots": 32, "aggregation": "uniform_components"},
    }


def warm_config(max_starts):
    return {
        "schema_version": 1,
        "type": "laboratory.warm_start_rqaoa_cpu",
        "seed": 20261102,
        "num_treads": 1,
        "parameters": {"finish_at": 2, "max_reductions": 2},
        "correlation_source": warm_source(max_starts),
        "selection": {"type": "max_abs_correlation", "parameters": {"pairs": "edges", "zero_sign": 1}},
        "finisher": {"type": "exact_cpu", "parameters": {"max_vertices": 8}},
        "limits": {"shots_total": 2048},
        "collect": {
            "reduction_history": True,
            "parameter_trajectory": False,
            "local_optimizer_history": False,
            "correlation_tables": False,
            "final_reduction_state": True,
        },
        "output": {"report": 0, "logs": 0, "color": "never"},
    }


def base_config():
    return {
        "schema_version": 1,
        "type": "rqaoa_cpu",
        "seed": 20261102,
        "num_treads": 1,
        "parameters": {"finish_at": 2, "max_reductions": 2},
        "correlation_source": {
            "type": "qaoa_statevector_cpu",
            "parameters": {"layers": 1, "max_vertices": 18, "dtype": "complex128"},
            "objective": {"type": "mean_cut"},
            "objective_reference": "working_graph",
            "estimation": {"mode": "shots", "shots_per_evaluation": 32},
            "optimizer": {
                "type": "adam",
                "parameters": {"max_steps": 3, "learning_rate": 0.05, "gradient_tolerance": 0.0},
                "gradient": {"type": "central_difference", "parameters": {"step": 0.01}},
            },
            "initialization": {"type": "fixed_angles", "parameters": {"values": [0.4, 0.2]}},
            "step_initialization": "restart",
            "correlations": {"mode": "shots", "shots": 32},
        },
        "selection": {"type": "max_abs_correlation", "parameters": {"pairs": "edges", "zero_sign": 1}},
        "finisher": {"type": "exact_cpu", "parameters": {"max_vertices": 8}},
        "limits": {"shots_total": 2048},
        "collect": {
            "reduction_history": True,
            "parameter_trajectory": False,
            "local_optimizer_history": False,
            "correlation_tables": False,
            "final_reduction_state": True,
        },
        "output": {"report": 0, "logs": 0, "color": "never"},
    }


data = {
    "schema_version": 1,
    "plan_seed": 1212,
    "trial_seeds": [31, 47],
    "graph_cases": [graph_case("signed_graph_a", 1200, 1201), graph_case("signed_graph_b", 1202, 1203)],
    "solver_cases": [
        {"name": "rqaoa_cpu", "config": base_config()},
        {"name": "ws_rqaoa_k1", "config": warm_config(1)},
        {"name": "ws_rqaoa_k2", "config": warm_config(2)},
    ],
    "references": {"type": "tiered", "exact": {"max_vertices": 8, "include_minimum": False}},
    "execution": {"mode": "isolated", "cpu_slots": 2, "solver_num_treads": 1, "gpu_policy": "exclusive_one_owner"},
    "measurement": {
        "timing_scope": "end_to_end",
        "solver_lifecycle": "fresh",
        "synchronize_gpu": True,
        "trial_output": {"report": 0, "logs": 0},
        "trial_save": False,
    },
    "series_limits": {"max_runs_total": 32, "shots_total": None},
    "metrics": {"fields": ["cut_value", "energy", "optimality_gap", "success"], "success": {"type": "fraction_of_exact_max", "fraction": 0.9}},
    "failure_policy": {"on_trial_error": "continue", "retry_failed": False},
    "collect": {"trial_records": True, "raw_measurements": True},
    "save": {
        "path": "results/series",
        "atts": ["expanded_plan", "trials", "references", "summary", "errors"],
        "filename": "warm_start_rqaoa_comparison",
        "save_original_data": [],
        "makedir": True,
        "savedir": True,
    },
}
OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("wrote", OUT)
