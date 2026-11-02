"""Local timing and memory profile for the analytic p=1 evaluator.

This is a small report, not a general benchmark or a parallelism profiler.
Run it from the project root with the project CPU environment.
"""

from __future__ import annotations

from itertools import combinations
import json
import math
import time
import tracemalloc

import numpy as np

from QA_max_cut_lib import AnalyticP1Evaluator, GraphReducer, RQAOACPU, WeightedGraph


def make_graph(n, density, seed):
    rng = np.random.default_rng(seed)
    nodes = list(range(n))
    edges = []
    for u, v in combinations(nodes, 2):
        if rng.random() < density:
            weight = float(rng.choice([-2.0, -1.0, -0.25, 0.5, 1.0, 2.0]))
            edges.append((u, v, weight))
    return WeightedGraph(edges, nodes=nodes)


def analytic_root(n):
    return {
        "type": "rqaoa_cpu",
        "seed": 20261104,
        "num_treads": 1,
        "parameters": {"finish_at": 4, "max_reductions": None},
        "correlation_source": {
            "type": "qaoa_analytic_p1_cpu",
            "parameters": {
                "layers": 1,
                "dtype": "float64",
                "pair_batch_size": 2048,
            },
            "objective": {"type": "mean_cut", "parameters": {}},
            "objective_reference": "working_graph",
            "estimation": {"mode": "exact"},
            "optimizer": {
                "type": "gamma_grid_exact_beta",
                "parameters": {
                    "gamma_bounds": [-math.pi, math.pi],
                    "points": 41,
                    "gamma_batch_size": 4,
                },
            },
            "correlations": {"mode": "exact"},
        },
        "selection": {
            "type": "max_abs_correlation",
            "parameters": {"pairs": "edges", "zero_sign": 1},
        },
        "finisher": {
            "type": "exact_cpu",
            "parameters": {"max_vertices": 16, "batch_size": 65536},
        },
        "limits": {"shots_total": 0},
        "collect": {
            "reduction_history": False,
            "parameter_trajectory": False,
            "local_optimizer_history": False,
            "correlation_tables": False,
            "final_reduction_state": False,
        },
        "output": {"report": 0, "logs": 0, "color": "never"},
    }


def measure(fn):
    tracemalloc.start()
    started = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - started
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return value, elapsed, peak / (1024 * 1024)


def main():
    report = {"measurement": "local analytic p=1 profile", "cases": []}
    for n, density, seed in [
        (32, 0.20, 101),
        (64, 0.15, 102),
        (128, 0.10, 103),
        (256, 0.05, 104),
        (1000, 0.02, 105),
    ]:
        graph = make_graph(n, density, seed)
        evaluator, build_time, build_peak = measure(
            lambda: AnalyticP1Evaluator(graph, pair_batch_size=2048)
        )
        selected_result, one_gamma_time, one_gamma_peak = measure(
            lambda: evaluator.energy_min_beta(0.7)
        )
        if n <= 256:
            scan_value, scan_time, scan_peak = measure(
                lambda: evaluator.scan_gamma(
                    gamma_bounds=[-math.pi, math.pi],
                    points=41,
                    gamma_batch_size=4,
                )
            )
            best_gamma = float(scan_value["best_gamma"])
            best_index = int(scan_value["best_index"])
            best_beta = float(scan_value["beta_values"][best_index])
        else:
            scan_time = None
            scan_peak = None
            best_gamma = float(selected_result["gamma"])
            best_beta = float(selected_result["beta_star"])
        correlations, corr_time, corr_peak = measure(
            lambda: evaluator.correlations(
                best_gamma,
                best_beta,
                pair_domain="all_pairs" if n <= 128 else "edges",
            )
        )
        case = {
            "n": n,
            "density": density,
            "edges": len(graph.edges),
            "requested_pairs": len(correlations),
            "build_seconds": build_time,
            "build_peak_mib": build_peak,
            "one_gamma_seconds": one_gamma_time,
            "one_gamma_peak_mib": one_gamma_peak,
            "gamma_profile_41_seconds": scan_time,
            "gamma_profile_41_peak_mib": scan_peak,
            "correlations_seconds": corr_time,
            "correlations_peak_mib": corr_peak,
            "gamma": best_gamma,
            "beta": best_beta,
            "shots_used": 0,
        }
        if n == 32:
            started = time.perf_counter()
            solver = RQAOACPU(analytic_root(n))
            solution = solver.solve(graph)
            case["rqaoa_seconds"] = time.perf_counter() - started
            case["rqaoa_status"] = solver.status
            case["rqaoa_reductions"] = solver.reductions_completed
            case["rqaoa_cut"] = solver.cut_value
            case["rqaoa_shots_used"] = solver.resources["shots_used"]
            case["rqaoa_solution_size"] = len(solution)
        report["cases"].append(case)
        print(json.dumps(case, ensure_ascii=False))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
