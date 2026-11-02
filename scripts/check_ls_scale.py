"""Run a bounded LS-RQAOA check on a 1000-vertex graph."""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

from QA_max_cut_lib import GraphFactory, LSRQAOACPU


def main():
    root = Path(__file__).resolve().parents[1]
    recipe = {
        "schema_version": 1,
        "type": "erdos_renyi",
        "seed": 20261106,
        "parameters": {"n_vertices": 1000, "probability": 0.008},
        "weights": {
            "type": "uniform",
            "seed": 20261107,
            "parameters": {"low": -1.0, "high": 1.0},
        },
    }
    started = time.perf_counter()
    graph = GraphFactory.from_recipe(recipe)
    graph_seconds = time.perf_counter() - started
    config = {
        "schema_version": 1,
        "type": "ls_rqaoa_cpu",
        "seed": 20261108,
        "num_treads": 4,
        "parameters": {"finish_at": 6, "max_reductions": 2},
        "subgraph_builder": {
            "type": "greedy_abs_weight_partition",
            "parameters": {"max_size": 8, "rounds": 1, "full_graph_when_fits": True},
        },
        "correlation_source": {
            "type": "qaoa_analytic_p1_cpu",
            "parameters": {"layers": 1, "dtype": "float64"},
            "objective": {"type": "mean_cut", "parameters": {}},
            "objective_reference": "working_graph",
            "estimation": {"mode": "exact"},
            "optimizer": {
                "type": "gamma_grid_exact_beta",
                "parameters": {"gamma_bounds": [-3.141592653589793, 3.141592653589793], "points": 3},
            },
            "correlations": {"mode": "exact"},
        },
        "aggregation": {"type": "mean", "parameters": {}},
        "selection": {"type": "max_abs_correlation", "parameters": {"pairs": "edges"}},
        "coverage": {"require_full": False, "empty_policy": "anchor_pair"},
        "update": {"type": "refresh_all"},
        "execution": {"backend": "process", "start_method": "spawn", "native_threads": 1},
        "budget_policy": {"type": "step_round_robin"},
        "finisher": {"type": "exact_cpu", "parameters": {"max_vertices": 16}},
        "limits": {"shots_total": 0},
        "collect": {"reduction_history": True, "local_task_history": True},
    }
    solver = LSRQAOACPU(config)
    tracemalloc.start()
    started = time.perf_counter()
    solution = solver.solve(graph, report=0, logs=0)
    seconds = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    output = {
        "package_version": "0.1.0a5",
        "n_vertices": len(graph),
        "n_edges": len(graph.edges),
        "graph_seconds": graph_seconds,
        "status": solver.status,
        "termination_reason": solver.termination_reason,
        "reductions_completed": solver.reductions_completed,
        "local_tasks_total": solver.resources["local_tasks_total"],
        "local_tasks_skipped": solver.resources["local_tasks_skipped"],
        "num_treads_requested": solver.resources["num_treads_requested"],
        "num_treads_applied": solver.resources["num_treads_applied"],
        "seconds": seconds,
        "peak_tracemalloc_mib": peak / (1024 * 1024),
        "shots_used": solver.resources["shots_used"],
        "cut_value": solver.cut_value,
        "complete_solution": set(solution) == set(graph.nodes),
        "note": "Bounded check; max_reductions=2 and no claim of a full 1000-vertex solve.",
    }
    path = root / "docs/SCALE_LS_0.1.0a5.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(output)
    print(path)


if __name__ == "__main__":
    main()
