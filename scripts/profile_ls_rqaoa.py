"""Measure a bounded LS-RQAOA run without changing scientific settings."""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

from QA_max_cut_lib import GraphFactory, LSRQAOACPU


def main():
    root = Path(__file__).resolve().parents[1]
    recipe = json.loads((root / "configs/graphs/erdos_signed_64.json").read_text(encoding="utf-8"))
    graph = GraphFactory.from_recipe(recipe)
    base = json.loads((root / "configs/solvers/ls_rqaoa_analytic.json").read_text(encoding="utf-8"))
    base.pop("save", None)
    base["parameters"] = {"finish_at": 6, "max_reductions": None}
    base["subgraph_builder"]["parameters"].update(
        {"max_size": 10, "rounds": 2, "full_graph_when_fits": True}
    )
    base["correlation_source"]["optimizer"]["parameters"]["points"] = 9
    base["collect"] = {
        "reduction_history": True,
        "parameter_trajectory": True,
        "local_task_history": True,
    }
    rows = []
    reference = None
    for threads in (1, 2, 4):
        solver = LSRQAOACPU({**base, "num_treads": threads})
        tracemalloc.start()
        started = time.perf_counter()
        solution = solver.solve(graph, report=0, logs=0)
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        signature = sorted(solution.items())
        if reference is None:
            reference = signature
        rows.append(
            {
                "requested_threads": threads,
                "applied_threads": solver.resources["num_treads_applied"],
                "seconds": elapsed,
                "peak_tracemalloc_mib": peak / (1024 * 1024),
                "cut_value": solver.cut_value,
                "solution_energy": solver.solution_energy,
                "reductions_completed": solver.reductions_completed,
                "local_tasks_total": solver.resources["local_tasks_total"],
                "local_tasks_skipped": solver.resources["local_tasks_skipped"],
                "shots_used": solver.resources["shots_used"],
                "same_solution_as_first": signature == reference,
            }
        )
        print(rows[-1], flush=True)
    output = {
        "package_version": "0.1.0a5",
        "graph": {"n_vertices": len(graph), "n_edges": len(graph.edges)},
        "source": "qaoa_analytic_p1_cpu",
        "builder": "greedy_abs_weight_partition",
        "max_size": 10,
        "rounds": 2,
        "gamma_points": 9,
        "finish_at": 6,
        "rows": rows,
    }
    path = root / "docs/PERFORMANCE_LS_0.1.0a5.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
