"""Measure CPU hierarchy scale and thread modes."""

from __future__ import annotations

from copy import deepcopy
import json
import time

from QA_max_cut_lib import GraphFactory, QAOAInQAOACPU, tools


def config(max_steps=3, threads=1, max_size=8, finish_at=4):
    cfg = tools.load_json("configs/solvers/qaoa_in_qaoa_cpu.json")
    cfg.pop("save", None)
    cfg["num_treads"] = threads
    cfg["parameters"] = {"finish_at": finish_at, "max_levels": None}
    cfg["partitioner"]["parameters"]["max_size"] = max_size
    cfg["partitioner"]["parameters"]["full_graph_when_fits"] = False
    cfg["local_solver"]["parameters"]["max_vertices"] = max_size
    cfg["local_solver"]["optimizer"]["parameters"]["max_steps"] = max_steps
    cfg["local_solver"]["readout"] = {"mode": "most_probable"}
    cfg["collect"] = {
        "hierarchy_history": True,
        "block_patterns": True,
        "parameter_trajectory": True,
    }
    return cfg


def run(path, threads, max_steps=3, max_size=8):
    graph = GraphFactory.from_recipe(tools.load_json(path))
    started = time.perf_counter()
    solver = QAOAInQAOACPU(config(max_steps=max_steps, threads=threads, max_size=max_size))
    solution = solver.solve(graph, report=0, logs=0)
    elapsed = time.perf_counter() - started
    return {
        "path": path,
        "n": len(graph),
        "threads": threads,
        "max_size": max_size,
        "max_steps": max_steps,
        "seconds": elapsed,
        "cut": solver.cut_value,
        "levels": solver.levels_completed,
        "local_calls": solver.local_solver_calls,
        "primitive": solver.primitive_reductions_completed,
        "restored": len(solution),
        "source": solver.solution_source,
        "resources": solver.resources,
    }


if __name__ == "__main__":
    records = []
    for path in ("configs/graphs/erdos_signed_64.json", "configs/graphs/erdos_signed_128.json"):
        for threads in (1, 2, 4):
            records.append(run(path, threads))
    with open("docs/PERFORMANCE_HIERARCHY_0.1.0a7.json", "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    for item in records:
        print(item["n"], item["threads"], round(item["seconds"], 3), item["levels"], item["local_calls"], round(item["cut"], 6))
