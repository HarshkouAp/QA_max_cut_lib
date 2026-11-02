"""Compact real measurements for classical methods and refinement."""

from __future__ import annotations

from copy import deepcopy
import json
from time import perf_counter

from QA_max_cut_lib import (
    ExactCPU,
    ExactBlockRefiner,
    GoemansWilliamsonCPU,
    SimulatedAnnealingCPU,
    tools,
)


def graph(path):
    return __import__("QA_max_cut_lib").GraphFactory.from_recipe(tools.load_json(path))


def gw(path, hyperplanes=16):
    g = graph(path)
    cfg = tools.load_json("configs/solvers/gw_cpu.json")
    cfg["parameters"]["num_hyperplanes"] = hyperplanes
    cfg["collect"]["candidate_history"] = True
    start = perf_counter()
    s = GoemansWilliamsonCPU(cfg)
    sol = s.solve(g, report=0, logs=0)
    return {
        "n": len(g),
        "hyperplanes": hyperplanes,
        "seconds": perf_counter() - start,
        "cut": s.cut_value,
        "status": s.status,
        "sdp_status": s.sdp_status,
        "sdp_objective_raw": s.sdp_objective_raw,
        "sdp_objective_repaired": s.sdp_objective_repaired,
        "min_eigenvalue": s.sdp_diagnostics["minimum_eigenvalue"],
        "solution": sol,
    }


def sa(path, threads=1, reads=8, sweeps=32):
    g = graph(path)
    cfg = tools.load_json("configs/solvers/simulated_annealing_cpu.json")
    cfg["num_treads"] = threads
    cfg["parameters"] = {
        "num_reads": reads,
        "num_sweeps": sweeps,
        "beta_schedule_type": "geometric",
        "beta_range": [0.1, 4.0],
    }
    cfg["limits"] = {
        "anneal_reads_total": reads,
        "anneal_sweeps_total": reads * sweeps,
    }
    cfg["collect"]["candidate_history"] = True
    start = perf_counter()
    s = SimulatedAnnealingCPU(cfg)
    sol = s.solve(g, report=0, logs=0)
    return {
        "n": len(g),
        "threads": threads,
        "reads": reads,
        "sweeps": sweeps,
        "seconds": perf_counter() - start,
        "cut": s.cut_value,
        "status": s.status,
        "solution": sol,
    }


def refiner(path, max_size=6, max_blocks=4):
    g = graph(path)
    cfg = {
        "type": "exact_block",
        "parameters": {
            "max_size": max_size,
            "batch_size": 256,
            "max_passes": 1,
            "max_blocks_total": max_blocks,
        },
        "selector": {
            "type": "pair_greedy_abs_weight",
            "parameters": {"max_seed_pairs": max_blocks},
        },
    }
    start = perf_counter()
    r = ExactBlockRefiner(cfg)
    sol = r.refine(g, {node: 1 for node in g.nodes})
    return {
        "n": len(g),
        "max_size": max_size,
        "max_blocks": max_blocks,
        "seconds": perf_counter() - start,
        "cut": g.cut_value(sol),
        "record": r.last_refinement,
    }


if __name__ == "__main__":
    out = []
    exact_graph = graph("configs/graphs/complete_signed_7.json")
    exact = ExactCPU({"parameters": {"max_vertices": 7}})
    exact_solution = exact.solve(exact_graph, report=0, logs=0)
    out.append({"kind": "exact", "n": len(exact_graph), "cut": exact.cut_value, "solution": exact_solution})
    out.append({"kind": "gw", **gw("configs/graphs/erdos_signed_32.json")})
    out.append({"kind": "gw", **gw("configs/graphs/erdos_signed_64.json")})
    for threads in (1, 2, 4):
        out.append({"kind": "sa", **sa("configs/graphs/erdos_signed_128.json", threads=threads)})
    out.append({"kind": "sa", **sa("configs/graphs/erdos_signed_512.json", threads=1, reads=4, sweeps=16)})
    out.append({"kind": "refiner", **refiner("configs/graphs/erdos_signed_128.json")})
    out.append({"kind": "refiner", **refiner("configs/graphs/erdos_signed_512.json")})
    with open("docs/PERFORMANCE_CLASSICAL_0.1.0a8.json", "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    for item in out:
        print(item.get("kind"), item.get("n"), item.get("seconds"), item.get("cut"), item.get("status", ""))
