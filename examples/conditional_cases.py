"""Small conditional-field cases: empty, singleton, full, p=2 and threshold."""

from __future__ import annotations

import json
import math
from pathlib import Path

from QA_max_cut_lib import WeightedGraph
from QA_max_cut_lib.laboratory import ConditionalQAOACPU

ROOT = Path(__file__).resolve().parents[1]


def base_config():
    return json.loads((ROOT / "configs/solvers/conditional_qaoa_cpu.json").read_text())


def main():
    graph = WeightedGraph(
        [(10, 30, 2), (10, 80, 3), (30, 80, -4), (30, 101, 5), (80, 101, -2)],
        nodes=[-7, 10, 30, 80, 101],
    )
    entries = [
        {"node": -7, "spin": -1},
        {"node": 10, "spin": 1},
        {"node": 30, "spin": 1},
        {"node": 80, "spin": 1},
        {"node": 101, "spin": -1},
    ]

    empty = base_config()
    empty["parameters"]["active_vertices"] = []
    empty["environment"]["source"] = {"type": "provided_spins", "parameters": {"entries": entries}}
    empty["optimizer"]["parameters"]["max_steps"] = 0
    empty["limits"]["shots_total"] = 0
    empty["readout"] = {"mode": "most_probable"}
    empty_solver = ConditionalQAOACPU(empty)
    print("empty:", empty_solver.solve(graph), empty_solver.result["termination_reason"])

    singleton_graph = WeightedGraph([(10, 80, 2)], nodes=[10, 80])
    singleton = base_config()
    singleton["parameters"]["active_vertices"] = [10]
    singleton["environment"]["source"] = {
        "type": "provided_spins",
        "parameters": {"entries": [{"node": 10, "spin": 1}, {"node": 80, "spin": 1}]},
    }
    singleton["initialization"]["parameters"]["values"] = [-math.pi / 4, math.pi / 4]
    singleton["optimizer"]["parameters"]["max_steps"] = 0
    singleton["limits"]["shots_total"] = 0
    singleton["readout"] = {"mode": "most_probable"}
    singleton_solver = ConditionalQAOACPU(singleton)
    print("singleton:", singleton_solver.solve(singleton_graph), singleton_solver.cut_value)

    full = base_config()
    full["parameters"]["active_vertices"] = [-7, 10, 30, 80, 101]
    full["parameters"]["layers"] = 2
    full["initialization"]["parameters"]["values"] = [0.2, 0.3, 0.4, 0.5]
    full["optimizer"]["parameters"]["max_steps"] = 0
    full["limits"]["shots_total"] = 0
    full["readout"] = {"mode": "most_probable"}
    full_solver = ConditionalQAOACPU(full)
    print("full:", full_solver.solve(graph), full_solver.cut_value)

    threshold = base_config()
    threshold["parameters"]["cut_offset"] = 7.0
    threshold["initialization"]["parameters"]["values"] = [0.0, 0.0]
    threshold["objective"] = {
        "type": "threshold_probability",
        "parameters": {"threshold": 12},
    }
    threshold["optimizer"]["parameters"]["max_steps"] = 0
    threshold["limits"]["shots_total"] = 0
    threshold["readout"] = {"mode": "most_probable"}
    threshold_solver = ConditionalQAOACPU(threshold)
    threshold_solver.solve(graph)
    print("threshold:", threshold_solver.objective_value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
