"""Incremental LS-RQAOA CPU example."""
from __future__ import annotations
import json
from itertools import combinations
from pathlib import Path
from QA_max_cut_lib import WeightedGraph
from QA_max_cut_lib.laboratory import IncrementalLSRQAOACPU

ROOT = Path(__file__).resolve().parents[1]

def main():
    labels = [-7, 10, 30, 80, 101, 205]
    pairs = list(combinations(labels, 2))
    graph = WeightedGraph([(u, v, (2 * i + 1) / 8) for i, (u, v) in enumerate(pairs)], nodes=labels)
    config = json.loads((ROOT / "configs/solvers/incremental_ls_keep.json").read_text())
    config["output"] = {"report": 0}
    solver = IncrementalLSRQAOACPU(config)
    solution = solver.solve(graph, num_treads=1)
    print("solution:", solution)
    print("cut:", solver.cut_value)
    print("reductions:", solver.result["reductions_completed"])
    print("tasks:", solver.result["resources"]["local_tasks_total"])
    print("shots:", solver.result["resources"]["shots_used"])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
