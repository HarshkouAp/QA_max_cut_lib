"""Standalone fixed-spin conditional QAOA example."""

from __future__ import annotations

import json
from pathlib import Path

from QA_max_cut_lib import WeightedGraph
from QA_max_cut_lib.laboratory import ConditionalQAOACPU

ROOT = Path(__file__).resolve().parents[1]


def main():
    graph = WeightedGraph(
        [(10, 30, 2), (10, 80, 3), (30, 80, -4), (30, 101, 5), (80, 101, -2)],
        nodes=[-7, 10, 30, 80, 101],
    )
    config = json.loads((ROOT / "configs/solvers/conditional_qaoa_cpu.json").read_text())
    config["output"]["report"] = 0
    solver = ConditionalQAOACPU(config)
    solution = solver.solve(graph)
    print("solution:", solution)
    print("full cut:", solver.cut_value)
    print("affected cut:", solver.affected_cut_value)
    print("outside cut:", solver.outside_cut)
    print("shots:", solver.result["resources"]["shots_used"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
