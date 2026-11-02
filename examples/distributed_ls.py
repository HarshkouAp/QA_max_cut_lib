"""Distributed LS-RQAOA example with fresh final correlations."""

from __future__ import annotations

import json
from pathlib import Path

from QA_max_cut_lib import WeightedGraph
from QA_max_cut_lib.laboratory import DistributedLSRQAOACPU

ROOT = Path(__file__).resolve().parents[1]


def main():
    graph = WeightedGraph(
        [(10, 30, 5), (10, 80, 1), (30, 80, 1), (80, 101, 2), (101, 200, 3), (30, 200, -2)],
        nodes=[-7, 10, 30, 80, 101, 200],
    )
    config = json.loads(
        (ROOT / "configs/solvers/distributed_ls_jacobi_cpu.json").read_text()
    )
    config["checkpoint"] = {"enabled": False}
    config["output"]["report"] = 0
    solver = DistributedLSRQAOACPU(config)
    solution = solver.solve(graph)
    print("solution:", solution)
    print("full cut:", solver.cut_value)
    print("reductions:", solver.result["reductions_completed"])
    print("source:", solver.result["solution_source"])
    print("shots:", solver.result["resources"]["shots_used"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
