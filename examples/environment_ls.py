"""Environment-aware LS-RQAOA example with a frozen all-plus reference."""

from __future__ import annotations

import json
from pathlib import Path

from QA_max_cut_lib import WeightedGraph
from QA_max_cut_lib.laboratory import EnvironmentLSRQAOACPU

ROOT = Path(__file__).resolve().parents[1]


def main():
    graph = WeightedGraph(
        [(10, 30, 2), (10, 80, 3), (30, 80, 5)],
        nodes=[10, 30, 80, 101],
    )
    config = json.loads((ROOT / "configs/solvers/environment_ls_rqaoa_cpu.json").read_text())
    config["output"]["report"] = 0
    solver = EnvironmentLSRQAOACPU(config)
    solution = solver.solve(graph)
    print("solution:", solution)
    print("full cut:", solver.cut_value)
    print("reductions:", solver.result["reductions_completed"])
    print("source:", solver.result["solution_source"])
    print("shots:", solver.result["resources"]["shots_used"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
