"""Run a real classical warm-start source (GW or SA)."""

from __future__ import annotations

import argparse

from QA_max_cut_lib import GraphFactory, tools
from QA_max_cut_lib.laboratory import WarmStartQAOACPU


def main(argv=None):
    parser = argparse.ArgumentParser(description="Warm-start from a classical CPU solver")
    parser.add_argument("--source", choices=("gw", "sa"), default="gw")
    args = parser.parse_args(argv)
    filename = (
        "warm_start_qaoa_cpu_gw.json"
        if args.source == "gw"
        else "warm_start_qaoa_cpu_sa.json"
    )
    graph = GraphFactory.from_recipe(tools.load_json("configs/graphs/warm_start_4.json"))
    config = tools.load_json("configs/solvers/" + filename)
    solver = WarmStartQAOACPU(config)
    solution = solver.solve(graph)
    print("solution:", solution)
    print("cut:", solver.cut_value, "source_cut:", solver.source_cut)
    print("source:", solver.source_type, "mixer:", solver.mixer_id)
    print("source_cost:", solver.source_cost)
    print("quantum_shots:", solver.resources["shots_used"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
