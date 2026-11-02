"""Run a real GW-round warm-start RQAOA example."""

from __future__ import annotations

from QA_max_cut_lib import GraphFactory, tools
from QA_max_cut_lib.laboratory import WarmStartRQAOACPU


def main():
    graph = GraphFactory.from_recipe(tools.load_json("configs/graphs/warm_start_4.json"))
    config = tools.load_json("configs/solvers/warm_start_rqaoa_cpu_gw.json")
    solver = WarmStartRQAOACPU(config)
    solution = solver.solve(graph)
    print("solution:", solution)
    print("C:", solver.cut_value, "offset:", solver.cut_offset)
    print("reductions:", solver.reductions_completed)
    for history in solver.warm_start_history:
        print("source:", history["source"]["type"], "U/K:", history["source"]["unique"], "/", history["source"]["selected_starts"])
    print("shots:", solver.resources["shots_used"])
    print("source_cost:", solver.resources.get("source_cost"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
