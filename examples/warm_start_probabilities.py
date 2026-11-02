"""Run a fractional-probability warm-start example."""

from __future__ import annotations

from QA_max_cut_lib import GraphFactory, tools
from QA_max_cut_lib.laboratory import WarmStartQAOACPU


def main():
    graph = GraphFactory.from_recipe(tools.load_json("configs/graphs/warm_start_4.json"))
    config = tools.load_json("configs/solvers/warm_start_qaoa_cpu_probabilities.json")
    solver = WarmStartQAOACPU(config)
    solution = solver.solve(graph)
    print("solution:", solution)
    print("source_cut:", solver.source_cut)
    print("q_clipped:", solver.warm_start_data["q_clipped"])
    print("initial_expected_cut:", solver.initial_expected_cut)
    print("shots:", solver.resources["shots_used"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
