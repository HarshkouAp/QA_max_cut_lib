"""Run the provided-spin warm-start example through the public API."""

from __future__ import annotations

from QA_max_cut_lib import GraphFactory, tools
from QA_max_cut_lib.laboratory import WarmStartQAOACPU


def main():
    graph = GraphFactory.from_recipe(tools.load_json("configs/graphs/warm_start_4.json"))
    config = tools.load_json("configs/solvers/warm_start_qaoa_cpu_provided.json")
    solver = WarmStartQAOACPU(config)
    solution = solver.solve(graph)
    print("solution:", solution)
    print("cut:", solver.cut_value, "source_cut:", solver.source_cut)
    print("mixer:", solver.mixer_id, "shots:", solver.resources["shots_used"])
    print("checkpoint:", None if solver.last_checkpoint is None else solver.last_checkpoint.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
