"""Pause a QAOA run, inspect the checkpoint, and resume it in a new object.

Run from the project root:

    python examples/checkpoint_resume.py

The example uses a real local checkpoint store.  Each normal solve gets a new
canonical run id; only ``resume()`` continues the stored run.
"""

from QA_max_cut_lib import GraphFactory, RunControl, SolverFactory, inspect_checkpoint, restore, tools


def main():
    graph_data = tools.load_json("configs/graphs/complete_signed_7.json")
    graph = GraphFactory.from_recipe(graph_data)
    config = tools.load_json("configs/solvers/qaoa_cpu_checkpoint.json")
    solver = SolverFactory.from_config(config)
    solver.solve(graph, control=RunControl(pause_after={"optimizer_iterations": 2}))
    assert solver.status == "paused", solver.status
    path = solver.last_checkpoint.path
    print("paused:", solver.resources["shots_used"], "shots;", path)
    print("inspect:", inspect_checkpoint(path))
    continued = restore(path)
    spins = continued.resume()
    print("resumed:", continued.status, continued.cut_value, "C;", continued.resources["shots_used"], "shots")
    print("complete solution:", spins)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
