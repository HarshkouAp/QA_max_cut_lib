"""Run a small example from the installed package."""

from pathlib import Path
from QA_max_cut_lib import GraphFactory, SolverFactory, ExactCPU
from QA_max_cut_lib import tools


def main():
    """Read JSON settings, solve the task, and compare the signed energy."""
    root = Path(__file__).resolve().parents[1]
    graph = GraphFactory.from_recipe(
        tools.load_json(root / "configs/graphs/complete_signed_7.json")
    )
    reference = ExactCPU()
    reference.solve(graph)
    solver = SolverFactory.from_config(
        tools.load_json(root / "configs/solvers/qaoa_cpu.json")
    )
    solver.solve(graph, report=1, logs=0)
    score = tools.quality_score(
        solver.cut_value, reference.cut_value, reference.worst_cut_value
    )
    print(
        f"Exact energy: {reference.solution_energy:g}; QAOA energy: {solver.solution_energy:g}; quality: {score:.3f}"
    )
    return solver.last_run


if __name__ == "__main__":
    main()
