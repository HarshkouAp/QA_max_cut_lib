"""Provided-batch warm-start RQAOA control example A."""

from __future__ import annotations

import math

from QA_max_cut_lib import WeightedGraph
from QA_max_cut_lib.laboratory import WarmStartRQAOACPU


def main():
    graph = WeightedGraph(
        [(10, 30, 2), (10, 80, 3), (30, 80, 5)],
        nodes=[10, 30, 80, 101],
    )
    a = {10: 1, 30: 1, 80: -1, 101: 1}
    b = {10: 1, 30: -1, 80: 1, 101: 1}
    record = {
        "graph": graph.snapshot(),
        "assignments": [
            {"candidate_id": "A", "spins": a},
            {"candidate_id": "-A", "spins": {node: -spin for node, spin in a.items()}},
            {"candidate_id": "B", "spins": b},
            {"candidate_id": "A2", "spins": a},
        ],
    }
    config = {
        "schema_version": 1,
        "type": "laboratory.warm_start_rqaoa_cpu",
        "seed": 20261102,
        "parameters": {"finish_at": 3, "max_reductions": 1},
        "correlation_source": {
            "type": "warm_start_ensemble_cpu",
            "parameters": {"layers": 1, "max_vertices": 18, "dtype": "complex128"},
            "candidate_source": {
                "type": "provided_by_graph",
                "parameters": {"records": [record]},
            },
            "starts": {"max_starts": 2, "on_shortage": "use_available"},
            "warm_start": {"epsilon": 0.25, "mixer": {"type": "rounded_modified"}},
            "objective": {"type": "mean_cut"},
            "estimation": {"mode": "exact", "shots_per_evaluation": 32},
            "optimizer": {
                "type": "adam",
                "parameters": {"max_steps": 0, "learning_rate": 0.05, "gradient_tolerance": 0.0},
                "gradient": {"type": "central_difference", "parameters": {"step": 0.01}},
            },
            "initialization": {
                "type": "fixed_angles",
                "parameters": {"values": [0.0, math.pi / 2]},
            },
            "step_initialization": "restart",
            "correlations": {"mode": "exact", "aggregation": "uniform_components"},
        },
        "selection": {
            "type": "max_abs_correlation",
            "parameters": {"pairs": "edges", "zero_sign": 1},
        },
        "finisher": {"type": "exact_cpu", "parameters": {"max_vertices": 16}},
        "limits": {"shots_total": 0},
        "output": {"report": 0, "logs": 0},
    }
    solver = WarmStartRQAOACPU(config)
    solution = solver.solve(graph)
    print("solution:", solution)
    print("cut:", solver.cut_value, "offset:", solver.cut_offset, "E_Ising:", solver.ising_constant)
    print("reductions:", solver.reductions_completed, "source:", solver.solution_source)
    print("history:", solver.warm_start_history[0]["source"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
