"""Run bounded module examples using the public configuration interface."""

import argparse
from ..generators.graphs import GraphFactory
from ..core.factory import load_json


def run_solver_demo(solver_class, argv=None):
    """Create a small weighted task, solve it, and print one report."""
    parser = argparse.ArgumentParser(description=solver_class.__name__)
    parser.add_argument("--graph", help="Graph recipe JSON path")
    parser.add_argument("--config", help="Solver configuration JSON path")
    parser.add_argument("--report", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--logs", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args(argv)
    recipe = (
        load_json(args.graph)
        if args.graph
        else {
            "type": "complete",
            "parameters": {"n_vertices": 6},
            "seed": 10,
            "weights": {
                "type": "signed_discrete",
                "parameters": {"values": [-2, -1, 1, 2]},
                "seed": 11,
            },
        }
    )
    config = (
        load_json(args.config)
        if args.config
        else {"type": solver_class.algorithm_type, "seed": 42}
    )
    solver = solver_class(config)
    solver.solve(GraphFactory.from_recipe(recipe), report=args.report, logs=args.logs)
    return 0
