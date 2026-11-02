"""Solve small weighted Max-Cut tasks by complete enumeration."""

import numpy as np
from .base import BaseSolver, positive_int


class ExactCPU(BaseSolver):
    """Find the exact maximum cut with a bounded exhaustive search.

    Parameters
    ----------
    config : dict, optional
        type='exact_cpu'. parameters.max_vertices defaults to 24 and
        parameters.batch_size defaults to 65536. See the common fields
        in BaseSolver and the complete API reference.

    Attributes
    ----------
    worst_cut_value : float
        The minimum cut value over all spin assignments.
    optimal_count : int
        Number of optimal full spin assignments within tie_tolerance.
    states_evaluated : int
        Number of evaluations across both passes. A representative is
        evaluated once to find the extrema and once to count ties.

    Notes
    -----
    The global spin flip leaves each product s_u*s_v unchanged.
    Fix the first spin to +1 to search 2**(N-1) representatives.
    This symmetry is valid because the public task has no local fields.
    Memory is O(batch_size), apart from the graph. Time is O(E*2**N).
    """

    algorithm_type = "exact_cpu"

    def _validate(self, config, graph):
        if any(
            key in config
            for key in (
                "objective",
                "estimation",
                "optimizer",
                "initialization",
                "readout",
            )
        ):
            raise ValueError("Quantum training options do not apply to ExactCPU.")
        if config.get("collect"):
            raise ValueError(
                "ExactCPU has no optional quantum-state collection in this milestone."
            )
        if set(config.get("limits", {})) - {"shots_total"}:
            raise ValueError("Unknown exact-solver limit.")
        p = config.setdefault("parameters", {})
        p.setdefault("max_vertices", 24)
        p.setdefault("batch_size", 65536)
        p.setdefault("tie_tolerance", 0.0)
        if set(p) - {"max_vertices", "batch_size", "tie_tolerance"}:
            raise ValueError("Unknown exact solver parameter.")
        limit = positive_int(p.get("max_vertices", 24), "max_vertices")
        if limit > 30:
            raise ValueError("The first exact implementation caps max_vertices at 30.")
        if len(graph) > limit:
            raise ValueError(f"Exact search limit is {limit} vertices.")
        positive_int(p.get("batch_size", 65536), "batch_size")
        tol = p.get("tie_tolerance", 0.0)
        if not np.isfinite(tol) or tol < 0:
            raise ValueError("tie_tolerance must be finite and nonnegative.")
        config.setdefault("collect", {})

    def _run(self, graph, config, rng):
        p = config.get("parameters", {})
        n = len(graph)
        size = 1 << max(0, n - 1)
        batch_size = p.get("batch_size", 65536)
        order = {node: i for i, node in enumerate(graph.nodes)}
        best = -np.inf
        worst = np.inf
        best_index = 0
        count = 0
        batches = 0
        tolerance = p.get("tie_tolerance", 0.0)
        for start in range(0, size, batch_size):
            indices = np.arange(start, min(size, start + batch_size), dtype=np.uint64)
            # Bit zero is the fixed first spin. Shift all free bits by one.
            states = indices << np.uint64(1)
            cuts = np.zeros(len(states), dtype=np.float64)
            for u, v, weight in graph.edges:
                different = (
                    (states >> np.uint64(order[u])) ^ (states >> np.uint64(order[v]))
                ) & np.uint64(1)
                cuts += weight * different
            if not np.isfinite(cuts).all():
                raise ValueError("Cut evaluation overflow.")
            local = float(cuts.max())
            if local > best:
                best = local
                best_index = int(indices[int(cuts.argmax())])
                # Recount ties after the maximum is known below.
            worst = min(worst, float(cuts.min()))
            batches += 1
        # A second pass gives an exact count relative to the final maximum.
        # Do not retain all 2**N costs only to count degenerate solutions.
        for start in range(0, size, batch_size):
            states = np.arange(
                start, min(size, start + batch_size), dtype=np.uint64
            ) << np.uint64(1)
            cuts = np.zeros(len(states))
            for u, v, weight in graph.edges:
                cuts += weight * (
                    ((states >> np.uint64(order[u])) ^ (states >> np.uint64(order[v])))
                    & np.uint64(1)
                )
            count += int(np.count_nonzero(np.abs(cuts - best) <= tolerance))
        bits = best_index << 1
        solution = {
            node: 1 - 2 * ((bits >> i) & 1) for i, node in enumerate(graph.nodes)
        }
        return {
            "solution": solution,
            "worst_cut_value": worst,
            "optimal_count": count * (2 if n else 1),
            "states_evaluated": size * 2,
            "status": "completed",
            "termination_reason": "exhaustive_search",
            "resources": {
                "shots_used": 0,
                "states_evaluated": size * 2,
                "num_stages": batches * 2,
                "num_treads": 1,
            },
            "data_status": {
                "statevector": "not_applicable",
                "probabilities": "not_applicable",
                "optimizer_history": "not_applicable",
            },
        }


def main(argv=None):
    """Run a bounded exact-solver example through the common interface."""
    from ..tools.demo import run_solver_demo

    return run_solver_demo(ExactCPU, argv)


if __name__ == "__main__":
    raise SystemExit(main())
