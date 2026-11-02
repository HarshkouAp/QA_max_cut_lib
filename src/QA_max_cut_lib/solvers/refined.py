"""Base solver followed by exact block refinement."""

from __future__ import annotations

from copy import deepcopy
import math

from ..core.records import to_json_data
from .base import BaseSolver
from .exact_block_refiner import ExactBlockRefiner

__all__ = ["RefinedSolver"]


class RefinedSolver(BaseSolver):
    """Run one base solver, then improve its complete spins on the input graph.

    Parameters
    ----------
    config : dict
        ``base_solver`` is a normal solver configuration and ``refiner`` is
        an ``exact_block`` configuration.  ``limits`` can cap annealing reads,
        annealing sweeps, and exact refinement states.

    Attributes
    ----------
    base_run : dict or None
        Compact independent snapshot of the base solver run.
    refinement_run : dict or None
        Exact refiner record, including optional block history.
    base_cut_value, base_energy : float or None
        Base result before refinement.
    gain : float or None
        Refined cut minus base cut.
    solution_source : str
        ``refined`` or a base/refinement stop origin.

    Notes
    -----
    The base solver receives the original graph.  RQAOA and hierarchical
    methods therefore operate after their own complete spin restoration.
    Refinement is sequential and never silently applies parallel gains.
    Quantum state, angles, and expected values remain in ``base_run``.
    """

    algorithm_type = "refined"
    supports_parallel = True
    extra_config_fields = ("base_solver", "refiner")
    optional_fields = BaseSolver.optional_fields + (
        "base_run",
        "refinement_run",
        "base_cut_value",
        "base_energy",
        "gain",
        "solution_source",
        "base_solution",
    )

    def _validate(self, config, graph):
        if "base_solver" not in config:
            raise ValueError("base_solver is required.")
        if not isinstance(config["base_solver"], dict):
            raise TypeError("base_solver must be a mapping.")
        base = deepcopy(config["base_solver"])
        kind = base.get("type")
        if kind is None:
            raise ValueError("base_solver.type is required.")
        if kind == "refined":
            raise ValueError("Nested refined solvers are not supported.")
        config["base_solver"] = base
        if "refiner" not in config:
            raise ValueError("refiner is required.")
        refiner = ExactBlockRefiner(config["refiner"])
        config["refiner"] = refiner.config
        limits = config.setdefault("limits", {})
        allowed_limits = {
            "anneal_reads_total",
            "anneal_sweeps_total",
            "refinement_states_total",
        }
        if set(limits) - allowed_limits:
            raise ValueError("Unknown refined-solver limit.")
        for key in ("anneal_reads_total", "anneal_sweeps_total", "refinement_states_total"):
            limits.setdefault(key, None)
            if limits[key] is not None:
                value = limits[key]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"{key} must be an integer >= 0.")
        collect = config.setdefault("collect", {})
        allowed_collect = {"base_run", "candidate_history", "refinement_history"}
        if set(collect) - allowed_collect:
            raise ValueError("Unknown refined-solver collect field.")
        for key in allowed_collect:
            collect.setdefault(key, True)
            if type(collect[key]) is not bool:
                raise ValueError(f"collect.{key} must be boolean.")

    @staticmethod
    def _base_limits(config):
        limits = {}
        root = config["limits"]
        if root["anneal_reads_total"] is not None:
            limits["anneal_reads_total"] = root["anneal_reads_total"]
        if root["anneal_sweeps_total"] is not None:
            limits["anneal_sweeps_total"] = root["anneal_sweeps_total"]
        return limits

    @staticmethod
    def _compact_base_run(record, *, include_solution=True):
        if record is None:
            return None
        data = record.result
        fields = {
            "algorithm_type": data.get("algorithm_type"),
            "run_id": data.get("run_id"),
            "status": data.get("status"),
            "termination_reason": data.get("termination_reason"),
            "package_version": data.get("package_version"),
            "cut_value": data.get("cut_value"),
            "solution_energy": data.get("solution_energy"),
            "resources": data.get("resources"),
            "expected_cut": data.get("expected_cut"),
            "expected_energy": data.get("expected_energy"),
            "optimal_angles": record.get("optimal_angles"),
            "solution_source": data.get("solution_source"),
            "num_treads": data.get("num_treads"),
        }
        if include_solution:
            fields["solution"] = record.get("solution")
        return fields

    def _run(self, graph, config, rng):
        from ..core.factory import SolverFactory

        base_config = deepcopy(config["base_solver"])
        base_config.pop("save", None)
        base_config.pop("output", None)
        base_config["seed"] = config.get("seed")
        root_limits = self._base_limits(config)
        if base_config.get("type") == "simulated_annealing_cpu" and root_limits:
            merged = deepcopy(base_config.get("limits", {}))
            merged.update(root_limits)
            base_config["limits"] = merged
        base = SolverFactory.from_config(base_config)
        base_threads = int(config["num_treads"]) if base.supports_parallel else 1
        base_solution = base.solve(graph, report=0, logs=0, num_treads=base_threads)
        if set(base_solution) != set(graph.nodes):
            raise RuntimeError("The base solver did not return a complete assignment.")
        base_cut = graph.cut_value(base_solution)
        base_record = base.last_run
        base_run = self._compact_base_run(
            base_record, include_solution=bool(config["collect"]["base_run"])
        )
        refiner = ExactBlockRefiner(config["refiner"])
        refined = refiner.refine(
            graph,
            base_solution,
            context={"refinement_states_total": config["limits"]["refinement_states_total"]},
        )
        refinement = refiner.last_refinement
        if refinement is None:
            raise RuntimeError("The refiner did not produce a record.")
        final_cut = graph.cut_value(refined)
        gain = final_cut - base_cut
        base_status = base_record.get("status") if base_record is not None else None
        if base_status != "completed":
            status = "stopped"
            reason = base_record.get("termination_reason", "base_stopped")
        elif refinement["status"] != "completed":
            status = "stopped"
            reason = refinement["stop_reason"]
        else:
            status = "completed"
            reason = "completed"
        base_resources = deepcopy(base_record.get("resources", {})) if base_record else {}
        resources = {
            "shots_used": int(base_resources.get("shots_used", 0)),
            "base_shots": int(base_resources.get("shots_used", 0)),
            "base_seconds": base_record.get("timing", {}).get("compute_seconds")
            if base_record is not None
            else None,
            "refinement_seconds": refinement["refinement_seconds"],
            "num_treads_requested": int(config["num_treads"]),
            "num_treads_applied_base": base_threads,
            "base_resources": base_resources,
            "refinement_states_evaluated": refinement["states_evaluated"],
            "base_cut_value": float(base_cut),
            "gain": float(gain),
        }
        if config["collect"]["refinement_history"]:
            refinement_run = deepcopy(refinement)
        else:
            refinement_run = {key: deepcopy(value) for key, value in refinement.items() if key != "history"}
            refinement_run["history"] = None
        return {
            "solution": refined,
            "status": status,
            "termination_reason": reason,
            "solution_source": "refined" if gain >= 0 else "refined_negative",
            "base_run": base_run,
            "refinement_run": refinement_run,
            "base_cut_value": float(base_cut),
            "base_energy": float(-base_cut),
            "gain": float(gain),
            "base_solution": deepcopy(base_solution) if config["collect"]["base_run"] else None,
            "resources": resources,
            "expected_cut": None,
            "expected_energy": None,
            "optimal_angles": None,
            "data_status": {
                "base_run": "collected" if config["collect"]["base_run"] else "not_collected",
                "refinement_history": (
                    "collected" if config["collect"]["refinement_history"] else "not_collected"
                ),
            },
        }


def main(argv=None):
    """Run a small refined example through the common interface."""
    from ..tools.demo import run_solver_demo

    return run_solver_demo(RefinedSolver, argv)


if __name__ == "__main__":
    raise SystemExit(main())
