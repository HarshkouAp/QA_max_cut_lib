"""Standalone QAOA on one fixed-spin conditional block."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from ..core.budget import ShotBudget
from .base import BaseSolver, positive_int
from .components import _validate_initialization, validate_adam_config
from .conditional import (
    ConditionalProblemError,
    build_conditional_problem,
    validate_reference,
)
from .objectives import validate_objective
from .qaoa_state import state_at_angles
from .qaoa_training import train_statevector

__all__ = ["ConditionalQAOACPU"]


class ConditionalQAOACPU(BaseSolver):
    """Train QAOA on an active block with fixed external spins.

    Parameters
    ----------
    config : dict, optional
        Configuration with ``parameters``, ``environment``, ``boundary``,
        ``objective``, ``objective_reference``, ``estimation``,
        ``optimizer``, ``initialization``, ``readout``, ``limits`` and
        ``collect`` sections.

    Notes
    -----
    External labels are classical fixed data.  The quantum state contains
    only active vertices and uses little-endian node order.  The result is
    completed to a full assignment and checked on the original graph.
    """

    algorithm_type = "laboratory.conditional_qaoa_cpu"
    supports_parallel = False
    extra_config_fields = ("environment", "boundary", "objective_reference")
    optional_fields = BaseSolver.optional_fields + (
        "active_nodes",
        "reference",
        "affected_cut_value",
        "outside_cut",
        "conditional_working_cut",
        "local_expected_energy",
        "local_expected_cut",
        "objective_reference",
        "objective_value",
        "local_problem",
        "boundary_type",
        "readout_completed",
    )

    def _validate(self, config, graph):
        p = config.setdefault("parameters", {})
        if set(p) - {"active_vertices", "layers", "max_active_vertices", "dtype", "cut_offset"}:
            raise ValueError("Unknown conditional-QAOA parameter.")
        p.setdefault("layers", 1)
        p.setdefault("max_active_vertices", 18)
        p.setdefault("dtype", "complex128")
        p.setdefault("cut_offset", 0.0)
        if not isinstance(p["cut_offset"], (int, float)) or isinstance(p["cut_offset"], bool) or not np.isfinite(p["cut_offset"]):
            raise ValueError("parameters.cut_offset must be finite.")
        positive_int(p["layers"], "layers")
        positive_int(p["max_active_vertices"], "max_active_vertices")
        if int(p["max_active_vertices"]) > 24:
            raise ValueError("The full-state conditional CPU cap is 24 active vertices.")
        if p["dtype"] not in ("complex64", "complex128"):
            raise ValueError("dtype must be complex64 or complex128.")
        active = p.get("active_vertices")
        if not isinstance(active, list):
            raise ValueError("parameters.active_vertices must be a list.")
        if len(active) > int(p["max_active_vertices"]):
            raise ValueError("active_vertices exceed max_active_vertices.")
        if len(set(active)) != len(active):
            raise ValueError("active_vertices contain duplicates.")
        for node in active:
            if isinstance(node, bool) or not isinstance(node, int) or node not in graph.nodes:
                raise ValueError(f"Active label is not in the graph: {node!r}")
        environment = config.setdefault("environment", {})
        if set(environment) - {"type", "source"} or environment.get("type") != "fixed_spins":
            raise ValueError("environment.type must be fixed_spins.")
        source = environment.setdefault("source", {"type": "all_plus"})
        if set(source) - {"type", "parameters"}:
            raise ValueError("Unknown environment source field.")
        if source.get("type") not in ("all_plus", "provided_spins"):
            raise ValueError("environment.source.type must be all_plus or provided_spins.")
        if source["type"] == "provided_spins":
            params = source.setdefault("parameters", {})
            if set(params) != {"entries"} or not isinstance(params["entries"], list):
                raise ValueError("provided_spins requires parameters.entries.")
        boundary = config.setdefault("boundary", {"type": "fixed_spins_conditional"})
        if set(boundary) - {"type"} or boundary.get("type") != "fixed_spins_conditional":
            raise ValueError("Only fixed_spins_conditional boundary is implemented.")
        config["objective"] = validate_objective(config.get("objective", {}))
        config.setdefault("objective_reference", "original_graph")
        if config["objective_reference"] not in (
            "conditional_affected_cut",
            "conditional_working_graph",
            "original_graph",
        ):
            raise ValueError("Unknown conditional objective_reference.")
        estimation = config.setdefault("estimation", {})
        if set(estimation) - {"mode", "shots_per_evaluation"}:
            raise ValueError("Unknown estimation field.")
        estimation.setdefault("mode", "exact")
        estimation.setdefault("shots_per_evaluation", 32)
        if estimation["mode"] not in ("exact", "shots"):
            raise ValueError("estimation.mode must be exact or shots.")
        positive_int(estimation["shots_per_evaluation"], "shots_per_evaluation")
        config["optimizer"] = validate_adam_config(config.get("optimizer", {}))
        if active:
            _validate_initialization(config, int(p["layers"]))
        else:
            init = config.setdefault("initialization", {"type": "fixed_angles"})
            init.setdefault("type", "fixed_angles")
            init.setdefault("parameters", {"values": [0.0] * (2 * int(p["layers"]))})
        readout = config.setdefault("readout", {})
        if set(readout) - {"mode", "shots"}:
            raise ValueError("Unknown readout field.")
        readout.setdefault("mode", "most_probable")
        if readout["mode"] == "best_of_shots":
            positive_int(readout.get("shots"), "readout.shots")
        elif readout["mode"] != "most_probable":
            raise ValueError("Unknown readout mode.")
        elif "shots" in readout:
            raise ValueError("most_probable does not consume readout shots.")
        limits = config.setdefault("limits", {})
        if set(limits) - {"shots_total"}:
            raise ValueError("Unknown conditional limit.")
        limits.setdefault("shots_total", None)
        if limits["shots_total"] is not None:
            positive_int(limits["shots_total"], "shots_total", minimum=0)
            if readout.get("shots", 0) > limits["shots_total"]:
                raise ValueError("The total shot budget cannot cover final readout.")
        collect = config.setdefault("collect", {})
        allowed = {"optimizer_history", "objective_history", "probabilities", "statevector"}
        if set(collect) - allowed:
            raise ValueError(f"Unknown conditional collect field: {set(collect) - allowed}")
        for key in allowed:
            collect.setdefault(key, False)
            if type(collect[key]) is not bool:
                raise ValueError(f"collect.{key} must be boolean.")

    def _reference(self, graph, config):
        source = config["environment"]["source"]
        if source["type"] == "all_plus":
            return {int(node): 1 for node in graph.nodes}
        entries = source["parameters"]["entries"]
        reference = {}
        for item in entries:
            if not isinstance(item, dict) or set(item) - {"node", "spin"}:
                raise ConditionalProblemError("Each provided entry needs node and spin.")
            node = item.get("node")
            if isinstance(node, bool) or not isinstance(node, int):
                raise ConditionalProblemError("Reference nodes must be integers.")
            if node in reference:
                raise ConditionalProblemError(f"Duplicate reference node: {node}")
            reference[node] = item.get("spin")
        return validate_reference(graph, reference)

    def _run(self, graph, config, rng):
        self._active_config = config
        p = config["parameters"]
        layers = int(p["layers"])
        reference = self._reference(graph, config)
        problem = build_conditional_problem(
            graph, p["active_vertices"], reference, cut_offset=float(p["cut_offset"])
        )
        budget = ShotBudget(config["limits"]["shots_total"])
        reserved = int(config["readout"].get("shots", 0))
        if budget.available() is not None and budget.available() < reserved:
            return self._fallback(
                reference,
                problem,
                status="stopped",
                reason="readout_resource",
            )
        if not problem.nodes:
            return self._result(
                reference,
                problem,
                chosen=0,
                training=None,
                budget=budget,
                readout_completed=False,
                status="completed",
                reason="empty_active_block",
            )
        reference_name = config["objective_reference"]
        if reference_name == "conditional_affected_cut":
            score_cuts = problem.affected_cuts
        elif reference_name == "conditional_working_graph":
            score_cuts = problem.working_cuts
        else:
            score_cuts = problem.original_cuts
        initial = np.asarray(
            config["initialization"]["parameters"].get(
                "values", [0.5] * layers + [0.3] * layers
            ),
            dtype=float,
        )
        if config["initialization"]["type"] != "fixed_angles":
            initial = np.r_[
                rng.uniform(-np.pi, np.pi, layers),
                rng.uniform(-np.pi / 2, np.pi / 2, layers),
            ]
        training = train_statevector(
            problem.affected_cuts,
            layers,
            objective=config["objective"],
            estimation_mode=config["estimation"]["mode"],
            shots_per_evaluation=config["estimation"]["shots_per_evaluation"],
            optimizer=config["optimizer"],
            initial_angles=initial,
            dtype=p["dtype"],
            rng=rng,
            budget=budget,
            reserved_shots=reserved,
            collect_optimizer_history=config["collect"]["optimizer_history"],
            collect_objective_history=config["collect"]["objective_history"],
            score_cuts=score_cuts,
        )
        probabilities = training["probabilities"]
        if reserved:
            if budget.available() is None or budget.available() >= reserved:
                if config["readout"]["mode"] == "best_of_shots":
                    indices = rng.choice(len(probabilities), size=reserved, p=probabilities)
                    chosen = int(indices[int(np.argmax(score_cuts[indices]))])
                else:
                    chosen = int(np.argmax(probabilities))
                budget.spend(reserved)
                readout_completed = True
            else:
                return self._fallback(
                    reference,
                    problem,
                    status="stopped",
                    reason="readout_resource",
                    training=training,
                )
        else:
            chosen = int(np.argmax(probabilities))
            readout_completed = True
        return self._result(
            reference,
            problem,
            chosen=chosen,
            training=training,
            budget=budget,
            readout_completed=readout_completed,
            status="completed" if training["reason"] != "shot_budget" else "stopped",
            reason=training["reason"],
        )

    def _fallback(self, reference, problem, *, status, reason, training=None):
        return self._result(
            reference,
            problem,
            chosen=0,
            training=training,
            budget=None,
            readout_completed=False,
            status=status,
            reason=reason,
            fallback=True,
        )

    def _result(
        self,
        reference,
        problem,
        *,
        chosen,
        training,
        budget,
        readout_completed,
        status,
        reason,
        fallback=False,
    ):
        solution = problem.completion(chosen) if problem.nodes else dict(reference)
        affected = float(problem.affected_cuts[chosen]) if problem.nodes else 0.0
        local_energy = -affected
        if training is None:
            state = None
            probabilities = None
            expected_cut = None
            objective_value = None
            optimizer_loss = None
            evaluations = 0
            shots = 0
            optimizer_steps = 0
        else:
            state = training["state"]
            probabilities = training["probabilities"]
            expected_cut = float(problem.affected_cuts @ probabilities)
            objective_value = training["objective_value"]
            optimizer_loss = training["best_loss"]
            if objective_value is None and probabilities is not None:
                from .objectives import objective_value as _criterion
                ref_name = self._active_config.get("objective_reference", "original_graph")
                score = {
                    "conditional_affected_cut": problem.affected_cuts,
                    "conditional_working_graph": problem.working_cuts,
                    "original_graph": problem.original_cuts,
                }[ref_name]
                objective_value = float(
                    _criterion(score, probabilities, self._active_config["objective"])
                )
                optimizer_loss = -objective_value
            evaluations = training["evaluations"]
            shots = training["shots"]
            optimizer_steps = training["optimizer_steps"]
        return {
            "solution": solution,
            "status": status,
            "termination_reason": reason,
            "active_nodes": list(problem.nodes),
            "reference": {int(k): int(v) for k, v in reference.items()},
            "boundary_type": "fixed_spins_conditional",
            "objective_reference": self._active_config.get("objective_reference", "original_graph"),
            "affected_cut_value": affected,
            "outside_cut": float(problem.outside_cut),
            "conditional_working_cut": float(problem.outside_cut + affected),
            "local_expected_cut": expected_cut,
            "local_expected_energy": None if expected_cut is None else -expected_cut,
            "objective_value": objective_value,
            "optimizer_loss": optimizer_loss,
            "optimal_angles": None if training is None else training["angles"],
            "optimizer_history": None if training is None else training["optimizer_history"],
            "objective_history": None if training is None else training["objective_history"],
            "probabilities": probabilities if self._active_config["collect"]["probabilities"] else None,
            "statevector": state if self._active_config["collect"]["statevector"] else None,
            "readout_completed": readout_completed,
            "local_problem": problem.to_dict(),
            "resources": {
                "shots_used": 0 if budget is None else budget.used,
                "training_shots": shots,
                "readout_shots": (
                    int(self._active_config["readout"].get("shots", 0))
                    if readout_completed
                    else 0
                ),
                "readout_reserved": (
                    0
                    if readout_completed
                    else int(self._active_config["readout"].get("shots", 0))
                ),
                "objective_evaluations": evaluations,
                "optimizer_steps": optimizer_steps,
                "num_treads": 1,
            },
        }


def main(argv=None):
    """Run a conditional QAOA example through the common CLI."""
    from ..tools.demo import run_solver_demo

    return run_solver_demo(ConditionalQAOACPU, argv)


if __name__ == "__main__":
    raise SystemExit(main())
