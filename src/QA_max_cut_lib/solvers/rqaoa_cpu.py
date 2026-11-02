"""First complete CPU RQAOA built on the exact reduction core."""

from __future__ import annotations

from copy import deepcopy
import math

from ..core.budget import ShotBudget, ShotBudgetExceeded
from ..core.reduction import GraphReducer
from .base import BaseSolver, positive_int
from .components import (
    ExactCPUFinisher,
    MaxAbsCorrelationSelector,
    QAOAStatevectorCorrelationSource,
    validate_correlation_source_config,
)

__all__ = ["RQAOACPU"]


class RQAOACPU(BaseSolver):
    """Run recursive QAOA on the complete current graph.

    Parameters
    ----------
    config : dict, optional
        The root RQAOA configuration.  It has ``parameters``,
        ``correlation_source``, ``selection``, ``finisher``, ``limits``,
        ``collect``, ``output``, and optional ``save`` sections.

    Attributes
    ----------
    reductions_completed : int
        Number of constraints actually applied.
    remaining_node_order : list of int
        Active labels after the final reduction.
    cut_offset : float
        Accumulated ``b_t``.
    ising_constant : float
        Full constant of the final working Ising problem.
    reduced_solution : dict of int to int
        Spins of the final residual task.
    reduced_cut_value : float
        ``C`` of the residual solution on the working graph.
    solution_source : str
        Exact finisher, zero-interaction rule, or all-plus fallback.
    reduction_history : list of dict or None
        Compact core events when collected.
    parameter_trajectory : list of dict or None
        Angles before each applied reduction when collected.
    local_optimizer_history : list of dict or None
        Local training histories when collected.
    correlation_tables : list of dict or None
        Full current correlation tables when collected.
    final_reduction_state : dict or None
        JSON snapshot of the mathematical reduction state when collected.

    Notes
    -----
    One call uses one root shot budget.  The nested QAOA source has no
    independent limit.  The final exact residual solver uses no shots.
    The algorithm is heuristic: a correct reduction is not a promise of
    the global optimum of the original graph.
    """

    algorithm_type = "rqaoa_cpu"
    extra_config_fields = ("correlation_source", "selection", "finisher")
    optional_fields = BaseSolver.optional_fields + (
        "reductions_completed",
        "remaining_node_order",
        "cut_offset",
        "ising_constant",
        "reduced_solution",
        "reduced_cut_value",
        "solution_source",
        "reduction_history",
        "parameter_trajectory",
        "local_optimizer_history",
        "correlation_tables",
        "final_reduction_state",
    )

    def __init__(
        self,
        config=None,
        *,
        correlation_source=None,
        selector=None,
        finisher=None,
    ):
        """Create RQAOA and optionally inject test components.

        Parameters
        ----------
        config : dict, optional
            Root configuration.
        correlation_source, selector, finisher : object, optional
            Replacements for the default components.  They must follow
            the same method contracts.  This is the supported way to test
            the coordinator without running QAOA or exact search.
        """
        super().__init__(config)
        self._injected_source = correlation_source
        self._injected_selector = selector
        self._injected_finisher = finisher

    def _validate(self, config, graph):
        parameters = config.setdefault("parameters", {})
        if set(parameters) - {"finish_at", "max_reductions"}:
            raise ValueError("Unknown RQAOA parameter.")
        parameters.setdefault("finish_at", 4)
        parameters.setdefault("max_reductions", None)
        positive_int(parameters["finish_at"], "finish_at")
        finish_at = parameters["finish_at"]
        max_reductions = parameters["max_reductions"]
        if max_reductions is not None:
            if (
                isinstance(max_reductions, bool)
                or not isinstance(max_reductions, int)
                or max_reductions < 0
            ):
                raise ValueError("max_reductions must be None or an integer >= 0.")
        if "correlation_source" not in config:
            raise ValueError("correlation_source is required.")
        source_config = validate_correlation_source_config(config["correlation_source"])
        config["correlation_source"] = source_config
        if "selection" not in config:
            raise ValueError("selection is required.")
        selector = MaxAbsCorrelationSelector(config["selection"])
        config["selection"] = selector.config
        if "finisher" not in config:
            raise ValueError("finisher is required.")
        finisher = ExactCPUFinisher(config["finisher"])
        config["finisher"] = finisher.config
        if finish_at > finisher.max_vertices:
            raise ValueError("finish_at exceeds the finisher vertex limit.")
        limits = config.setdefault("limits", {})
        if set(limits) - {"shots_total"}:
            raise ValueError("Unknown RQAOA limit.")
        limits.setdefault("shots_total", None)
        if limits["shots_total"] is not None:
            positive_int(limits["shots_total"], "shots_total", minimum=0)
        collect = config.setdefault("collect", {})
        allowed_collect = {
            "reduction_history",
            "parameter_trajectory",
            "local_optimizer_history",
            "correlation_tables",
            "final_reduction_state",
        }
        if set(collect) - allowed_collect:
            raise ValueError(f"Unknown collect fields: {set(collect) - allowed_collect}")
        for key in allowed_collect:
            collect.setdefault(key, False)
            if type(collect[key]) is not bool:
                raise ValueError(f"collect.{key} must be boolean.")
        source_max = source_config["parameters"].get("max_vertices")
        has_interactions = any(weight != 0.0 for _, _, weight in graph.edges)
        if (
            source_config["type"] == "qaoa_statevector_cpu"
            and source_max is not None
            and len(graph) > source_max
            and has_interactions
            and len(graph) > finish_at
            and (max_reductions is None or max_reductions > 0)
        ):
            raise ValueError(
                "The state-vector source cannot handle the initial graph. "
                "Reduce the graph size or increase source max_vertices."
            )

    def _run(self, graph, config, rng):
        from ..tools.reporting import log_message

        reducer = GraphReducer()
        state = reducer.start(graph)
        budget = ShotBudget(config["limits"]["shots_total"])
        if self._injected_source is not None:
            source = self._injected_source
        elif config["correlation_source"]["type"] == "qaoa_statevector_cpu":
            source = QAOAStatevectorCorrelationSource(
                config["correlation_source"], seed=config.get("seed")
            )
        elif config["correlation_source"]["type"] == "qaoa_analytic_p1_cpu":
            from .analytic_p1 import AnalyticP1CorrelationSource

            source = AnalyticP1CorrelationSource(
                config["correlation_source"], seed=config.get("seed")
            )
        else:
            raise ValueError("Unknown correlation source type.")
        selector = self._injected_selector or MaxAbsCorrelationSelector(
            config["selection"]
        )
        finisher = self._injected_finisher or ExactCPUFinisher(config["finisher"])
        collect = config["collect"]
        finish_at = config["parameters"]["finish_at"]
        max_reductions = config["parameters"]["max_reductions"]

        reductions = 0
        source_attempts = 0
        source_calls = 0
        training_shots = 0
        correlation_shots = 0
        objective_evaluations = 0
        optimizer_steps = 0
        gamma_points = 0
        coefficient_passes = 0
        pair_correlators = 0
        classical_seconds = 0.0
        reduction_history = []
        parameter_trajectory = []
        local_optimizer_history = []
        correlation_tables = []
        status = "completed"
        termination_reason = "completed"
        solution_source = None
        residual_solution = None
        reduced_cut_value = None

        log_message(config, "RQAOA reduction loop starts.", detail=False)

        if len(graph) == 0:
            residual_solution = {}
            reduced_cut_value = 0.0
            solution_source = "empty_graph"
            termination_reason = "empty_graph"
        elif state.ising.edge_count == 0:
            residual_solution = {node: 1 for node in state.current_nodes}
            reduced_cut_value = 0.0
            solution_source = "zero_interaction_initial"
            termination_reason = "no_interactions"
        else:
            while True:
                n_active = len(state.current_nodes)
                if n_active <= finish_at:
                    if state.ising.edge_count == 0:
                        residual_solution = {node: 1 for node in state.current_nodes}
                        reduced_cut_value = 0.0
                        solution_source = "zero_interaction_residual"
                        termination_reason = "no_interactions"
                    else:
                        residual_graph = state.current_graph()
                        residual_solution, _ = finisher.solve(residual_graph)
                        reduced_cut_value = residual_graph.cut_value(residual_solution)
                        solution_source = "exact_finisher"
                        termination_reason = "finish_at"
                    status = "completed"
                    break

                if state.ising.edge_count == 0:
                    residual_solution = {node: 1 for node in state.current_nodes}
                    reduced_cut_value = 0.0
                    solution_source = "zero_interaction_residual"
                    termination_reason = "no_interactions"
                    status = "completed"
                    break

                if max_reductions is not None and reductions >= max_reductions:
                    residual_solution = {node: 1 for node in state.current_nodes}
                    reduced_cut_value = 0.0
                    solution_source = "all_plus_reduced_fallback"
                    termination_reason = "max_reductions"
                    status = "stopped"
                    break

                source_attempts += 1
                try:
                    correlations = source.estimate(
                        state,
                        budget=budget,
                        pair_domain=config["selection"]["parameters"]["pairs"],
                        collect_optimizer_history=collect["local_optimizer_history"],
                        collect_objective_history=False,
                    )
                except ShotBudgetExceeded:
                    residual_solution = {node: 1 for node in state.current_nodes}
                    reduced_cut_value = state.current_graph().cut_value(residual_solution)
                    solution_source = "all_plus_reduced_fallback"
                    termination_reason = "shot_budget"
                    status = "stopped"
                    break

                source_calls += 1
                source_resources = correlations.resources or {}
                training = correlations.training or {}
                training_shots += int(source_resources.get("training_shots", 0))
                correlation_shots += int(source_resources.get("correlation_shots", 0))
                objective_evaluations += int(
                    source_resources.get("objective_evaluations", 0)
                )
                optimizer_steps += int(source_resources.get("optimizer_steps", 0))
                gamma_points += int(source_resources.get("gamma_points", 0))
                coefficient_passes += int(
                    source_resources.get("coefficient_passes", 0)
                )
                pair_correlators += int(source_resources.get("pair_correlators", 0))
                classical_seconds += float(
                    source_resources.get("classical_seconds", 0.0)
                )
                if collect["local_optimizer_history"]:
                    local_optimizer_history.append(
                        {
                            "reduction_step": reductions + 1,
                            "task_version": correlations.task_version,
                            "angles": correlations.angles.tolist(),
                            "optimizer_history": training.get("optimizer_history"),
                            "profile_type": training.get("profile_type"),
                            "gamma_profile": training.get("gamma_profile"),
                        }
                    )
                if collect["correlation_tables"]:
                    correlation_tables.append(
                        {
                            "reduction_step": reductions + 1,
                            "task_version": correlations.task_version,
                            "node_order": list(correlations.node_order),
                            "mode": correlations.mode,
                            "pairs": [
                                {
                                    "u": u,
                                    "v": v,
                                    "physical_correlation": correlations.values[
                                        (u, v)
                                    ],
                                }
                                for u, v in correlations.pairs
                            ],
                        }
                    )
                if collect["parameter_trajectory"]:
                    parameter_trajectory.append(
                        {
                            "reduction_step": reductions + 1,
                            "task_version": correlations.task_version,
                            "layers": correlations.layers,
                            "angle_names": [
                                f"gamma_{index}"
                                for index in range(correlations.layers)
                            ]
                            + [
                                f"beta_{index}"
                                for index in range(correlations.layers)
                            ],
                            "angles": correlations.angles.tolist(),
                            "initialization": training.get("initialization"),
                            "source_type": getattr(source, "type", type(source).__name__),
                            "objective_reference": correlations.objective_reference,
                            "objective_value": training.get("objective_value"),
                            "objective_value_original": training.get(
                                "objective_value_original"
                            ),
                            "objective_threshold_requested": training.get(
                                "objective_threshold_requested"
                            ),
                            "objective_threshold_effective": training.get(
                                "objective_threshold_effective"
                            ),
                        }
                    )
                selection = selector.select(state, correlations)
                event = reducer.reduce(
                    state,
                    selection.removed,
                    selection.retained,
                    selection.sign,
                    correlation=selection.correlation,
                    selection_score=selection.score,
                )
                reductions += 1
                if collect["reduction_history"]:
                    event = deepcopy(event)
                    event["source"] = {
                        "type": getattr(source, "type", type(source).__name__),
                        "task_version": correlations.task_version,
                        "mode": correlations.mode,
                        "reduction_step": reductions,
                    }
                    reduction_history.append(event)
                log_message(
                    config,
                    f"Reduction {reductions}: {selection.removed}->{selection.retained} "
                    f"sign={selection.sign}; active={len(state.current_nodes)}",
                    detail=False,
                )

        if residual_solution is None:
            raise RuntimeError("RQAOA did not produce a residual solution.")
        solution = reducer.restore(state, residual_solution)
        cut_value = graph.cut_value(solution)
        energy = -cut_value
        cut_offset = state.cut_offset
        ising_constant = state.ising.constant
        if cut_offset is None:
            raise RuntimeError("A graph-origin reduction must have a cut offset.")
        expected_energy = -cut_offset - float(reduced_cut_value)
        scale = max(1.0, abs(energy), abs(expected_energy))
        if not math.isclose(energy, expected_energy, rel_tol=1e-9, abs_tol=1e-9 * scale):
            raise ArithmeticError("RQAOA energy and cut offset disagree.")
        resources = {
            "shots_used": budget.used,
            "training_shots": training_shots,
            "correlation_shots": correlation_shots,
            "readout_shots": 0,
            "objective_evaluations": objective_evaluations,
            "optimizer_steps": optimizer_steps,
            "source_attempts": source_attempts,
            "source_calls": source_calls,
            "reductions_completed": reductions,
            "num_stages": source_calls + 1,
            "num_treads": 1,
        }
        source_type = getattr(source, "type", type(source).__name__)
        if gamma_points or coefficient_passes or pair_correlators:
            resources.update(
                {
                    "gamma_points": gamma_points,
                    "coefficient_passes": coefficient_passes,
                    "pair_correlators": pair_correlators,
                    "classical_seconds": classical_seconds,
                }
            )
        finisher_type = getattr(finisher, "type", type(finisher).__name__)
        data_status = {
            key: ("collected" if collect[key] else "not_collected")
            for key in collect
        }
        log_message(
            config,
            f"RQAOA stops: {termination_reason}; reductions={reductions}; "
            f"shots={budget.used}.",
            detail=False,
        )
        return {
            "solution": solution,
            "status": status,
            "termination_reason": termination_reason,
            "reductions_completed": reductions,
            "remaining_node_order": list(state.current_nodes),
            "cut_offset": cut_offset,
            "ising_constant": ising_constant,
            "reduced_solution": {int(k): int(v) for k, v in residual_solution.items()},
            "reduced_cut_value": float(reduced_cut_value),
            "solution_source": solution_source,
            "reduction_history": reduction_history if collect["reduction_history"] else None,
            "parameter_trajectory": (
                parameter_trajectory if collect["parameter_trajectory"] else None
            ),
            "local_optimizer_history": (
                local_optimizer_history if collect["local_optimizer_history"] else None
            ),
            "correlation_tables": (
                correlation_tables if collect["correlation_tables"] else None
            ),
            "final_reduction_state": (
                state.snapshot() if collect["final_reduction_state"] else None
            ),
            "resources": resources,
            "data_status": data_status,
            "report": {
                "algorithm_type": self.algorithm_type,
                "reductions_completed": reductions,
                "solution_source": solution_source,
                "remaining_nodes": len(state.current_nodes),
                "cut_offset": cut_offset,
                "ising_constant": ising_constant,
                "source_type": source_type,
                "finisher_type": finisher_type,
            },
        }


def main(argv=None):
    """Run a small RQAOA example through the common demo interface."""
    from ..tools.demo import run_solver_demo

    return run_solver_demo(RQAOACPU, argv)


if __name__ == "__main__":
    raise SystemExit(main())
