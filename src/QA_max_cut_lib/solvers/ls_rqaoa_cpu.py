"""First complete CPU LS-RQAOA with real local tasks and process execution."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
import os
from time import perf_counter

from ..core.budget import ShotBudget
from ..core.reduction import GraphReducer
from ..generators.subgraphs import build_subgraph_builder, greedy_anchor_block
from .base import BaseSolver, positive_int
from .components import ExactCPUFinisher, validate_correlation_source_config
from .ls_components import (
    MeanCorrelationAggregator,
    ObservedMaxAbsCorrelationSelector,
    build_correlation_source,
)
from .ls_worker import solve_local_task

__all__ = ["LSRQAOACPU"]


def _stable_seed(root, *parts):
    if root is None:
        return None
    material = "|".join([str(root)] + [str(part) for part in parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def _canonical_pair(u, v):
    return (min(int(u), int(v)), max(int(u), int(v)))


def _allowed_pairs(state, pairs):
    if pairs == "edges":
        return sorted(
            _canonical_pair(u, v)
            for u, v, weight in state.current_graph().edges
            if float(weight) != 0.0
        )
    nodes = sorted(int(node) for node in state.current_nodes)
    return [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes))]


def _task_pairs(nodes, allowed):
    chosen = set(int(node) for node in nodes)
    return [pair for pair in allowed if pair[0] in chosen and pair[1] in chosen]


def _task_requirements(source_config):
    """Return mandatory shots, iteration cost, and configured iterations."""
    if source_config["type"] == "qaoa_analytic_p1_cpu":
        return 0, 0, 0
    parameters = source_config["parameters"]
    layers = parameters["layers"]
    if source_config["estimation"]["mode"] == "shots":
        iteration_cost = (4 * layers + 1) * source_config["estimation"][
            "shots_per_evaluation"
        ]
    else:
        iteration_cost = 0
    if source_config["correlations"]["mode"] == "shots":
        mandatory = int(source_config["correlations"]["shots"])
    else:
        mandatory = 0
    iterations = int(source_config["optimizer"]["parameters"]["max_steps"])
    return mandatory, iteration_cost, iterations


def _round_robin_quotas(mandatory, costs, limits, total, used):
    """Allocate one full iteration per task per round.

    Returns ``None`` when the mandatory correlation samples do not fit.
    """
    reserve = sum(mandatory)
    if total is not None and total - used < reserve:
        return None
    remaining = None if total is None else total - used - reserve
    steps = [0] * len(mandatory)
    while True:
        progress = False
        for index, limit in enumerate(limits):
            if steps[index] >= limit:
                continue
            cost = costs[index]
            if cost and remaining is not None and remaining < cost:
                continue
            steps[index] += 1
            if cost and remaining is not None:
                remaining -= cost
            progress = True
        if not progress:
            break
    quotas = [mandatory[i] + costs[i] * steps[i] for i in range(len(mandatory))]
    return steps, quotas


def _execute_bounded(executor, payloads, workers):
    """Run payloads with a bounded submission window.

    The returned list is in payload order, independent of completion order.
    The caller owns the executor and closes it in ``finally``.
    """
    from concurrent.futures import FIRST_COMPLETED, wait

    iterator = iter(enumerate(payloads))
    pending = {}
    results = [None] * len(payloads)
    for _ in range(min(max(1, workers) * 2, len(payloads))):
        try:
            index, payload = next(iterator)
        except StopIteration:
            break
        pending[executor.submit(solve_local_task, payload)] = index
    while pending:
        done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
        for future in done:
            index = pending.pop(future)
            results[index] = future.result()
            try:
                next_index, payload = next(iterator)
            except StopIteration:
                continue
            pending[executor.submit(solve_local_task, payload)] = next_index
    return results


class LSRQAOACPU(BaseSolver):
    """Run local-search RQAOA on the complete current graph.

    Parameters
    ----------
    config : dict, optional
        Root configuration with ``parameters``, ``subgraph_builder``,
        ``correlation_source``, ``aggregation``, ``selection``,
        ``coverage``, ``update``, ``execution``, ``budget_policy``,
        ``finisher``, ``limits``, and ``collect`` sections.

    Attributes
    ----------
    reductions_completed : int
        Number of global pair constraints applied.
    remaining_node_order : list of int
        Active labels in the final residual task.
    cut_offset : float
        Accumulated Max-Cut offset ``b_t``.
    ising_constant : float
        Constant of the final working Ising problem.
    reduced_solution : dict
        Spins of the final local residual task.
    local_task_history : list or None
        Compact records for all evaluated local tasks.
    coverage_history : list or None
        Allowed and observed pair coverage at every step.
    parameter_trajectory : list or None
        One row per local task with its own angles.

    Notes
    -----
    A local task is an induced subgraph of the current graph.  Boundary
    edges are intentionally omitted.  The selected pair is reduced on the
    full working graph with the shared Stage 02 reducer.  A local
    correlation is not promoted to a global QAOA expectation.
    """

    algorithm_type = "ls_rqaoa_cpu"
    supports_parallel = True
    extra_config_fields = (
        "subgraph_builder",
        "correlation_source",
        "aggregation",
        "selection",
        "coverage",
        "update",
        "execution",
        "budget_policy",
        "finisher",
    )
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
        "local_task_history",
        "local_optimizer_history",
        "correlation_tables",
        "coverage_history",
        "final_reduction_state",
    )

    def _validate(self, config, graph):
        parameters = config.setdefault("parameters", {})
        if set(parameters) - {"finish_at", "max_reductions"}:
            raise ValueError("Unknown LS-RQAOA parameter.")
        parameters.setdefault("finish_at", 6)
        parameters.setdefault("max_reductions", None)
        positive_int(parameters["finish_at"], "finish_at")
        max_reductions = parameters["max_reductions"]
        if max_reductions is not None and (
            isinstance(max_reductions, bool)
            or not isinstance(max_reductions, int)
            or max_reductions < 0
        ):
            raise ValueError("max_reductions must be None or an integer >= 0.")

        builder_config = config.setdefault(
            "subgraph_builder",
            {
                "type": "greedy_abs_weight_partition",
                "parameters": {"max_size": 10, "rounds": 2},
            },
        )
        builder = build_subgraph_builder(builder_config, seed=config.get("seed"))
        config["subgraph_builder"] = builder.config

        if "correlation_source" not in config:
            config["correlation_source"] = {
                "type": "qaoa_analytic_p1_cpu",
                "parameters": {"layers": 1, "dtype": "float64"},
                "objective": {"type": "mean_cut", "parameters": {}},
                "objective_reference": "working_graph",
                "estimation": {"mode": "exact"},
                "optimizer": {
                    "type": "gamma_grid_exact_beta",
                    "parameters": {
                        "gamma_bounds": [-3.141592653589793, 3.141592653589793],
                        "points": 17,
                    },
                },
                "correlations": {"mode": "exact"},
            }
        source_config = validate_correlation_source_config(config["correlation_source"])
        if source_config.get("objective_reference") != "working_graph":
            raise ValueError("LS local objectives must use objective_reference='working_graph'.")
        config["correlation_source"] = source_config

        aggregation = MeanCorrelationAggregator(config.get("aggregation"))
        config["aggregation"] = aggregation.config

        selection = ObservedMaxAbsCorrelationSelector(
            config.get("selection", {"type": "max_abs_correlation"})
        )
        config["selection"] = selection.config

        coverage = config.setdefault("coverage", {})
        if set(coverage) - {"require_full", "empty_policy"}:
            raise ValueError("Unknown coverage field.")
        coverage.setdefault("require_full", False)
        coverage.setdefault("empty_policy", "anchor_pair")
        if type(coverage["require_full"]) is not bool:
            raise ValueError("coverage.require_full must be boolean.")
        if coverage["empty_policy"] != "anchor_pair":
            raise ValueError("Only empty_policy='anchor_pair' is implemented.")

        update = config.setdefault("update", {})
        if set(update) - {"type"} or update.setdefault("type", "refresh_all") != "refresh_all":
            raise ValueError("Only update.type='refresh_all' is implemented.")

        execution = config.setdefault("execution", {})
        if set(execution) - {"backend", "start_method", "native_threads"}:
            raise ValueError("Unknown execution field.")
        execution.setdefault("backend", "process")
        execution.setdefault("start_method", "spawn")
        execution.setdefault("native_threads", 1)
        if execution["backend"] != "process":
            raise ValueError("Only execution.backend='process' is implemented.")
        if execution["start_method"] != "spawn":
            raise ValueError("Only execution.start_method='spawn' is implemented.")
        if positive_int(execution["native_threads"], "native_threads") != 1:
            raise ValueError("The first LS implementation supports native_threads=1 only.")

        policy = config.setdefault("budget_policy", {})
        if set(policy) - {"type"} or policy.setdefault("type", "step_round_robin") != "step_round_robin":
            raise ValueError("Only budget_policy.type='step_round_robin' is implemented.")

        finisher = ExactCPUFinisher(
            config.get("finisher", {"type": "exact_cpu", "parameters": {}})
        )
        config["finisher"] = finisher.config
        if parameters["finish_at"] > finisher.max_vertices:
            raise ValueError("finish_at exceeds the finisher vertex limit.")

        limits = config.setdefault("limits", {})
        if set(limits) - {"shots_total", "max_local_tasks_total"}:
            raise ValueError("Unknown LS limit.")
        limits.setdefault("shots_total", None)
        limits.setdefault("max_local_tasks_total", None)
        if limits["shots_total"] is not None:
            positive_int(limits["shots_total"], "shots_total", minimum=0)
        if limits["max_local_tasks_total"] is not None:
            positive_int(limits["max_local_tasks_total"], "max_local_tasks_total")

        collect = config.setdefault("collect", {})
        allowed_collect = {
            "reduction_history",
            "parameter_trajectory",
            "local_task_history",
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

        if source_config["type"] == "qaoa_statevector_cpu":
            local_limit = source_config["parameters"]["max_vertices"]
            if builder.max_size > local_limit and len(graph) > parameters["finish_at"]:
                raise ValueError(
                    "The state-vector source limit is smaller than the builder max_size."
                )
        config["num_treads"] = int(config.get("num_treads", 1))

    def _plan_tasks(self, state, config):
        graph = state.current_graph()
        builder = build_subgraph_builder(
            config["subgraph_builder"],
            seed=_stable_seed(config.get("seed"), "builder", state.version),
        )
        tasks = builder.build(graph, version=state.version)
        for task in tasks:
            task["repair"] = False
            task["local_graph"] = graph.subgraph(task["nodes"])
        allowed = _allowed_pairs(state, config["selection"]["parameters"]["pairs"])
        observed = set()
        for task in tasks:
            observed.update(_task_pairs(task["nodes"], allowed))
        repair_added = False
        if allowed and not observed:
            anchor = allowed[0]
            nodes = greedy_anchor_block(graph, anchor, builder.max_size)
            tasks.append(
                {
                    "round": 0,
                    "block": len(tasks),
                    "nodes": tuple(nodes),
                    "task_id": f"r{state.version}:repair:anchor",
                    "boundary": "induced_no_fields",
                    "repair": True,
                    "repair_reason": "empty_coverage_anchor_pair",
                    "repair_pair": list(anchor),
                    "local_graph": graph.subgraph(nodes),
                }
            )
            repair_added = True
            observed.update(_task_pairs(nodes, allowed))
        missing = [pair for pair in allowed if pair not in observed]
        coverage = {
            "task_version": int(state.version),
            "allowed_pairs": len(allowed),
            "observed_pairs": len(observed),
            "missing_pairs": len(missing),
            "fraction": (len(observed) / len(allowed)) if allowed else None,
            "definition": "observed_allowed_pairs_inside_planned_tasks",
            "require_full": bool(config["coverage"]["require_full"]),
            "empty_policy": config["coverage"]["empty_policy"],
            "repair_added": repair_added,
            "task_ids": [task["task_id"] for task in tasks],
            "missing_examples": [list(pair) for pair in missing[:8]],
        }
        if config["coverage"]["require_full"] and missing:
            raise ValueError(
                f"The planned tasks do not cover {len(missing)} required pair(s); "
                "reduce max_size or use a builder with larger blocks."
            )
        return tasks, allowed, observed, coverage

    def _make_payload(self, task, state, config, steps, quota):
        source_config = deepcopy(config["correlation_source"])
        if source_config["type"] == "qaoa_statevector_cpu":
            source_config["optimizer"]["parameters"]["max_steps"] = int(steps)
        return {
            "task_id": task["task_id"],
            "nodes": [int(node) for node in task["nodes"]],
            "edges": [
                [int(u), int(v), float(weight)] for u, v, weight in task["local_graph"].edges
            ],
            "parent_version": int(state.version),
            "source_config": source_config,
            "seed": _stable_seed(config.get("seed"), "source", task["task_id"]),
            "pair_domain": config["selection"]["parameters"]["pairs"],
            "shot_quota": int(quota),
            "training_steps": int(steps),
            "collect_optimizer_history": bool(config["collect"]["local_optimizer_history"]),
            "collect_objective_history": bool(config["collect"]["local_optimizer_history"]),
        }

    def _run(self, graph, config, rng):
        from ..tools.reporting import log_message

        reducer = GraphReducer()
        state = reducer.start(graph)
        budget = ShotBudget(config["limits"]["shots_total"])
        aggregator = MeanCorrelationAggregator(config["aggregation"])
        selector = ObservedMaxAbsCorrelationSelector(config["selection"])
        finisher = ExactCPUFinisher(config["finisher"])
        finish_at = config["parameters"]["finish_at"]
        max_reductions = config["parameters"]["max_reductions"]
        collect = config["collect"]
        requested_threads = int(config["num_treads"])
        applied_threads = 1
        max_applied_threads = 1

        reductions = 0
        local_tasks_total = 0
        local_tasks_completed = 0
        local_tasks_skipped = 0
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
        local_task_history = []
        local_optimizer_history = []
        correlation_tables = []
        coverage_history = []
        budget_history = []
        status = "completed"
        termination_reason = "completed"
        solution_source = None
        residual_solution = None
        reduced_cut_value = None
        error_text = None
        executor = None
        last_mandatory = []
        last_quotas = []
        unknown_worker_spend = False

        def fallback(reason, state_status="stopped"):
            nonlocal residual_solution, reduced_cut_value, solution_source
            nonlocal termination_reason, status
            residual_solution = {int(node): 1 for node in state.current_nodes}
            reduced_cut_value = state.current_graph().cut_value(residual_solution)
            solution_source = "all_plus_reduced_fallback"
            termination_reason = reason
            status = state_status

        try:
            if len(graph) == 0:
                residual_solution = {}
                reduced_cut_value = 0.0
                solution_source = "empty_graph"
                termination_reason = "empty_graph"
            elif state.ising.edge_count == 0:
                residual_solution = {int(node): 1 for node in state.current_nodes}
                reduced_cut_value = 0.0
                solution_source = "zero_interaction_initial"
                termination_reason = "no_interactions"
            else:
                while True:
                    if len(state.current_nodes) <= finish_at:
                        if state.ising.edge_count == 0:
                            residual_solution = {int(node): 1 for node in state.current_nodes}
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
                        residual_solution = {int(node): 1 for node in state.current_nodes}
                        reduced_cut_value = 0.0
                        solution_source = "zero_interaction_residual"
                        termination_reason = "no_interactions"
                        break
                    if max_reductions is not None and reductions >= max_reductions:
                        fallback("max_reductions")
                        break

                    tasks, allowed, observed, coverage = self._plan_tasks(state, config)
                    coverage_history.append(coverage)
                    if not allowed:
                        fallback("empty_allowed_pair_domain")
                        break
                    viable = [
                        task for task in tasks
                        if len(_task_pairs(task["nodes"], allowed)) > 0
                    ]
                    skipped = [task for task in tasks if task not in viable]
                    local_tasks_skipped += len(skipped)
                    for task in skipped:
                        if collect["local_task_history"]:
                            local_task_history.append(
                                {
                                    "reduction_step": reductions + 1,
                                    "parent_version": int(state.version),
                                    "task_version": int(state.version),
                                    "task_id": task["task_id"],
                                    "round": task["round"],
                                    "block": task["block"],
                                    "repair": bool(task["repair"]),
                                    "nodes": list(task["nodes"]),
                                    "requested_pairs": 0,
                                    "pair_domain": config["selection"]["parameters"]["pairs"],
                                    "source_type": config["correlation_source"]["type"],
                                    "status": "skipped_no_requested_pairs",
                                    "shot_quota": 0,
                                    "training_steps": 0,
                                    "resources": {},
                                }
                            )
                    if not viable:
                        fallback("empty_observed_pair_domain")
                        break
                    max_tasks = config["limits"]["max_local_tasks_total"]
                    if max_tasks is not None and len(viable) > max_tasks:
                        raise ValueError(
                            f"The step plans {len(viable)} local source calls, above "
                            f"max_local_tasks_total={max_tasks}."
                        )
                    mandatory = []
                    costs = []
                    limits = []
                    for task in viable:
                        m, cost, iterations = _task_requirements(config["correlation_source"])
                        mandatory.append(m)
                        costs.append(cost)
                        limits.append(iterations)
                    allocation = _round_robin_quotas(
                        mandatory,
                        costs,
                        limits,
                        budget.total,
                        budget.used,
                    )
                    if allocation is None:
                        fallback("shot_budget")
                        break
                    steps, quotas = allocation
                    last_mandatory = list(mandatory)
                    last_quotas = list(quotas)
                    active_count = len(viable)
                    applied_threads = max(
                        1, min(requested_threads, os.cpu_count() or 1, active_count)
                    )
                    max_applied_threads = max(max_applied_threads, applied_threads)
                    payloads = [
                        self._make_payload(task, state, config, steps[index], quotas[index])
                        for index, task in enumerate(viable)
                    ]
                    started_step = perf_counter()
                    try:
                        if applied_threads == 1:
                            results = [solve_local_task(payload) for payload in payloads]
                        else:
                            if executor is None:
                                import multiprocessing as mp
                                from concurrent.futures import ProcessPoolExecutor

                                executor = ProcessPoolExecutor(
                                    max_workers=applied_threads,
                                    mp_context=mp.get_context(config["execution"]["start_method"]),
                                )
                            results = _execute_bounded(executor, payloads, applied_threads)
                    except Exception as exc:
                        error_text = f"{type(exc).__name__}: {exc}"
                        unknown_worker_spend = True
                        fallback("worker_error", state_status="stopped")
                        break
                    step_seconds = perf_counter() - started_step
                    step_actual_shots = 0
                    for index, result in enumerate(results):
                        if result.get("task_id") != viable[index]["task_id"]:
                            raise RuntimeError("A worker returned a result for another task.")
                        if int(result.get("parent_version", -1)) != int(state.version):
                            raise RuntimeError("A worker returned a stale task version.")
                        if list(result.get("nodes", [])) != list(viable[index]["nodes"]):
                            raise RuntimeError("A worker returned a different local node set.")
                        requested = set(_task_pairs(viable[index]["nodes"], allowed))
                        reported = {
                            _canonical_pair(pair[0], pair[1])
                            for pair in result.get("values", {})
                        }
                        if not requested <= reported:
                            raise RuntimeError(
                                "A worker did not report every requested local pair."
                            )
                        used = sum(
                            int(result.get("resources", {}).get(key, 0))
                            for key in ("training_shots", "correlation_shots", "readout_shots")
                        )
                        if used > quotas[index]:
                            raise RuntimeError("A local task spent more shots than its quota.")
                        budget.spend(used)
                        step_actual_shots += used
                        local_tasks_total += 1
                        local_tasks_completed += 1
                        source_calls += 1
                        training_shots += int(result["resources"].get("training_shots", 0))
                        correlation_shots += int(result["resources"].get("correlation_shots", 0))
                        objective_evaluations += int(result["resources"].get("objective_evaluations", 0))
                        optimizer_steps += int(result["resources"].get("optimizer_steps", 0))
                        gamma_points += int(result["resources"].get("gamma_points", 0))
                        coefficient_passes += int(result["resources"].get("coefficient_passes", 0))
                        pair_correlators += int(result["resources"].get("pair_correlators", 0))
                        classical_seconds += float(result["resources"].get("classical_seconds", 0.0))
                        angles = list(result.get("angles", []))
                        if collect["parameter_trajectory"]:
                            parameter_trajectory.append(
                                {
                                    "reduction_step": reductions + 1,
                                    "task_id": result["task_id"],
                                    "round": viable[index]["round"],
                                    "block": viable[index]["block"],
                                    "repair": bool(viable[index]["repair"]),
                                    "nodes": list(result["nodes"]),
                                    "layers": int(result.get("layers", 0)),
                                    "angle_names": [
                                        f"gamma_{i}" for i in range(int(result.get("layers", 0)))
                                    ]
                                    + [
                                        f"beta_{i}" for i in range(int(result.get("layers", 0)))
                                    ],
                                    "angles": angles,
                                    "initial_angles": result.get("initial_angles"),
                                    "source_type": config["correlation_source"]["type"],
                                    "objective_reference": result.get("objective_reference"),
                                    "objective_value": result.get("training", {}).get("objective_value"),
                                    "initialization": result.get("training", {}).get("initialization"),
                                    "shot_quota": quotas[index],
                                    "training_steps": steps[index],
                                }
                            )
                        if collect["local_task_history"]:
                            local_task_history.append(
                                {
                                    "reduction_step": reductions + 1,
                                    "parent_version": int(state.version),
                                    "task_version": int(state.version),
                                    "task_id": result["task_id"],
                                    "round": viable[index]["round"],
                                    "block": viable[index]["block"],
                                    "repair": bool(viable[index]["repair"]),
                                    "nodes": list(result["nodes"]),
                                    "requested_pairs": len(_task_pairs(result["nodes"], allowed)),
                                    "pair_domain": config["selection"]["parameters"]["pairs"],
                                    "source_type": config["correlation_source"]["type"],
                                    "parameters": deepcopy(config["correlation_source"].get("parameters", {})),
                                    "angles": angles,
                                    "initial_angles": result.get("initial_angles"),
                                    "local_energy": result.get("training", {}).get("objective_value"),
                                    "status": result.get("status", "completed"),
                                    "seconds": result.get("seconds"),
                                    "started_unix": result.get("started_unix"),
                                    "finished_unix": result.get("finished_unix"),
                                    "shot_quota": quotas[index],
                                    "training_steps": steps[index],
                                    "resources": deepcopy(result.get("resources", {})),
                                    "native_pools": deepcopy(result.get("native_pools")),
                                }
                            )
                        if collect["local_optimizer_history"]:
                            local_optimizer_history.append(
                                {
                                    "reduction_step": reductions + 1,
                                    "task_id": result["task_id"],
                                    "angles": angles,
                                    "optimizer_history": result.get("training", {}).get("optimizer_history"),
                                    "objective_history": result.get("training", {}).get("objective_history"),
                                    "profile_type": result.get("training", {}).get("profile_type"),
                                    "gamma_profile": result.get("training", {}).get("gamma_profile"),
                                }
                            )
                        if collect["correlation_tables"]:
                            correlation_tables.append(
                                {
                                    "reduction_step": reductions + 1,
                                    "task_id": result["task_id"],
                                    "nodes": list(result["nodes"]),
                                    "mode": result.get("mode"),
                                    "pairs": [
                                        {
                                            "u": int(pair[0]),
                                            "v": int(pair[1]),
                                            "physical_correlation": result["values"][tuple(pair)],
                                        }
                                        for pair in result["pairs"]
                                    ],
                                }
                            )
                    aggregated = aggregator.aggregate(
                        state, results, pairs=config["selection"]["parameters"]["pairs"]
                    )
                    if not aggregated["observed_pairs"]:
                        fallback("empty_observed_pair_domain")
                        break
                    coverage_history[-1]["observed_pairs_actual"] = len(aggregated["observed_pairs"])
                    coverage_history[-1]["missing_pairs_actual"] = len(
                        set(aggregated["allowed_pairs"]) - set(aggregated["observed_pairs"])
                    )
                    budget_history.append(
                        {
                            "reduction_step": reductions + 1,
                            "task_ids": [task["task_id"] for task in viable],
                            "mandatory_shots": list(mandatory),
                            "training_steps": list(steps),
                            "shot_quotas": list(quotas),
                            "actual_shots": int(step_actual_shots),
                            "seconds": step_seconds,
                        }
                    )
                    table = aggregator.make_table(state, aggregated)
                    selection = selector.select(state, table)
                    event = reducer.reduce(
                        state,
                        selection.removed,
                        selection.retained,
                        selection.sign,
                        correlation=selection.correlation,
                        selection_score=selection.score,
                    )
                    reductions += 1
                    event = deepcopy(event)
                    event["correlation_origin"] = "local_subgraph_mixture"
                    event["physical_correlation"] = selection.correlation
                    event["expected_fixed_energy_local"] = event.get("expected_fixed_energy")
                    event["source"] = {
                        "type": config["correlation_source"]["type"],
                        "builder_type": config["subgraph_builder"]["type"],
                        "aggregation": config["aggregation"]["type"],
                        "selection": config["selection"]["type"],
                        "task_ids": [result["task_id"] for result in results],
                    }
                    event["aggregation_observations"] = int(aggregated["counts"][
                        _canonical_pair(selection.removed, selection.retained)
                    ])
                    event["local_task_ids"] = list(
                        aggregated["task_ids"][_canonical_pair(selection.removed, selection.retained)]
                    )
                    event["coverage"] = coverage
                    event["step_seconds"] = step_seconds
                    event["local_tasks"] = len(results)
                    event["global_expected_energy"] = None
                    event["global_expected_energy_reason"] = "local_subgraph_mixture_not_global"
                    if collect["reduction_history"]:
                        reduction_history.append(deepcopy(event))
                    log_message(
                        config,
                        f"LS reduction {reductions}: {selection.removed}->{selection.retained} "
                        f"sign={selection.sign}; active={len(state.current_nodes)}.",
                        detail=False,
                    )
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

        if residual_solution is None:
            fallback("no_residual_solution")
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
            raise ArithmeticError("LS-RQAOA energy and cut offset disagree.")
        resources = {
            "shots_used": budget.used,
            "training_shots": training_shots,
            "correlation_shots": correlation_shots,
            "readout_shots": 0,
            "objective_evaluations": objective_evaluations,
            "optimizer_steps": optimizer_steps,
            "source_calls": source_calls,
            "local_tasks_total": local_tasks_total,
            "local_tasks_completed": local_tasks_completed,
            "local_tasks_skipped": local_tasks_skipped,
            "reductions_completed": reductions,
            "num_treads_requested": requested_threads,
            "num_treads_applied": max_applied_threads,
            "num_treads": max_applied_threads,
            "native_threads": 1,
            "backend": config["execution"]["backend"],
            "start_method": config["execution"]["start_method"],
            "gamma_points": gamma_points,
            "coefficient_passes": coefficient_passes,
            "pair_correlators": pair_correlators,
            "classical_seconds": classical_seconds,
            "budget": budget.snapshot(),
            "mandatory_shots_last_step": list(last_mandatory),
            "unknown_worker_spend": bool(unknown_worker_spend),
            "shots_upper_bound": budget.used + (sum(last_quotas) if unknown_worker_spend else 0),
        }
        data_status = {key: ("collected" if collect[key] else "not_collected") for key in collect}
        report = {
            "algorithm_type": self.algorithm_type,
            "reductions_completed": reductions,
            "solution_source": solution_source,
            "remaining_nodes": len(state.current_nodes),
            "cut_offset": cut_offset,
            "ising_constant": ising_constant,
            "source_type": config["correlation_source"]["type"],
            "builder_type": config["subgraph_builder"]["type"],
            "num_treads_requested": requested_threads,
            "num_treads_applied": max_applied_threads,
            "local_tasks_total": local_tasks_total,
            "local_tasks_completed": local_tasks_completed,
            "coverage_fraction": coverage_history[-1]["fraction"] if coverage_history else None,
        }
        if error_text is not None:
            report["worker_error"] = error_text
        log_message(
            config,
            f"LS-RQAOA stops: {termination_reason}; reductions={reductions}; "
            f"local_tasks={local_tasks_total}; shots={budget.used}.",
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
            "num_treads": max_applied_threads,
            "reduction_history": reduction_history if collect["reduction_history"] else None,
            "parameter_trajectory": parameter_trajectory if collect["parameter_trajectory"] else None,
            "local_task_history": local_task_history if collect["local_task_history"] else None,
            "local_optimizer_history": local_optimizer_history if collect["local_optimizer_history"] else None,
            "correlation_tables": correlation_tables if collect["correlation_tables"] else None,
            "coverage_history": coverage_history,
            "budget_history": budget_history,
            "final_reduction_state": state.snapshot() if collect["final_reduction_state"] else None,
            "resources": resources,
            "data_status": data_status,
            "report": report,
        }


def main(argv=None):
    """Run a small LS-RQAOA example through the common demo interface."""
    from ..tools.demo import run_solver_demo

    return run_solver_demo(LSRQAOACPU, argv)


if __name__ == "__main__":
    raise SystemExit(main())
