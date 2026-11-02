"""Incremental LS-RQAOA with an explicit epoch correlation bank.

The solver reuses the common local Hamiltonian, reducer, finisher, source
workers, budget, and record machinery.  Its scientific additions are:

* versioned subgraph refreshers (full refresh and retired-center cover);
* an epoch mean over all completed local tasks;
* an explicit freshness policy with bounded stale reuse;
* four selection score rules.

The bank is a collection of local physical observations, not a global
joint distribution.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
import os
from time import perf_counter

import numpy as np

from ..core.budget import ShotBudget
from ..core.reduction import GraphReducer
from ..generators.incremental import (
    build_incremental_initial_builder,
    survivor_neighbor_cover,
)
from ..generators.subgraphs import greedy_anchor_block
from .base import BaseSolver, positive_int
from .components import ExactCPUFinisher, SelectionResult, validate_correlation_source_config
from .ls_rqaoa_cpu import _execute_bounded, _round_robin_quotas, _stable_seed, _task_requirements
from .ls_worker import solve_local_task

__all__ = [
    "IncrementalLSRQAOACPU",
    "IncrementalSelection",
    "normalize_incremental_source",
]


def _canonical_pair(u, v):
    return (min(int(u), int(v)), max(int(u), int(v)))


def _allowed_pairs(state, pairs):
    if pairs == "edges":
        return tuple(
            sorted(
                _canonical_pair(u, v)
                for u, v, weight in state.current_graph().edges
                if float(weight) != 0.0
            )
        )
    nodes = sorted(int(node) for node in state.current_nodes)
    return tuple(
        (nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes))
    )


def _task_pairs(nodes, allowed):
    chosen = set(int(node) for node in nodes)
    return tuple(pair for pair in allowed if pair[0] in chosen and pair[1] in chosen)


def _finite_nonnegative_int(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _fixed_angles(values, layers):
    if values is None:
        values = [0.5] * int(layers) + [0.3] * int(layers)
    values = [float(value) for value in values]
    if len(values) != 2 * int(layers):
        raise ValueError("initial angles must contain 2*layers values.")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("initial angles must be finite.")
    return values


def normalize_incremental_source(config):
    """Accept the compact stage-17 source recipe and the common source form."""
    config = deepcopy(config)
    if not isinstance(config, dict):
        raise ValueError("correlation_source must be an object.")
    kind = config.get("type")
    if kind == "qaoa_statevector_cpu" and (
        "p" in config or "training" in config or "parameters" not in config
    ):
        layers = int(config.get("p", 1))
        training = deepcopy(config.get("training", {}))
        if set(training) - {
            "criterion", "mode", "shots", "optimizer", "max_steps",
            "gradient_tolerance", "learning_rate", "initial_angles",
        }:
            raise ValueError("Unknown compact training field.")
        mode = training.get("mode", "exact")
        shots = training.get("shots", 256)
        criterion = training.get("criterion", "mean_cut")
        optimizer_type = training.get("optimizer", "adam")
        if optimizer_type != "adam":
            raise ValueError("Only the shared ADAM source optimizer is supported.")
        values = training.get("initial_angles", config.get("initial_angles"))
        values = _fixed_angles(values, layers)
        correlations = deepcopy(config.get("correlations", {"mode": "exact"}))
        if set(correlations) - {"mode", "shots"}:
            raise ValueError("Unknown compact correlations field.")
        result = {
            "type": "qaoa_statevector_cpu",
            "parameters": {
                "layers": layers,
                "max_vertices": int(config.get("max_vertices", 20)),
                "dtype": config.get("dtype", "complex128"),
            },
            "objective": {"type": criterion, "parameters": {}},
            "objective_reference": "working_graph",
            "estimation": {"mode": mode, "shots_per_evaluation": int(shots)},
            "optimizer": {
                "type": "adam",
                "parameters": {
                    "max_steps": int(training.get("max_steps", 1)),
                    "learning_rate": float(training.get("learning_rate", 0.05)),
                    "gradient_tolerance": float(training.get("gradient_tolerance", 0.0)),
                },
            },
            "initialization": {
                "type": "fixed_angles",
                "parameters": {"values": values},
            },
            "correlations": correlations,
        }
        return validate_correlation_source_config(result)
    if kind == "qaoa_analytic_p1_cpu" and "p" in config and "parameters" not in config:
        result = {
            "type": "qaoa_analytic_p1_cpu",
            "parameters": {
                "layers": 1,
                "dtype": "float64",
                "pair_batch_size": int(config.get("pair_batch_size", 2048)),
            },
            "objective": {"type": "mean_cut", "parameters": {}},
            "objective_reference": "working_graph",
            "estimation": {"mode": "exact"},
            "optimizer": {
                "type": "gamma_grid_exact_beta",
                "parameters": {
                    "gamma_bounds": config.get(
                        "optimizer", {}
                    ).get("parameters", {}).get(
                        "gamma_bounds", config.get("gamma_bounds", [-math.pi, math.pi])
                    ),
                    "points": int(
                        config.get("optimizer", {})
                        .get("parameters", {})
                        .get("points", config.get("points", 17))
                    ),
                },
            },
            "correlations": {"mode": "exact"},
        }
        return validate_correlation_source_config(result)
    return validate_correlation_source_config(config)


@dataclass(frozen=True)
class IncrementalSelection:
    """One selection with its score and bank provenance kept separate."""

    removed: int
    retained: int
    sign: int
    score: float
    correlation: float
    pair: tuple
    score_rule: str
    score_direction: str
    weight: float
    weight_power: int
    observed_epoch: int
    age: int
    stale: bool
    task_ids: tuple
    contexts: tuple


class IncrementalLSRQAOACPU(BaseSolver):
    """Run a bounded incremental LS-RQAOA epoch process on CPU."""

    algorithm_type = "laboratory.incremental_ls_rqaoa_cpu"
    supports_parallel = True
    supports_checkpoint = False
    extra_config_fields = (
        "finish_at",
        "pairs",
        "zero_sign",
        "require_full_coverage",
        "local_problem",
        "initial_builder",
        "update",
        "bank",
        "selection",
        "correlation_source",
        "finisher",
        "execution",
        "limits",
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
        "epoch_history",
        "bank_history",
        "selection_history",
        "coverage_history",
        "budget_history",
        "local_task_history",
        "final_bank",
    )

    def __init__(self, config=None):
        config = deepcopy(config or {})
        output = config.setdefault("output", {})
        for key in ("report", "logs"):
            if key in config:
                value = config.pop(key)
                if key not in output:
                    output[key] = value
        super().__init__(config)

    def _validate(self, config, graph):
        finish_at = config.setdefault("finish_at", min(6, max(1, len(graph))))
        positive_int(finish_at, "finish_at")
        pairs = config.setdefault("pairs", "edges")
        if pairs not in ("edges", "all_pairs"):
            raise ValueError("pairs must be edges or all_pairs.")
        zero_sign = config.setdefault("zero_sign", 1)
        if zero_sign not in (-1, 1):
            raise ValueError("zero_sign must be +1 or -1.")
        require_full = config.setdefault("require_full_coverage", False)
        if type(require_full) is not bool:
            raise TypeError("require_full_coverage must be boolean.")
        local_problem = config.setdefault("local_problem", {"boundary": "induced_no_fields"})
        if set(local_problem) - {"boundary"} or local_problem.get("boundary") != "induced_no_fields":
            raise ValueError("Only local_problem.boundary='induced_no_fields' is supported.")

        limits = config.setdefault("limits", {})
        if set(limits) - {
            "max_reductions", "max_total_shots", "max_tasks_per_epoch",
            "max_total_tasks", "shots_total", "max_local_tasks_total",
        }:
            raise ValueError("Unknown incremental limits.")
        max_reductions = limits.setdefault("max_reductions", 2)
        if max_reductions is not None:
            _finite_nonnegative_int(max_reductions, "max_reductions", 0)
        total_shots = limits.get("max_total_shots", limits.get("shots_total"))
        if total_shots is not None:
            _finite_nonnegative_int(total_shots, "max_total_shots", 0)
        limits["max_total_shots"] = total_shots
        limits["shots_total"] = total_shots
        max_tasks_epoch = limits.get("max_tasks_per_epoch", 4)
        max_total_tasks = limits.get("max_total_tasks", None)
        _finite_nonnegative_int(max_tasks_epoch, "max_tasks_per_epoch", 1)
        if max_total_tasks is not None:
            _finite_nonnegative_int(max_total_tasks, "max_total_tasks", 1)
        limits["max_tasks_per_epoch"] = max_tasks_epoch
        limits["max_total_tasks"] = max_total_tasks

        initial = config.setdefault(
            "initial_builder",
            {
                "type": "greedy_abs_weight_partition",
                "parameters": {"max_size": 6, "rounds": 1, "full_graph_when_fits": True},
            },
        )
        # The new builder accepts the compact shape; legacy builders accept
        # their established parameters.
        builder_config = deepcopy(initial)
        parameters = builder_config.setdefault("parameters", {})
        for key in (
            "max_size", "rounds", "full_graph_when_fits", "n_restarts",
            "local_iter", "candidate_pool", "boundary",
        ):
            if key in builder_config:
                parameters.setdefault(key, builder_config.pop(key))
        if builder_config.get("type") == "single_densest_edge_count_v1":
            parameters.setdefault("rounds", 1)
        else:
            parameters.setdefault("max_size", 6)
            parameters.setdefault("rounds", 1)
            parameters.setdefault("full_graph_when_fits", True)
        builder = build_incremental_initial_builder(builder_config, seed=config.get("seed"))
        config["initial_builder"] = builder.config

        update = config.setdefault("update", {"type": "survivor_neighbor_cover_v1"})
        if set(update) - {"type", "growth_metric"}:
            raise ValueError("Unknown update field.")
        update.setdefault("type", "refresh_all")
        if update["type"] not in ("refresh_all", "survivor_neighbor_cover_v1"):
            raise ValueError("Unknown update policy.")
        update.setdefault("growth_metric", "count")
        if update["growth_metric"] not in ("count", "abs_weight"):
            raise ValueError("Unknown growth metric.")

        bank = config.setdefault("bank", {})
        if set(bank) - {"aggregation", "policy", "max_age_steps"}:
            raise ValueError("Unknown bank field.")
        bank.setdefault("aggregation", "epoch_mean")
        if bank["aggregation"] != "epoch_mean":
            raise ValueError("Only epoch_mean aggregation is implemented.")
        bank.setdefault("policy", "keep_unobserved")
        if bank["policy"] not in ("current_epoch_only", "keep_unobserved"):
            raise ValueError("Unknown bank policy.")
        if bank["policy"] == "keep_unobserved":
            if "max_age_steps" not in bank:
                raise ValueError("keep_unobserved requires finite max_age_steps.")
            _finite_nonnegative_int(bank["max_age_steps"], "max_age_steps", 0)
        elif "max_age_steps" in bank:
            raise ValueError("current_epoch_only does not take max_age_steps.")

        selection = config.setdefault("selection", {"type": "max_abs_correlation"})
        if set(selection) - {"type", "parameters", "weight_power"}:
            raise ValueError("Unknown selection field.")
        selection.setdefault("type", "max_abs_correlation")
        if selection["type"] not in (
            "max_abs_correlation",
            "max_abs_weighted_correlation",
            "min_signed_edge_product",
        ):
            raise ValueError("Unknown selection rule.")
        selection.setdefault("parameters", {})
        if set(selection["parameters"]) - {"pairs", "zero_sign", "weight_power"}:
            raise ValueError("Unknown selection parameter.")
        selection["parameters"].setdefault("pairs", pairs)
        if selection["parameters"]["pairs"] != pairs:
            raise ValueError("selection and top-level pairs disagree.")
        selection["parameters"].setdefault("zero_sign", zero_sign)
        selection["parameters"]["weight_power"] = int(
            selection.get("weight_power", selection["parameters"].get("weight_power", 1))
        )
        if selection["type"] == "max_abs_weighted_correlation":
            q = selection["parameters"]["weight_power"]
            if q not in (1, 2):
                raise ValueError("weight_power must be 1 or 2.")
            if pairs != "edges":
                raise ValueError("weighted selection needs pairs='edges'.")

        source = normalize_incremental_source(config.get("correlation_source", {}))
        if source["type"] not in ("qaoa_statevector_cpu", "qaoa_analytic_p1_cpu"):
            raise ValueError("Only CPU statevector and analytic p=1 sources are supported.")
        if source.get("objective_reference") != "working_graph":
            raise ValueError("Incremental induced tasks need objective_reference='working_graph'.")
        config["correlation_source"] = source
        max_vertices = source.get("parameters", {}).get("max_vertices")
        if max_vertices is not None and builder.max_size > max_vertices:
            raise ValueError("initial_builder.max_size exceeds the source max_vertices.")

        execution = config.setdefault("execution", {})
        if set(execution) - {"backend", "start_method", "native_threads"}:
            raise ValueError("Unknown execution field.")
        execution.setdefault("backend", "process")
        execution.setdefault("start_method", "spawn")
        execution.setdefault("native_threads", 1)
        if execution["backend"] != "process":
            raise ValueError("Only the CPU process backend is implemented.")
        if execution["start_method"] != "spawn":
            raise ValueError("Only spawn is implemented.")
        if positive_int(execution["native_threads"], "native_threads") != 1:
            raise ValueError("native_threads=1 only.")

        finisher_config = deepcopy(
            config.get("finisher", {"type": "exact_cpu", "parameters": {}})
        )
        if "max_nodes" in finisher_config:
            finisher_config.setdefault("parameters", {})["max_vertices"] = int(
                finisher_config.pop("max_nodes")
            )
        finisher = ExactCPUFinisher(finisher_config)
        config["finisher"] = finisher.config
        if finish_at > finisher.max_vertices:
            raise ValueError("finish_at exceeds the finisher limit.")

        collect = config.setdefault("collect", {})
        if set(collect) - {"summary", "epoch_bank", "local_histories"}:
            raise ValueError("Unknown collect field.")
        collect.setdefault("summary", True)
        collect.setdefault("epoch_bank", False)
        collect.setdefault("local_histories", False)
        for key, value in collect.items():
            if type(value) is not bool:
                raise TypeError(f"collect.{key} must be boolean.")

        config["num_treads"] = int(config.get("num_treads", 1))
        if int(config["num_treads"]) < 1:
            raise ValueError("num_treads must be positive.")

    def _source_requirements(self, config):
        return _task_requirements(config["correlation_source"])

    def _plan(self, state, config, retained_center):
        graph = state.current_graph()
        builder_config = deepcopy(config["initial_builder"])
        builder = build_incremental_initial_builder(
            builder_config,
            seed=_stable_seed(config.get("seed"), "builder", state.version, "initial"),
        )
        if config["update"]["type"] == "survivor_neighbor_cover_v1" and state.version > 0:
            if retained_center is None or retained_center not in state.current_nodes:
                raise ValueError("Survivor cover needs an active retained center.")
            tasks = survivor_neighbor_cover(
                graph,
                retained_center,
                max_size=builder.max_size,
                growth_metric=config["update"]["growth_metric"],
                rounds=int(builder.config["parameters"].get("rounds", 1)),
                version=state.version,
                full_graph_when_fits=bool(
                    builder.config["parameters"].get("full_graph_when_fits", True)
                ),
            )
        else:
            tasks = builder.build(graph, version=state.version)
        for task in tasks:
            task["repair"] = False
            task["local_graph"] = graph.subgraph(task["nodes"])
        allowed = _allowed_pairs(state, config["pairs"])
        observed = set()
        for task in tasks:
            observed.update(_task_pairs(task["nodes"], allowed))
        if allowed and not observed:
            anchor = allowed[0]
            nodes = greedy_anchor_block(graph, anchor, builder.max_size)
            tasks.append(
                {
                    "round": 0,
                    "block": len(tasks),
                    "nodes": tuple(sorted(int(node) for node in nodes)),
                    "task_id": f"r{state.version}:repair:anchor",
                    "boundary": "induced_no_fields",
                    "builder_type": "anchor_pair",
                    "repair": True,
                    "repair_reason": "empty_coverage_anchor_pair",
                    "repair_pair": list(anchor),
                    "local_graph": graph.subgraph(nodes),
                }
            )
            observed.update(_task_pairs(nodes, allowed))
        missing = [pair for pair in allowed if pair not in observed]
        if config["require_full_coverage"] and missing:
            raise ValueError(
                f"The current plan leaves {len(missing)} allowed pair(s) uncovered."
            )
        return tasks, allowed, observed, missing

    def _make_payload(self, task, state, config, steps, quota):
        source_config = deepcopy(config["correlation_source"])
        if source_config["type"] == "qaoa_statevector_cpu":
            source_config["optimizer"]["parameters"]["max_steps"] = int(steps)
        return {
            "task_id": task["task_id"],
            "nodes": [int(node) for node in task["nodes"]],
            "edges": [
                [int(u), int(v), float(weight)]
                for u, v, weight in task["local_graph"].edges
            ],
            "parent_version": int(state.version),
            "source_config": source_config,
            "seed": _stable_seed(config.get("seed"), "source", state.version, task["task_id"]),
            "pair_domain": config["pairs"],
            "shot_quota": int(quota),
            "training_steps": int(steps),
            "collect_optimizer_history": bool(config["collect"]["local_histories"]),
            "collect_objective_history": bool(config["collect"]["local_histories"]),
        }

    def _select(self, state, prepared, config, epoch):
        pairs = config["pairs"]
        allowed = set(_allowed_pairs(state, pairs))
        candidates = []
        for pair, entry in prepared.items():
            if pair not in allowed:
                continue
            c = float(entry["physical_correlation"])
            if not math.isfinite(c) or abs(c) > 1.0 + 1e-9:
                raise ValueError("A bank correlation is invalid.")
            c = min(1.0, max(-1.0, c))
            weight = float(state.current_graph().weight(*pair))
            age = int(epoch - int(entry["observed_epoch"]))
            if age < 0:
                raise ValueError("A bank entry claims a future epoch.")
            rule = config["selection"]["type"]
            q = int(config["selection"]["parameters"]["weight_power"])
            if rule == "max_abs_correlation":
                score = abs(c)
                direction = "max"
            elif rule == "max_abs_weighted_correlation":
                if pairs != "edges":
                    raise ValueError("weighted correlation needs current edges.")
                score = abs(weight) ** q * abs(c)
                direction = "max"
            elif rule == "min_signed_edge_product":
                if pairs != "edges":
                    raise ValueError("min_signed_edge_product needs current edges.")
                score = weight * c
                direction = "min"
            else:
                raise ValueError("Unknown selection rule.")
            if not math.isfinite(score):
                raise ValueError("A selection score is not finite.")
            candidates.append((pair, c, weight, score, direction, entry, age))
        if not candidates:
            raise ValueError("No eligible bank pair exists.")
        if candidates[0][4] == "min":
            chosen = min(candidates, key=lambda item: (item[3], item[0][0], item[0][1]))
        else:
            chosen = min(candidates, key=lambda item: (-item[3], item[0][0], item[0][1]))
        pair, c, weight, score, direction, entry, age = chosen
        sign = 1 if c > 0 else -1 if c < 0 else config["zero_sign"]
        return IncrementalSelection(
            removed=pair[0],
            retained=pair[1],
            sign=sign,
            score=float(score),
            correlation=float(c),
            pair=pair,
            score_rule=config["selection"]["type"],
            score_direction=direction,
            weight=float(weight),
            weight_power=int(config["selection"]["parameters"]["weight_power"]),
            observed_epoch=int(entry["observed_epoch"]),
            age=int(age),
            stale=bool(age > 0),
            task_ids=tuple(entry.get("task_ids", ())),
            contexts=tuple(entry.get("contexts", ())),
        )

    def _fallback(self, reducer, state):
        residual = {int(node): 1 for node in state.current_nodes}
        solution = reducer.restore(state, residual)
        return solution, {int(node): 1 for node in state.current_nodes}

    def _run(self, graph, config, rng):
        from ..tools.reporting import log_message

        reducer = GraphReducer()
        state = reducer.start(graph)
        budget = ShotBudget(config["limits"]["max_total_shots"])
        finisher = ExactCPUFinisher(config["finisher"])
        finish_at = int(config["finish_at"])
        max_reductions = config["limits"]["max_reductions"]
        bank = {}
        bank_history = []
        epoch_history = []
        selection_history = []
        coverage_history = []
        budget_history = []
        local_task_history = []
        reduction_history = []
        retained_center = None
        reductions = 0
        total_tasks = 0
        total_training_shots = 0
        total_correlation_shots = 0
        total_objective_evaluations = 0
        total_optimizer_steps = 0
        status = "completed"
        termination_reason = "completed"
        solution_source = None
        residual_solution = None
        error_text = None
        unknown_spend = False
        last_quotas = []
        applied_threads = 1
        requested_threads = int(config["num_treads"])
        executor = None
        started = perf_counter()

        def finish(reason, solver_status="completed"):
            nonlocal residual_solution, solution_source, termination_reason, status
            if len(state.current_nodes) <= finish_at and state.ising.edge_count > 0:
                local_solution, _ = finisher.solve(state.current_graph())
                residual_solution = {int(k): int(v) for k, v in local_solution.items()}
                solution_source = "exact_finisher"
            else:
                residual_solution = {int(node): 1 for node in state.current_nodes}
                solution_source = "all_plus_reduced_fallback"
            termination_reason = reason
            status = solver_status

        try:
            if len(graph) == 0:
                residual_solution = {}
                solution_source = "empty_graph"
                termination_reason = "empty_graph"
            elif state.ising.edge_count == 0:
                residual_solution = {int(node): 1 for node in state.current_nodes}
                solution_source = "zero_interaction_initial"
                termination_reason = "no_interactions"
            else:
                while True:
                    if len(state.current_nodes) <= finish_at:
                        finish("finish_at")
                        break
                    if state.ising.edge_count == 0:
                        residual_solution = {int(node): 1 for node in state.current_nodes}
                        solution_source = "zero_interaction_residual"
                        termination_reason = "no_interactions"
                        break
                    if max_reductions is not None and reductions >= max_reductions:
                        finish("max_reductions", "stopped")
                        break
                    if len(state.current_nodes) <= 1:
                        finish("single_residual")
                        break

                    epoch = reductions
                    tasks, allowed, planned_pairs, missing = self._plan(state, config, retained_center)
                    planned_pairs_set = set(planned_pairs)
                    if not allowed:
                        finish("empty_allowed_pair_domain", "stopped")
                        break
                    if len(tasks) > config["limits"]["max_tasks_per_epoch"]:
                        raise ValueError("The epoch plan exceeds max_tasks_per_epoch.")
                    if config["limits"]["max_total_tasks"] is not None and (
                        total_tasks + len(tasks) > config["limits"]["max_total_tasks"]
                    ):
                        raise ValueError("The plan exceeds max_total_tasks.")
                    viable = []
                    skipped = []
                    for task in tasks:
                        requested = _task_pairs(task["nodes"], allowed)
                        if requested:
                            viable.append(task)
                        else:
                            skipped.append(task)
                    if not viable:
                        finish("empty_observed_pair_domain", "stopped")
                        break
                    mandatory = []
                    costs = []
                    iteration_limits = []
                    for _ in viable:
                        m, c, i = self._source_requirements(config)
                        mandatory.append(m)
                        costs.append(c)
                        iteration_limits.append(i)
                    allocation = _round_robin_quotas(
                        mandatory,
                        costs,
                        iteration_limits,
                        config["limits"]["max_total_shots"],
                        budget.used,
                    )
                    if allocation is None:
                        finish("budget_exhausted", "stopped")
                        break
                    steps, quotas = allocation
                    last_quotas = list(quotas)
                    applied_threads = max(1, min(requested_threads, os.cpu_count() or 1, len(viable)))
                    payloads = [
                        self._make_payload(task, state, config, steps[index], quotas[index])
                        for index, task in enumerate(viable)
                    ]
                    step_start = perf_counter()
                    if applied_threads == 1:
                        results = [solve_local_task(payload) for payload in payloads]
                    else:
                        if executor is None:
                            import multiprocessing as mp
                            from concurrent.futures import ProcessPoolExecutor

                            executor = ProcessPoolExecutor(
                                max_workers=applied_threads,
                                mp_context=mp.get_context("spawn"),
                            )
                        results = _execute_bounded(executor, payloads, applied_threads)
                    step_seconds = perf_counter() - step_start
                    step_shots = 0
                    observations = []
                    for index, result in enumerate(results):
                        if result.get("task_id") != viable[index]["task_id"]:
                            raise RuntimeError("A worker returned a different task.")
                        if int(result.get("parent_version", -1)) != int(state.version):
                            raise RuntimeError("A worker returned a stale parent version.")
                        if list(result.get("nodes", [])) != list(viable[index]["nodes"]):
                            raise RuntimeError("A worker returned a different node set.")
                        requested = set(_task_pairs(viable[index]["nodes"], allowed))
                        reported = {
                            _canonical_pair(pair[0], pair[1])
                            for pair in result.get("values", {})
                        }
                        if not requested <= reported:
                            raise RuntimeError("A worker omitted a requested local pair.")
                        used = sum(
                            int(result.get("resources", {}).get(key, 0))
                            for key in ("training_shots", "correlation_shots", "readout_shots")
                        )
                        if used > quotas[index]:
                            raise RuntimeError("A worker exceeded its shot quota.")
                        budget.spend(used)
                        step_shots += used
                        total_tasks += 1
                        total_training_shots += int(
                            result.get("resources", {}).get("training_shots", 0)
                        )
                        total_correlation_shots += int(
                            result.get("resources", {}).get("correlation_shots", 0)
                        )
                        total_objective_evaluations += int(
                            result.get("resources", {}).get("objective_evaluations", 0)
                        )
                        total_optimizer_steps += int(
                            result.get("resources", {}).get("optimizer_steps", 0)
                        )
                        values = {
                            _canonical_pair(pair[0], pair[1]): float(value)
                            for pair, value in result.get("values", {}).items()
                            if _canonical_pair(pair[0], pair[1]) in requested
                        }
                        observations.append(
                            {
                                "task_id": result["task_id"],
                                "epoch": int(epoch),
                                "parent_version": int(state.version),
                                "nodes": list(result["nodes"]),
                                "values": values,
                                "angles": list(result.get("angles", [])),
                                "layers": int(result.get("layers", 0)),
                                "training": deepcopy(result.get("training", {})),
                                "resources": deepcopy(result.get("resources", {})),
                                "mode": result.get("mode"),
                                "source_type": config["correlation_source"]["type"],
                            }
                        )
                        if config["collect"]["local_histories"]:
                            local_task_history.append(
                                {
                                    "epoch": int(epoch),
                                    "task_id": result["task_id"],
                                    "nodes": list(result["nodes"]),
                                    "requested_pairs": [list(pair) for pair in sorted(requested)],
                                    "values": [[int(pair[0]), int(pair[1]), float(value)] for pair, value in sorted(values.items())],
                                    "angles": list(result.get("angles", [])),
                                    "resources": deepcopy(result.get("resources", {})),
                                    "seconds": result.get("seconds"),
                                }
                            )
                    for task in skipped:
                        if config["collect"]["local_histories"]:
                            local_task_history.append(
                                {
                                    "epoch": int(epoch),
                                    "task_id": task["task_id"],
                                    "nodes": list(task["nodes"]),
                                    "requested_pairs": [],
                                    "status": "skipped_no_requested_pairs",
                                    "resources": {},
                                }
                            )
                    # One epoch sum/count, after every round task completed.
                    sums = {}
                    counts = {}
                    origins = {}
                    for observation in observations:
                        for pair, value in observation["values"].items():
                            sums[pair] = sums.get(pair, 0.0) + value
                            counts[pair] = counts.get(pair, 0) + 1
                            origins.setdefault(pair, []).append(observation["task_id"])
                    fresh = {
                        pair: {
                            "physical_correlation": sums[pair] / counts[pair],
                            "d": counts[pair],
                            "task_ids": tuple(origins[pair]),
                            "observed_epoch": int(epoch),
                            "contexts": [
                                {
                                    "task_id": observation["task_id"],
                                    "nodes": list(observation["nodes"]),
                                    "parent_version": int(observation["parent_version"]),
                                }
                                for observation in observations
                                if pair in observation["values"]
                            ],
                        }
                        for pair in sorted(sums)
                    }
                    # Build the prepared bank only after the complete barrier.
                    prepared = {}
                    policy = config["bank"]["policy"]
                    for pair in allowed:
                        if pair in fresh:
                            prepared[pair] = dict(fresh[pair])
                        elif policy == "keep_unobserved":
                            old = bank.get(pair)
                            if old is None:
                                continue
                            age = epoch - int(old["observed_epoch"])
                            if age < 0 or age > config["bank"]["max_age_steps"]:
                                continue
                            entry = deepcopy(old)
                            entry["stale"] = True
                            entry["parent_stale"] = True
                            entry["age"] = int(age)
                            if any(not set(context["nodes"]) <= set(state.current_nodes) for context in entry.get("contexts", [])):
                                entry["stale_reason"] = "source_members_removed"
                            prepared[pair] = entry
                    if config["require_full_coverage"] and set(fresh) != set(allowed):
                        raise ValueError("The completed epoch does not cover all allowed pairs.")
                    selection = self._select(state, prepared, config, epoch)
                    selected_pair = selection.pair
                    event = reducer.reduce(
                        state,
                        selection.removed,
                        selection.retained,
                        selection.sign,
                        correlation=selection.correlation,
                        selection_score=selection.score,
                    )
                    reductions += 1
                    retained_center = int(selection.retained)
                    # Project the committed bank to the new active graph.  An
                    # entry whose endpoint disappeared is never resurrected.
                    next_allowed = set(_allowed_pairs(state, config["pairs"]))
                    next_bank = {}
                    for pair, entry in prepared.items():
                        if pair not in next_allowed:
                            continue
                        if pair[0] not in state.current_nodes or pair[1] not in state.current_nodes:
                            continue
                        next_bank[pair] = deepcopy(entry)
                    bank = next_bank
                    event = deepcopy(event)
                    event["selection"] = {
                        "pair": [int(selection.removed), int(selection.retained)],
                        "sign": int(selection.sign),
                        "physical_correlation": float(selection.correlation),
                        "score": float(selection.score),
                        "score_rule": selection.score_rule,
                        "score_direction": selection.score_direction,
                        "weight": float(selection.weight),
                        "weight_power": int(selection.weight_power),
                        "observed_epoch": int(selection.observed_epoch),
                        "age": int(selection.age),
                        "stale": bool(selection.stale),
                        "task_ids": list(selection.task_ids),
                        "contexts": [dict(context) for context in selection.contexts],
                    }
                    event["epoch"] = int(epoch)
                    event["bank_counts"] = {
                        "fresh": len(fresh),
                        "prepared": len(prepared),
                        "stale": sum(1 for entry in prepared.values() if entry.get("stale")),
                        "committed": len(bank),
                    }
                    event["local_tasks"] = len(observations)
                    event["actual_shots"] = int(step_shots)
                    event["seconds"] = float(step_seconds)
                    epoch_history.append(
                        {
                            "epoch": int(epoch),
                            "parent_version": int(state.version - 1),
                            "task_ids": [task["task_id"] for task in viable],
                            "skipped_task_ids": [task["task_id"] for task in skipped],
                            "fresh_pairs": sorted([list(pair) for pair in fresh]),
                            "planned_pairs": len(planned_pairs_set),
                            "allowed_pairs": len(allowed),
                            "fresh_coverage": len(fresh),
                            "missing_fresh": sorted([list(pair) for pair in set(allowed) - set(fresh)]),
                            "bank_prepared": len(prepared),
                            "bank_stale": sum(1 for entry in prepared.values() if entry.get("stale")),
                            "selection": deepcopy(event["selection"]),
                            "reduction": {
                                "b": state.cut_offset,
                                "removed": int(selection.removed),
                                "retained": int(selection.retained),
                            },
                            "budget": {
                                "mandatory": list(mandatory),
                                "steps": list(steps),
                                "quotas": list(quotas),
                                "actual_shots": int(step_shots),
                            },
                        }
                    )
                    coverage_history.append(
                        {
                            "epoch": int(epoch),
                            "allowed_pairs": sorted([list(pair) for pair in allowed]),
                            "planned_pairs": sorted([list(pair) for pair in planned_pairs]),
                            "fresh_pairs": sorted([list(pair) for pair in fresh]),
                            "missing_fresh": sorted([list(pair) for pair in set(allowed) - set(fresh)]),
                            "stale_selected": bool(selection.stale),
                        }
                    )
                    selection_history.append(deepcopy(event["selection"]))
                    budget_history.append(
                        {
                            "epoch": int(epoch),
                            "mandatory": list(mandatory),
                            "training_steps": list(steps),
                            "quotas": list(quotas),
                            "actual_shots": int(step_shots),
                        }
                    )
                    if config["collect"]["summary"] or config["collect"]["epoch_bank"]:
                        bank_history.append(
                            {
                                "epoch": int(epoch),
                                "fresh": [
                                    [int(pair[0]), int(pair[1]), float(entry["physical_correlation"]), int(entry["d"])]
                                    for pair, entry in sorted(fresh.items())
                                ],
                                "prepared": [
                                    [int(pair[0]), int(pair[1]), float(entry["physical_correlation"]), int(entry.get("observed_epoch", epoch)), int(entry.get("age", 0))]
                                    for pair, entry in sorted(prepared.items())
                                ],
                            }
                        )
                    if config["collect"]["local_histories"]:
                        reduction_history.append(event)
                    log_message(
                        config,
                        f"incremental epoch {epoch}: {selection.removed}->{selection.retained} "
                        f"sign={selection.sign} score={selection.score:.6g} stale={selection.stale}.",
                        detail=False,
                    )
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

        if residual_solution is None:
            solution, residual_solution = self._fallback(reducer, state)
            solution_source = "all_plus_reduced_fallback"
            status = "stopped"
            termination_reason = termination_reason or "no_residual_solution"
        else:
            solution = reducer.restore(state, residual_solution)
        cut_value = graph.cut_value(solution)
        cut_offset = state.cut_offset
        ising_constant = state.ising.constant
        if cut_offset is None:
            raise RuntimeError("A graph-origin reduction must have a cut offset.")
        expected_energy = -cut_offset - float(
            state.current_graph().cut_value(residual_solution)
        )
        scale = max(1.0, abs(-cut_value), abs(expected_energy))
        if not math.isclose(-cut_value, expected_energy, rel_tol=1e-9, abs_tol=1e-9 * scale):
            raise ArithmeticError("Incremental reduction and full cut disagree.")
        stale_selected = sum(1 for row in selection_history if row["stale"])
        resources = {
            "shots_used": int(budget.used),
            "training_shots": int(total_training_shots),
            "correlation_shots": int(total_correlation_shots),
            "objective_evaluations": int(total_objective_evaluations),
            "optimizer_steps": int(total_optimizer_steps),
            "readout_shots": 0,
            "source_calls": int(total_tasks),
            "local_tasks_total": int(total_tasks),
            "local_tasks_completed": int(total_tasks),
            "reductions_completed": int(reductions),
            "epochs": int(reductions),
            "num_treads_requested": int(requested_threads),
            "num_treads_applied": int(applied_threads),
            "num_treads": int(applied_threads),
            "unknown_worker_spend": bool(unknown_spend),
            "shots_upper_bound": int(budget.used + (sum(last_quotas) if unknown_spend else 0)),
            "budget": budget.snapshot(),
            "classical_seconds": float(perf_counter() - started),
            "stale_selection_share": (stale_selected / len(selection_history)) if selection_history else None,
        }
        return {
            "solution": solution,
            "status": status,
            "termination_reason": termination_reason,
            "reductions_completed": int(reductions),
            "remaining_node_order": list(state.current_nodes),
            "cut_offset": float(cut_offset),
            "ising_constant": float(ising_constant),
            "reduced_solution": {int(k): int(v) for k, v in residual_solution.items()},
            "reduced_cut_value": float(state.current_graph().cut_value(residual_solution)),
            "solution_source": solution_source,
            "num_treads": int(applied_threads),
            "epoch_history": epoch_history if config["collect"]["summary"] else None,
            "bank_history": bank_history if (config["collect"]["summary"] or config["collect"]["epoch_bank"]) else None,
            "selection_history": selection_history if config["collect"]["summary"] else None,
            "coverage_history": coverage_history,
            "budget_history": budget_history,
            "local_task_history": local_task_history if config["collect"]["local_histories"] else None,
            "reduction_history": reduction_history if config["collect"]["local_histories"] else None,
            "final_bank": [
                {
                    "pair": [int(pair[0]), int(pair[1])],
                    "physical_correlation": float(entry["physical_correlation"]),
                    "observed_epoch": int(entry["observed_epoch"]),
                    "d": int(entry.get("d", 0)),
                    "stale": bool(entry.get("stale", False)),
                }
                for pair, entry in sorted(bank.items())
            ],
            "resources": resources,
            "report": {
                "algorithm_type": self.algorithm_type,
                "reductions_completed": int(reductions),
                "solution_source": solution_source,
                "builder_type": config["initial_builder"]["type"],
                "update_type": config["update"]["type"],
                "bank_policy": config["bank"]["policy"],
                "selection_type": config["selection"]["type"],
                "source_type": config["correlation_source"]["type"],
                "stale_selection_share": resources["stale_selection_share"],
                "local_tasks_total": int(total_tasks),
            },
        }


def main(argv=None):
    """Run the incremental LS exemplar through the common CLI."""
    from ..tools.demo import run_solver_demo

    return run_solver_demo(IncrementalLSRQAOACPU, argv)


if __name__ == "__main__":
    raise SystemExit(main())
