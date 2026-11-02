"""Hierarchical QAOA-in-QAOA with exact block projection.

The public algorithm is a small, explicit hierarchy:

1. partition the current graph into non-overlapping blocks;
2. run one configured readout solver on every nontrivial induced block;
3. normalize each block pattern at its anchor;
4. apply the block spin constraints through ``GraphReducer``;
5. solve the resulting orientation Max-Cut task at the next level.

The module does not copy QAOA training, GPU kernels, or pair-reduction
formulas.  It owns only the hierarchy, task scheduling, and readout glue.
"""

from __future__ import annotations

from copy import deepcopy
import math
import os
from time import perf_counter

import numpy as np

from ..core.budget import ShotBudget
from ..core.graph import WeightedGraph
from ..core.reduction import GraphReducer
from .base import BaseSolver, positive_int
from .components import ExactCPUFinisher
from .hierarchy_components import (
    local_solver_config,
    local_task_requirements,
    normalize_block_pattern,
    project_blocks,
    solve_hierarchy_local_task,
    validate_local_solver,
)

__all__ = ["QAOAInQAOACPU"]


def _run_bounded(executor, function, payloads, workers):
    """Run serializable tasks with a bounded submission window."""
    from concurrent.futures import FIRST_COMPLETED, wait

    iterator = iter(enumerate(payloads))
    pending = {}
    results = [None] * len(payloads)
    for _ in range(min(max(1, workers) * 2, len(payloads))):
        try:
            index, payload = next(iterator)
        except StopIteration:
            break
        pending[executor.submit(function, payload)] = index
    while pending:
        done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
        for future in done:
            index = pending.pop(future)
            results[index] = future.result()
            try:
                next_index, payload = next(iterator)
            except StopIteration:
                continue
            pending[executor.submit(function, payload)] = next_index
    return results


def _seed_for(root_seed, level, block_index):
    base = 0 if root_seed is None else int(root_seed)
    sequence = np.random.SeedSequence(
        [base & 0xFFFFFFFF, int(level), int(block_index)]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _round_robin_quotas(mandatory, costs, limits, total, used):
    """Reserve all readouts, then distribute full iterations round-robin."""
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


def _as_blocks(raw_blocks, graph, level):
    """Normalize a builder result into validated block records."""
    if raw_blocks is None:
        raise ValueError("The partitioner returned no blocks.")
    blocks = []
    for index, item in enumerate(raw_blocks):
        if isinstance(item, dict):
            nodes = item.get("nodes")
            anchor = item.get("anchor")
        else:
            nodes = item
            anchor = None
        if nodes is None:
            raise ValueError("A partition block is missing its nodes.")
        nodes = tuple(sorted(int(node) for node in nodes))
        if not nodes:
            raise ValueError("A partition block is empty.")
        if len(set(nodes)) != len(nodes):
            raise ValueError("A partition block contains a duplicate node.")
        expected_anchor = nodes[0]
        if anchor is not None and int(anchor) != expected_anchor:
            raise ValueError("A block anchor must be its smallest node.")
        blocks.append(
            {
                "level": int(level),
                "block": index,
                "anchor": expected_anchor,
                "nodes": nodes,
                "task_id": f"L{int(level)}:B{index}:A{expected_anchor}",
            }
        )
    by_node = {node: 0 for node in graph.nodes}
    used = set()
    for block in blocks:
        for node in block["nodes"]:
            if node not in by_node:
                raise ValueError("A partition contains a node outside the graph.")
            if node in used:
                raise ValueError("Partition blocks overlap.")
            used.add(node)
    if used != set(graph.nodes):
        raise ValueError("Partition blocks do not cover every graph node.")
    if len({block["anchor"] for block in blocks}) != len(blocks):
        raise ValueError("Partition anchors are not unique.")
    return blocks


def _graph_edge_map(graph):
    return {(u, v): float(value) for u, v, value in graph.edges}


def _compare_projected_graphs(expected, actual, *, tolerance=1e-9):
    if set(expected.nodes) != set(actual.nodes):
        raise ArithmeticError("The reduced anchor set differs from the direct projection.")
    expected_edges = _graph_edge_map(expected)
    actual_edges = _graph_edge_map(actual)
    if set(expected_edges) != set(actual_edges):
        raise ArithmeticError("The reduced edge set differs from the direct projection.")
    scale = max(1.0, max((abs(v) for v in expected_edges.values()), default=0.0))
    for edge, value in expected_edges.items():
        if not math.isclose(value, actual_edges[edge], rel_tol=tolerance, abs_tol=tolerance * scale):
            raise ArithmeticError("A reduced metagraph weight differs from the projection.")


def _build_tree(levels, residual_nodes):
    """Build an explicit composite-ID hierarchy tree."""
    block_nodes = {}
    for record in levels:
        level = int(record["level"])
        for block in record["blocks"]:
            block_nodes[(level, block["anchor"])] = {
                "id": f"L{level}:A{block['anchor']}",
                "kind": "block",
                "level": level,
                "anchor": block["anchor"],
                "size": len(block["nodes"]),
                "local_solver_type": block.get("local_solver_type"),
                "children": [],
            }
    for record in levels:
        level = int(record["level"])
        for block in record["blocks"]:
            parent = block_nodes[(level, block["anchor"])]
            pattern = block.get("normalized_pattern", {})
            for node in block["nodes"]:
                if level == 0:
                    child_id = f"leaf:{node}"
                    child = {
                        "id": child_id,
                        "kind": "leaf",
                        "level": -1,
                        "node": node,
                        "children": [],
                    }
                else:
                    child = block_nodes.get((level - 1, node))
                    if child is None:
                        raise RuntimeError("The hierarchy tree references a missing block.")
                item = {
                    "id": child["id"],
                    "relative_sign": int(pattern.get(node, 1)),
                }
                if level > 0:
                    item["child"] = deepcopy(child)
                parent["children"].append(item)
    roots = []
    top_level = len(levels) - 1
    for node in residual_nodes:
        if top_level < 0:
            roots.append(
                {
                    "id": f"leaf:{node}",
                    "kind": "leaf",
                    "level": -1,
                    "node": int(node),
                    "children": [],
                }
            )
        else:
            root = block_nodes.get((top_level, node))
            if root is None:
                raise RuntimeError("A residual anchor is missing from the hierarchy tree.")
            roots.append(deepcopy(root))
    return {"root_level": top_level, "roots": roots, "blocks": list(block_nodes.values())}


class QAOAInQAOACPU(BaseSolver):
    """Run the basic CPU QAOA-in-QAOA hierarchy.

    Parameters
    ----------
    config : dict, optional
        Root configuration with ``parameters``, ``partitioner``,
        ``local_solver``, ``execution``, ``budget_policy``, ``finisher``,
        ``limits``, ``collect``, ``output``, and optional ``save``.
    partitioner : object, optional
        Test or user replacement with a ``build(graph, version=...)``
        method.  It must obey the same non-overlapping complete partition
        contract.
    local_solver : callable or object, optional
        Test replacement.  A callable receives ``(graph, payload)`` and
        returns either a spin mapping or a result mapping with
        ``solution``.  An object may expose ``solve(graph, payload)``.
    finisher : object, optional
        Test replacement with ``solve(graph)``.

    Notes
    -----
    A block pattern fixes the internal relative spins.  The block anchor is
    the only free orientation at the next level.  The final solution is
    always evaluated on the original graph.  A correct projection is not
    a guarantee of the original global optimum.
    """

    algorithm_type = "qaoa_in_qaoa_cpu"
    supports_parallel = True
    extra_config_fields = (
        "partitioner",
        "local_solver",
        "execution",
        "budget_policy",
        "finisher",
    )
    optional_fields = BaseSolver.optional_fields + (
        "levels_completed",
        "primitive_reductions_completed",
        "local_solver_calls",
        "remaining_node_order",
        "remaining_size",
        "final_cut_offset",
        "final_ising_constant",
        "solution_source",
        "hierarchy_history",
        "block_patterns",
        "parameter_trajectory",
        "local_optimizer_history",
        "coarse_graphs",
        "final_reduction_state",
        "hierarchy_tree",
    )

    def __init__(
        self,
        config=None,
        *,
        partitioner=None,
        local_solver=None,
        finisher=None,
    ):
        super().__init__(config)
        self._injected_partitioner = partitioner
        self._injected_local_solver = local_solver
        self._injected_finisher = finisher

    @property
    def backend_name(self):
        """Return the explicit execution backend of this class."""
        return "gpu" if self.algorithm_type == "qaoa_in_qaoa_gpu" else "cpu"

    def _validate(self, config, graph):
        parameters = config.setdefault("parameters", {})
        if set(parameters) - {"finish_at", "max_levels"}:
            raise ValueError("Unknown QAOA-in-QAOA parameter.")
        parameters.setdefault("finish_at", 4)
        parameters.setdefault("max_levels", None)
        positive_int(parameters["finish_at"], "finish_at")
        max_levels = parameters["max_levels"]
        if max_levels is not None:
            if (
                isinstance(max_levels, bool)
                or not isinstance(max_levels, int)
                or max_levels < 0
            ):
                raise ValueError("max_levels must be None or an integer >= 0.")

        if "partitioner" not in config:
            raise ValueError("partitioner is required.")
        if self._injected_partitioner is not None:
            builder = self._injected_partitioner
        else:
            from ..generators.subgraphs import build_subgraph_builder

            builder = build_subgraph_builder(
                config["partitioner"], seed=config.get("seed")
            )
            config["partitioner"] = builder.config
        partitioner_parameters = config["partitioner"].get("parameters", {})
        if partitioner_parameters.get("rounds", 1) != 1:
            raise ValueError("The hierarchy accepts partitioner.rounds=1 only.")
        if partitioner_parameters.get("boundary", "induced_no_fields") != "induced_no_fields":
            raise ValueError("Only induced_no_fields partition blocks are supported.")

        if "local_solver" not in config:
            raise ValueError("local_solver is required.")
        local = validate_local_solver(config["local_solver"], backend=self.backend_name)
        if self._injected_local_solver is None:
            config["local_solver"] = local
        local_max = local.get("parameters", {}).get("max_vertices")
        partition_max = partitioner_parameters.get("max_size")
        if local_max is not None and partition_max is not None and partition_max > local_max:
            raise ValueError("local_solver.max_vertices is smaller than partitioner.max_size.")

        execution = config.setdefault("execution", {})
        if set(execution) - {"backend", "start_method", "native_threads"}:
            raise ValueError("Unknown execution field.")
        execution.setdefault("backend", "process" if self.backend_name == "cpu" else "gpu_owner")
        execution.setdefault("start_method", "spawn")
        execution.setdefault("native_threads", 1)
        expected_backend = "process" if self.backend_name == "cpu" else "gpu_owner"
        if execution["backend"] != expected_backend:
            raise ValueError(
                f"{self.algorithm_type} requires execution.backend={expected_backend!r}."
            )
        if execution["start_method"] != "spawn":
            raise ValueError("Only execution.start_method='spawn' is implemented.")
        if positive_int(execution["native_threads"], "native_threads") != 1:
            raise ValueError("This hierarchy supports native_threads=1 only.")

        policy = config.setdefault("budget_policy", {})
        if set(policy) - {"type"} or policy.setdefault("type", "level_round_robin") != "level_round_robin":
            raise ValueError("Only budget_policy.type='level_round_robin' is implemented.")

        finisher = (
            self._injected_finisher
            if self._injected_finisher is not None
            else ExactCPUFinisher(config.get("finisher", {"type": "exact_cpu", "parameters": {}}))
        )
        if self._injected_finisher is None:
            config["finisher"] = finisher.config
        if parameters["finish_at"] > finisher.max_vertices:
            raise ValueError("finish_at exceeds the finisher vertex limit.")

        limits = config.setdefault("limits", {})
        if set(limits) - {"shots_total", "max_local_tasks_total"}:
            raise ValueError("Unknown hierarchy limit.")
        limits.setdefault("shots_total", None)
        limits.setdefault("max_local_tasks_total", None)
        if limits["shots_total"] is not None:
            positive_int(limits["shots_total"], "shots_total", minimum=0)
        if limits["max_local_tasks_total"] is not None:
            positive_int(limits["max_local_tasks_total"], "max_local_tasks_total", minimum=0)

        collect = config.setdefault("collect", {})
        allowed_collect = {
            "hierarchy_history",
            "block_patterns",
            "parameter_trajectory",
            "local_optimizer_history",
            "coarse_graphs",
            "final_reduction_state",
        }
        if set(collect) - allowed_collect:
            raise ValueError(f"Unknown collect fields: {set(collect) - allowed_collect}")
        for key in allowed_collect:
            collect.setdefault(key, False)
            if type(collect[key]) is not bool:
                raise ValueError(f"collect.{key} must be boolean.")

    def _local_solve(self, graph, payload):
        if self._injected_local_solver is None:
            return solve_hierarchy_local_task(payload)
        if hasattr(self._injected_local_solver, "solve"):
            result = self._injected_local_solver.solve(graph, payload)
        else:
            result = self._injected_local_solver(graph, payload)
        if isinstance(result, dict) and "solution" in result:
            result = dict(result)
            result.setdefault("resources", {})
        else:
            result = {"solution": result, "resources": {}}
        result.setdefault("level", payload.get("level"))
        result.setdefault("block", payload.get("block"))
        result.setdefault("pid", os.getpid())
        result.setdefault("local_solver_type", type(self._injected_local_solver).__name__)
        if set(result["solution"]) != set(graph.nodes):
            raise ValueError("An injected local solver returned an incomplete solution.")
        return result

    def _run(self, graph, config, rng):
        from ..tools.reporting import log_message

        reducer = GraphReducer()
        state = reducer.start(graph)
        budget = ShotBudget(config["limits"]["shots_total"])
        finish_at = int(config["parameters"]["finish_at"])
        max_levels = config["parameters"]["max_levels"]
        builder = self._injected_partitioner
        if builder is None:
            from ..generators.subgraphs import build_subgraph_builder

            builder = build_subgraph_builder(config["partitioner"], seed=config.get("seed"))
        finisher = (
            self._injected_finisher
            if self._injected_finisher is not None
            else ExactCPUFinisher(config["finisher"])
        )
        collect = config["collect"]
        requested_threads = int(config.get("num_treads", 1) or 1)
        max_tasks = config["limits"]["max_local_tasks_total"]
        local_config = config["local_solver"]

        levels = []
        block_patterns = []
        parameter_trajectory = []
        local_optimizer_history = []
        coarse_graphs = []
        local_solver_calls = 0
        local_tasks_total = 0
        primitive_reductions = 0
        training_shots = 0
        readout_shots = 0
        objective_evaluations = 0
        optimizer_steps = 0
        gradient_evaluations = 0
        state_evaluations = 0
        max_applied_threads = 0
        status = "completed"
        termination_reason = "completed"
        solution_source = None
        residual_solution = None
        residual_graph_snapshot = None
        local_seconds = 0.0
        step_seconds = 0.0
        executor = None

        def set_solution_from_residual(residual, source, final_status, reason):
            nonlocal solution_source, status, termination_reason
            solution_source = source
            status = final_status
            termination_reason = reason
            return reducer.restore(state, residual)

        log_message(config, "QAOA-in-QAOA hierarchy starts.", detail=False)

        if len(graph) == 0:
            residual_solution = {}
            solution = reducer.restore(state, residual_solution)
            solution_source = "empty_graph"
            termination_reason = "empty_graph"
        elif state.ising.edge_count == 0:
            residual_solution = {node: 1 for node in state.current_nodes}
            residual_graph_snapshot = state.current_graph()
            solution = reducer.restore(state, residual_solution)
            solution_source = "zero_interaction_initial"
            termination_reason = "no_interactions"
        else:
            while True:
                n_active = len(state.current_nodes)
                if n_active <= finish_at:
                    residual_graph = state.current_graph()
                    residual_solution, _ = finisher.solve(residual_graph)
                    residual_graph_snapshot = residual_graph
                    solution = set_solution_from_residual(
                        residual_solution, "exact_finisher", "completed", "finish_at"
                    )
                    break

                if state.ising.edge_count == 0:
                    residual_solution = {node: 1 for node in state.current_nodes}
                    residual_graph_snapshot = state.current_graph()
                    solution = set_solution_from_residual(
                        residual_solution, "zero_interaction", "completed", "no_interactions"
                    )
                    break

                if max_levels is not None and len(levels) >= max_levels:
                    residual_solution = {node: 1 for node in state.current_nodes}
                    residual_graph_snapshot = state.current_graph()
                    solution = set_solution_from_residual(
                        residual_solution,
                        "hierarchy_all_plus_fallback",
                        "stopped",
                        "max_levels",
                    )
                    break

                current_graph = state.current_graph()
                raw_blocks = builder.build(current_graph, version=state.version)
                blocks = _as_blocks(raw_blocks, current_graph, len(levels))
                if len(blocks) >= n_active:
                    residual_solution = {node: 1 for node in state.current_nodes}
                    residual_graph_snapshot = state.current_graph()
                    solution = set_solution_from_residual(
                        residual_solution,
                        "hierarchy_all_plus_fallback",
                        "stopped",
                        "no_partition_progress",
                    )
                    break
                nontrivial = [block for block in blocks if len(block["nodes"]) > 1]
                if max_tasks is not None and local_tasks_total + len(nontrivial) > max_tasks:
                    residual_solution = {node: 1 for node in state.current_nodes}
                    residual_graph_snapshot = state.current_graph()
                    solution = set_solution_from_residual(
                        residual_solution,
                        "hierarchy_all_plus_fallback",
                        "stopped",
                        "max_local_tasks_total",
                    )
                    break
                mandatory = []
                costs = []
                step_limits = []
                for block in nontrivial:
                    m, cost, steps = local_task_requirements(local_config)
                    mandatory.append(m)
                    costs.append(cost)
                    step_limits.append(steps)
                allocation = _round_robin_quotas(
                    mandatory, costs, step_limits, budget.total, budget.used
                )
                if allocation is None:
                    residual_solution = {node: 1 for node in state.current_nodes}
                    residual_graph_snapshot = state.current_graph()
                    solution = set_solution_from_residual(
                        residual_solution,
                        "hierarchy_all_plus_fallback",
                        "stopped",
                        "shot_budget",
                    )
                    break
                steps, quotas = allocation
                applied_threads = max(
                    1,
                    min(
                        requested_threads,
                        os.cpu_count() or 1,
                        len(nontrivial) if nontrivial else 1,
                    ),
                )
                if config["execution"]["backend"] == "gpu_owner" or not nontrivial:
                    applied_threads = 1
                if self._injected_local_solver is not None:
                    applied_threads = 1
                max_applied_threads = max(max_applied_threads, applied_threads)
                payloads = []
                for index, block in enumerate(nontrivial):
                    block_graph = current_graph.subgraph(block["nodes"])
                    payloads.append(
                        {
                            "task_id": block["task_id"],
                            "level": len(levels),
                            "block": block["block"],
                            "nodes": list(block["nodes"]),
                            "edges": [
                                [int(u), int(v), float(weight)]
                                for u, v, weight in block_graph.edges
                            ],
                            "local_config": local_config,
                            "seed": _seed_for(config.get("seed"), len(levels), block["block"]),
                            "quota": quotas[index],
                            "collect_optimizer_history": collect["local_optimizer_history"],
                        }
                    )
                started_step = perf_counter()
                try:
                    if not nontrivial:
                        results = []
                    elif applied_threads == 1:
                        results = [
                            self._local_solve(
                                current_graph.subgraph(block["nodes"]), payload
                            )
                            for block, payload in zip(nontrivial, payloads)
                        ]
                    else:
                        import multiprocessing as mp
                        from concurrent.futures import ProcessPoolExecutor

                        executor = ProcessPoolExecutor(
                            max_workers=applied_threads,
                            mp_context=mp.get_context(
                                config["execution"]["start_method"]
                            ),
                        )
                        results = _run_bounded(
                            executor,
                            solve_hierarchy_local_task,
                            payloads,
                            applied_threads,
                        )
                finally:
                    if executor is not None:
                        executor.shutdown(wait=True, cancel_futures=True)
                        executor = None
                step_seconds = perf_counter() - started_step
                local_seconds += step_seconds

                # All local results belong to this level.  Validate before
                # changing the mathematical state.
                result_by_block = {block["block"]: result for block, result in zip(nontrivial, results)}
                normalized_patterns = {}
                pattern_records = []
                level_blocks = []
                for block in blocks:
                    if len(block["nodes"]) == 1:
                        normalized_patterns[block["anchor"]] = {block["anchor"]: 1}
                        pattern_records.append(
                            {
                                "block": block["block"],
                                "anchor": block["anchor"],
                                "nodes": list(block["nodes"]),
                                "task_id": None,
                                "local_solver_type": "singleton",
                                "raw_pattern": {block["anchor"]: 1},
                                "normalized_pattern": {block["anchor"]: 1},
                                "quota": 0,
                            }
                        )
                        level_blocks.append(
                            {
                                **block,
                                "local_solver_type": "singleton",
                                "raw_pattern": {block["anchor"]: 1},
                                "normalized_pattern": {block["anchor"]: 1},
                            }
                        )
                        continue
                    result = result_by_block[block["block"]]
                    if int(result.get("level", -1)) != len(levels):
                        raise RuntimeError("A local task returned a stale hierarchy level.")
                    if int(result.get("block", -1)) != block["block"]:
                        raise RuntimeError("A local task returned another block.")
                    if list(result.get("nodes", [])) != list(block["nodes"]):
                        raise RuntimeError("A local task returned another block node set.")
                    raw = result["solution"]
                    pattern = normalize_block_pattern(block["nodes"], raw)
                    normalized_patterns[block["anchor"]] = pattern["normalized_pattern"]
                    used = int(result.get("resources", {}).get("shots_used", 0))
                    if used > quotas[nontrivial.index(block)]:
                        raise RuntimeError("A local task exceeded its shot quota.")
                    budget.spend(used)
                    readout_shots += int(
                        result.get("resources", {}).get("readout_shots", 0)
                    )
                    training_shots += int(
                        result.get("resources", {}).get("training_shots", 0)
                    )
                    objective_evaluations += int(
                        result.get("resources", {}).get("objective_evaluations", 0)
                    )
                    optimizer_steps += int(
                        result.get("resources", {}).get("optimizer_steps", 0)
                    )
                    gradient_evaluations += int(
                        result.get("resources", {}).get("gradient_evaluations", 0)
                    )
                    state_evaluations += int(
                        result.get("resources", {}).get("state_evaluations", 0)
                    )
                    local_solver_calls += 1
                    local_tasks_total += 1
                    pattern_records.append(
                        {
                            "block": block["block"],
                            "anchor": block["anchor"],
                            "nodes": list(block["nodes"]),
                            "task_id": block["task_id"],
                            "local_solver_type": result.get("local_solver_type"),
                            "raw_pattern": pattern["raw_pattern"],
                            "normalized_pattern": pattern["normalized_pattern"],
                            "quota": quotas[nontrivial.index(block)],
                            "resources": deepcopy(result.get("resources", {})),
                        }
                    )
                    level_blocks.append(
                        {
                            **block,
                            "local_solver_type": result.get("local_solver_type"),
                            "raw_pattern": pattern["raw_pattern"],
                            "normalized_pattern": pattern["normalized_pattern"],
                        }
                    )
                    if collect["parameter_trajectory"]:
                        parameter_trajectory.append(
                            {
                                "level": len(levels),
                                "block": block["block"],
                                "task_id": block["task_id"],
                                "nodes": list(block["nodes"]),
                                "layers": result.get("layers", 0),
                                "angles": result.get("angles"),
                                "local_solver_type": result.get("local_solver_type"),
                            }
                        )
                    if collect["local_optimizer_history"]:
                        local_optimizer_history.append(
                            {
                                "level": len(levels),
                                "block": block["block"],
                                "task_id": block["task_id"],
                                "nodes": list(block["nodes"]),
                                "optimizer_history": result.get("optimizer_history"),
                                "resources": deepcopy(result.get("resources", {})),
                            }
                        )
                if collect["block_patterns"]:
                    block_patterns.append(
                        {
                            "level": len(levels),
                            "blocks": deepcopy(pattern_records),
                        }
                    )

                # Independent direct projection and exact core projection.
                expected_meta, delta_b, direct_records = project_blocks(
                    current_graph,
                    [block["nodes"] for block in blocks],
                    normalized_patterns,
                )
                working = deepcopy(state)
                primitive_before = primitive_reductions
                block_events = []
                for block in blocks:
                    anchor = block["anchor"]
                    for node in block["nodes"]:
                        if node == anchor:
                            continue
                        event = reducer.reduce(
                            working,
                            node,
                            anchor,
                            int(normalized_patterns[anchor][node]),
                            correlation=None,
                            selection_score=None,
                        )
                        event = deepcopy(event)
                        event["correlation_reason"] = "constraint_from_block_readout"
                        event["selection_score_reason"] = "block_readout"
                        block_events.append(event)
                        primitive_reductions += 1
                actual_meta = working.current_graph()
                _compare_projected_graphs(expected_meta, actual_meta)
                actual_delta = float(working.cut_offset - state.cut_offset)
                if not math.isclose(
                    actual_delta,
                    delta_b,
                    rel_tol=1e-9,
                    abs_tol=1e-9 * max(1.0, abs(delta_b)),
                ):
                    raise ArithmeticError("The exact cut offset differs from Delta b.")
                if len(working.current_nodes) != len(blocks):
                    raise ArithmeticError("The block projection did not produce one anchor per block.")
                level_record = {
                    "level": len(levels),
                    "parent_version": int(state.version),
                    "parent_nodes": list(state.current_nodes),
                    "nodes_after": list(working.current_nodes),
                    "blocks": deepcopy(level_blocks),
                    "delta_cut_offset": actual_delta,
                    "cut_offset_before": float(state.cut_offset),
                    "cut_offset_after": float(working.cut_offset),
                    "ising_constant_after": float(working.ising.constant),
                    "primitive_reductions": len(block_events),
                    "local_solver_calls": len(nontrivial),
                    "local_tasks_total": len(nontrivial),
                    "quotas": [int(value) for value in quotas],
                    "steps": [int(value) for value in steps],
                    "requested_threads": requested_threads,
                    "applied_threads": applied_threads,
                    "seconds": step_seconds,
                    "task_seconds": [
                        float(result.get("seconds", 0.0)) for result in results
                    ],
                    "task_pids": [result.get("pid") for result in results],
                    "events": block_events if collect["hierarchy_history"] else None,
                }
                if collect["coarse_graphs"]:
                    level_record["coarse_graph_before"] = current_graph.snapshot()
                    level_record["coarse_graph_after"] = working.current_graph().snapshot()
                    coarse_graphs.append(
                        {
                            "level": len(levels),
                            "before": deepcopy(level_record["coarse_graph_before"]),
                            "after": deepcopy(level_record["coarse_graph_after"]),
                        }
                    )
                else:
                    level_record["coarse_graph_before"] = None
                    level_record["coarse_graph_after"] = None
                if collect["hierarchy_history"]:
                    levels.append(level_record)
                else:
                    # Keep only the compact fields needed by the final report
                    # and tree; the event list is not retained.
                    levels.append(
                        {
                            key: deepcopy(value)
                            for key, value in level_record.items()
                            if key != "events"
                        }
                    )
                state = working
                log_message(
                    config,
                    f"Hierarchy level {len(levels)} accepted: "
                    f"{len(blocks)} blocks, b={state.cut_offset}.",
                    detail=False,
                )

        if residual_graph_snapshot is None:
            residual_graph_snapshot = state.current_graph()
        cut_value = graph.cut_value(solution)
        expected_cut = float(state.cut_offset) + residual_graph_snapshot.cut_value(
            residual_solution
        )
        if not math.isclose(
            cut_value,
            expected_cut,
            rel_tol=1e-9,
            abs_tol=1e-9 * max(1.0, abs(cut_value), abs(expected_cut)),
        ):
            raise ArithmeticError("The restored cut and the hierarchy offset disagree.")
        final_b = float(state.cut_offset)
        final_ising = float(state.ising.constant)
        resources = {
            "shots_used": budget.used,
            "training_shots": training_shots,
            "readout_shots": readout_shots,
            "local_solver_calls": local_solver_calls,
            "local_tasks_total": local_tasks_total,
            "primitive_reductions_completed": primitive_reductions,
            "levels_completed": len(levels),
            "objective_evaluations": objective_evaluations,
            "optimizer_steps": optimizer_steps,
            "gradient_evaluations": gradient_evaluations,
            "state_evaluations": state_evaluations,
            "num_treads_requested": requested_threads,
            "num_treads_applied": max(1, max_applied_threads),
            "native_threads": int(config["execution"]["native_threads"]),
            "backend": "cupy" if self.backend_name == "gpu" else "cpu",
            "max_levels": max_levels,
            "max_local_tasks_total": max_tasks,
            "local_seconds": local_seconds,
            "step_seconds": step_seconds,
        }
        return {
            "solution": solution,
            "status": status,
            "termination_reason": termination_reason,
            "solution_source": solution_source,
            "levels_completed": len(levels),
            "primitive_reductions_completed": primitive_reductions,
            "local_solver_calls": local_solver_calls,
            "remaining_node_order": list(state.current_nodes),
            "remaining_size": len(state.current_nodes),
            "final_cut_offset": final_b,
            "final_ising_constant": final_ising,
            "hierarchy_history": levels if collect["hierarchy_history"] else None,
            "block_patterns": block_patterns if collect["block_patterns"] else None,
            "parameter_trajectory": (
                parameter_trajectory if collect["parameter_trajectory"] else None
            ),
            "local_optimizer_history": (
                local_optimizer_history if collect["local_optimizer_history"] else None
            ),
            "coarse_graphs": coarse_graphs if collect["coarse_graphs"] else None,
            "final_reduction_state": (
                reducer.snapshot(state) if collect["final_reduction_state"] else None
            ),
            "hierarchy_tree": _build_tree(levels, state.current_nodes),
            "resources": resources,
            "optimal_angles": None,
            "expected_cut": None,
            "expected_energy": None,
            "data_status": {
                key: "collected" if collect[key] else "not_collected"
                for key in collect
            },
        }


def main(argv=None):
    """Run a bounded CPU QAOA-in-QAOA example through the common interface."""
    from ..tools.demo import run_solver_demo

    return run_solver_demo(QAOAInQAOACPU, argv)


if __name__ == "__main__":
    raise SystemExit(main())
