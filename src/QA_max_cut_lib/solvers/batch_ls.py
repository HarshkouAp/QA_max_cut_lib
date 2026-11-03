"""Batch LS-RQAOA controller for frozen proposal forests.

This is a compact CPU implementation of the batch contract: all local
observations complete before the graph changes, proposals are frozen, and
one of two explicit reconciliation policies is applied.  It does not claim
bitwise equivalence with the historical MK_3 controller.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from time import perf_counter

from ..core.budget import ShotBudget
from ..core.reduction import GraphReducer
from ..generators.plans import SubproblemPlan, _coverage
from ..generators.subgraphs import build_subgraph_builder
from .base import BaseSolver, positive_int
from .components import ExactCPUFinisher, SelectionResult
from .incremental_ls import (
    _round_robin_quotas,
    _stable_seed,
    _task_requirements,
    normalize_incremental_source,
)
from .ls_worker import solve_local_task
from .strategies import ParityForest

__all__ = ["BatchLSRQAOACPU", "BatchProposal"]


@dataclass(frozen=True)
class BatchProposal:
    task_id: str
    removed: int
    retained: int
    sign: int
    correlation: float
    score: float
    pair: tuple
    weight: float


def _canonical(u, v):
    return (min(int(u), int(v)), max(int(u), int(v)))


class BatchLSRQAOACPU(BaseSolver):
    """Run one source epoch and reconcile frozen pair proposals."""

    algorithm_type = "laboratory.batch_ls_rqaoa_cpu"
    supports_parallel = True
    extra_config_fields = (
        "finish_at", "batch", "messages", "local_problem", "source",
        "initial_builder", "finisher", "limits", "max_reductions",
    )
    optional_fields = BaseSolver.optional_fields + (
        "reductions_completed", "remaining_node_order", "cut_offset",
        "ising_constant", "reduced_solution", "reduced_cut_value",
        "solution_source", "batch_history", "reduction_history",
        "proposal_history", "resources",
    )

    def __init__(self, config=None):
        config = deepcopy(config or {})
        output = config.setdefault("output", {})
        for key in ("report", "logs"):
            if key in config:
                value = config.pop(key)
                output.setdefault(key, value)
        super().__init__(config)

    def _validate(self, config, graph):
        finish_at = config.setdefault("finish_at", min(4, max(1, len(graph))))
        positive_int(finish_at, "finish_at")
        max_red = config.setdefault("max_reductions", None)
        if max_red is not None:
            positive_int(max_red, "max_reductions", minimum=0)
        batch = config.setdefault("batch", {})
        if set(batch) - {"policy", "proposal_order", "pair_scope", "proposal_repetition"}:
            raise ValueError("Unknown batch field.")
        batch.setdefault("policy", "parity_forest")
        batch.setdefault("proposal_order", "task_id")
        batch.setdefault("pair_scope", "frozen_origin")
        batch.setdefault("proposal_repetition", "last")
        if batch["policy"] not in ("parity_forest", "active_endpoints_only"):
            raise ValueError("Unknown batch policy.")
        if batch["pair_scope"] not in ("frozen_origin", "current_edge_only"):
            raise ValueError("Unknown proposal scope.")
        local = config.setdefault("local_problem", {"type": "legacy_field_affine", "field_scale": 1.0, "rho_feedback": False})
        if set(local) - {"type", "field_scale", "rho_feedback"}:
            raise ValueError("Unknown local_problem field.")
        if local.get("type") not in ("legacy_field_affine", "conditional_cut"):
            raise ValueError("Unknown local_problem type.")
        if config.get("messages"):
            raise ValueError("Batch message feedback is not implemented in this CPU slice.")
        source = config.get("source", config.get("correlation_source"))
        if source is None:
            raise ValueError("batch solver requires source.")
        config["correlation_source"] = normalize_incremental_source(source)
        if config["correlation_source"]["type"] not in ("qaoa_statevector_cpu", "qaoa_analytic_p1_cpu"):
            raise ValueError("Batch CPU source must be statevector or analytic p=1.")
        builder = config.setdefault(
            "initial_builder",
            {"type": "random_partition", "parameters": {"max_size": 3, "rounds": 1, "full_graph_when_fits": True}},
        )
        builder_config = deepcopy(builder)
        parser = builder_config.setdefault("parameters", {})
        for key in ("max_size", "rounds", "full_graph_when_fits"):
            if key in builder_config:
                parser.setdefault(key, builder_config.pop(key))
        built = build_subgraph_builder(builder_config, seed=config.get("seed"))
        config["initial_builder"] = built.config
        limits = config.setdefault("limits", {})
        if set(limits) - {"max_parent_epochs", "max_source_calls", "max_shots"}:
            raise ValueError("Unknown batch limit.")
        limits.setdefault("max_parent_epochs", 4)
        limits.setdefault("max_source_calls", 24)
        limits.setdefault("max_shots", 4096)
        for key in limits:
            positive_int(limits[key], key)
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
            raise ValueError("finish_at exceeds finisher max_vertices.")
        collect = config.setdefault("collect", {})
        if set(collect) - {"batch_history", "proposal_history", "reduction_history"}:
            raise ValueError("Unknown collect field.")
        for key in ("batch_history", "proposal_history", "reduction_history"):
            collect.setdefault(key, False)
            if type(collect[key]) is not bool:
                raise TypeError(f"collect.{key} must be boolean.")

    def _make_payload(self, task, state, config, steps, quota):
        source_config = deepcopy(config["correlation_source"])
        if source_config["type"] == "qaoa_statevector_cpu":
            source_config["optimizer"]["parameters"]["max_steps"] = int(steps)
        return {
            "task_id": task["task_id"],
            "nodes": [int(node) for node in task["nodes"]],
            "edges": [[int(u), int(v), float(w)] for u, v, w in task["local_graph"].edges],
            "parent_version": int(state.version),
            "source_config": source_config,
            "seed": _stable_seed(config.get("seed"), "source", state.version, task["task_id"]),
            "pair_domain": "edges",
            "shot_quota": int(quota),
            "training_steps": int(steps),
            "collect_optimizer_history": False,
            "collect_objective_history": False,
        }

    def _proposal_from_result(self, result, state):
        graph = state.current_graph()
        values = result.get("values", {})
        candidates = []
        for pair, value in values.items():
            key = _canonical(*pair)
            if graph.weight(*key) == 0.0:
                continue
            c = float(value)
            if not math.isfinite(c) or abs(c) > 1.0 + 1e-9:
                raise ValueError("Local correlation is invalid.")
            candidates.append((abs(c), key, c))
        if not candidates:
            return None
        _, pair, c = min(candidates, key=lambda item: (-item[0], item[1]))
        sign = 1 if c > 0 else -1 if c < 0 else 1
        return BatchProposal(
            task_id=str(result["task_id"]),
            removed=pair[0],
            retained=pair[1],
            sign=sign,
            correlation=c,
            score=abs(c),
            pair=pair,
            weight=float(graph.weight(*pair)),
        )

    def _apply_proposal(self, state, proposal, policy, forest):
        if policy == "parity_forest":
            event = forest.add(proposal.removed, proposal.retained, proposal.sign, task_id=proposal.task_id)
            if event["status"] != "accepted":
                return {"status": event["status"], "proposal": proposal}
            removed, retained = event["removed"], event["retained"]
        else:
            if proposal.removed not in state.current_nodes or proposal.retained not in state.current_nodes:
                return {"status": "skipped_removed_endpoint", "proposal": proposal}
            removed, retained = proposal.removed, proposal.retained
        if removed not in state.current_nodes or retained not in state.current_nodes:
            return {"status": "skipped_missing_root", "proposal": proposal}
        return {"status": "accepted", "proposal": proposal, "removed": removed, "retained": retained}

    def _run(self, graph, config, rng):
        reducer = GraphReducer()
        state = reducer.start(graph)
        budget = ShotBudget(config["limits"]["max_shots"])
        finisher = ExactCPUFinisher(config["finisher"])
        batch = config["batch"]
        reductions = 0
        epochs = 0
        source_calls = 0
        training_shots = 0
        correlation_shots = 0
        batch_history = []
        proposal_history = []
        reduction_history = []
        status = "completed"
        termination = "completed"
        residual = None
        solution_source = None
        while True:
            if len(state.current_nodes) <= int(config["finish_at"]):
                local_solution, _ = finisher.solve(state.current_graph())
                residual = {int(k): int(v) for k, v in local_solution.items()}
                solution_source = "exact_finisher"
                termination = "finish_at"
                break
            if config["max_reductions"] is not None and reductions >= int(config["max_reductions"]):
                residual = {int(node): 1 for node in state.current_nodes}
                solution_source = "all_plus_reduced_fallback"
                status, termination = "stopped", "max_reductions"
                break
            if epochs >= config["limits"]["max_parent_epochs"]:
                residual = {int(node): 1 for node in state.current_nodes}
                solution_source = "all_plus_reduced_fallback"
                status, termination = "stopped", "max_parent_epochs"
                break
            if state.ising.edge_count == 0:
                residual = {int(node): 1 for node in state.current_nodes}
                solution_source = "zero_interaction"
                termination = "no_interactions"
                break
            builder = build_subgraph_builder(
                config["initial_builder"],
                seed=_stable_seed(config.get("seed"), "builder", state.version),
            )
            tasks = builder.build(state.current_graph(), version=state.version)
            allowed_edges = {
                _canonical(u, v) for u, v, w in state.current_graph().edges if float(w) != 0.0
            }
            viable = []
            for task in tasks:
                nodes = set(int(node) for node in task["nodes"])
                requested = [pair for pair in allowed_edges if pair[0] in nodes and pair[1] in nodes]
                if requested:
                    task = dict(task)
                    task["local_graph"] = state.current_graph().subgraph(task["nodes"])
                    task["requested"] = requested
                    viable.append(task)
            if not viable:
                residual = {int(node): 1 for node in state.current_nodes}
                solution_source = "all_plus_reduced_fallback"
                status, termination = "stopped", "empty_proposal_epoch"
                break
            mandatory, costs, iterations = [], [], []
            for _ in viable:
                m, c, i = _task_requirements(config["correlation_source"])
                mandatory.append(m)
                costs.append(c)
                iterations.append(i)
            allocation = _round_robin_quotas(
                mandatory, costs, iterations, budget.total, budget.used
            )
            if allocation is None:
                residual = {int(node): 1 for node in state.current_nodes}
                solution_source = "all_plus_reduced_fallback"
                status, termination = "stopped", "budget_exhausted"
                break
            steps, quotas = allocation
            payloads = [
                self._make_payload(task, state, config, steps[index], quotas[index])
                for index, task in enumerate(viable)
            ]
            results = [solve_local_task(payload) for payload in payloads]
            source_calls += len(results)
            if config["limits"]["max_source_calls"] is not None and source_calls > config["limits"]["max_source_calls"]:
                residual = {int(node): 1 for node in state.current_nodes}
                solution_source = "all_plus_reduced_fallback"
                status, termination = "stopped", "max_source_calls"
                break
            proposals = []
            for index, result in enumerate(results):
                if result["task_id"] != viable[index]["task_id"]:
                    raise RuntimeError("Batch worker task mismatch.")
                used = sum(int(result.get("resources", {}).get(key, 0)) for key in ("training_shots", "correlation_shots", "readout_shots"))
                if used > quotas[index]:
                    raise RuntimeError("Batch worker exceeded quota.")
                budget.spend(used)
                training_shots += int(result.get("resources", {}).get("training_shots", 0))
                correlation_shots += int(result.get("resources", {}).get("correlation_shots", 0))
                proposal = self._proposal_from_result(result, state)
                if proposal is not None:
                    proposals.append(proposal)
            # All source work for the frozen epoch is complete.  Reconcile
            # without re-running any source.
            forest = ParityForest(state.current_nodes)
            decisions = []
            for proposal in sorted(proposals, key=lambda item: item.task_id):
                if config["max_reductions"] is not None and reductions >= int(config["max_reductions"]):
                    decisions.append({"status": "skipped_max_reductions", "proposal": proposal})
                    continue
                decision = self._apply_proposal(state, proposal, batch["policy"], forest)
                decisions.append(decision)
                if decision["status"] == "accepted":
                    reducer.reduce(
                        state,
                        decision["removed"],
                        decision["retained"],
                        proposal.sign,
                        correlation=proposal.correlation,
                        selection_score=proposal.score,
                    )
                    reductions += 1
                    reduction_history.append(
                        {
                            "removed": int(decision["removed"]),
                            "retained": int(decision["retained"]),
                            "sign": int(proposal.sign),
                            "physical_correlation": float(proposal.correlation),
                            "score": float(proposal.score),
                            "task_id": proposal.task_id,
                            "b": state.cut_offset,
                        }
                    )
            epochs += 1
            batch_history.append(
                {
                    "epoch": epochs,
                    "proposals": len(proposals),
                    "accepted": sum(1 for item in decisions if item["status"] == "accepted"),
                    "conflict": sum(1 for item in decisions if item["status"] == "conflict"),
                    "redundant": sum(1 for item in decisions if item["status"] == "redundant"),
                    "b": state.cut_offset,
                }
            )
            if config["collect"]["proposal_history"]:
                proposal_history.append(
                    [
                        {
                            "task_id": item.task_id,
                            "pair": [int(item.removed), int(item.retained)],
                            "sign": int(item.sign),
                            "physical_correlation": float(item.correlation),
                            "score": float(item.score),
                        }
                        for item in proposals
                    ]
                )
        if residual is None:
            residual = {int(node): 1 for node in state.current_nodes}
            solution_source = "all_plus_reduced_fallback"
        solution = reducer.restore(state, residual)
        cut = graph.cut_value(solution)
        expected = -float(state.cut_offset) - state.current_graph().cut_value(residual)
        if not math.isclose(-cut, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise ArithmeticError("Batch reduction invariant failed.")
        resources = {
            "shots_used": int(budget.used),
            "training_shots": int(training_shots),
            "correlation_shots": int(correlation_shots),
            "source_calls": int(source_calls),
            "reductions_completed": int(reductions),
            "num_treads_requested": int(config.get("num_treads", 1)),
            "num_treads_applied": 1,
            "num_treads": 1,
            "budget": budget.snapshot(),
        }
        return {
            "solution": solution,
            "status": status,
            "termination_reason": termination,
            "reductions_completed": int(reductions),
            "remaining_node_order": list(state.current_nodes),
            "cut_offset": float(state.cut_offset),
            "ising_constant": float(state.ising.constant),
            "reduced_solution": {int(k): int(v) for k, v in residual.items()},
            "reduced_cut_value": float(state.current_graph().cut_value(residual)),
            "solution_source": solution_source,
            "num_treads": 1,
            "batch_history": batch_history if config["collect"]["batch_history"] else None,
            "proposal_history": proposal_history if config["collect"]["proposal_history"] else None,
            "reduction_history": reduction_history if config["collect"]["reduction_history"] else None,
            "resources": resources,
            "report": {
                "algorithm_type": self.algorithm_type,
                "reductions_completed": int(reductions),
                "batch_policy": batch["policy"],
                "source_calls": int(source_calls),
                "solution_source": solution_source,
            },
        }


def main(argv=None):
    from ..tools.demo import run_solver_demo
    return run_solver_demo(BatchLSRQAOACPU, argv)


if __name__ == "__main__":
    raise SystemExit(main())
