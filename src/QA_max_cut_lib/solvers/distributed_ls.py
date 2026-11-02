"""Distributed LS-RQAOA with mean-field episodes and fresh correlations."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
import math

import numpy as np

from ..core.budget import ShotBudget
from ..core.checkpoint import CheckpointManager
from ..core.experiment_ids import stable_hash
from ..core.reduction import GraphReducer, ReductionState
from ..generators.subgraphs import build_subgraph_builder, greedy_anchor_block
from .base import BaseSolver, positive_int
from .block_mean_field import build_block_problem, canonical_edge
from .block_mean_field_qaoa_cpu import BlockMeanFieldQAOACPU
from .components import ExactCPUFinisher
from .qaoa_state import state_at_angles

__all__ = ["DistributedLSRQAOACPU"]


class EpisodeBudget:
    """A per-episode quota that charges one shared root budget."""

    def __init__(self, root, quota):
        self.root = root
        self.limit = None if quota is None else int(quota)
        self._used = 0

    @property
    def used(self):
        return self._used

    def available(self, reserved=0):
        quota_left = None if self.limit is None else self.limit - self._used
        root_left = self.root.available(reserved)
        if quota_left is None:
            return root_left
        if root_left is None:
            return max(0, quota_left - int(reserved))
        return max(0, min(quota_left, root_left) - int(reserved))

    def can_spend(self, amount, reserved=0):
        amount = int(amount)
        return self.available(reserved) is None or self.available(reserved) >= amount

    def spend(self, amount, reserved=0):
        amount = int(amount)
        if self.limit is not None and self._used + amount > self.limit:
            from ..core.budget import ShotBudgetExceeded

            raise ShotBudgetExceeded(amount, self.available(reserved))
        self.root.spend(amount, reserved=reserved)
        self._used += amount
        return self._used

    def snapshot(self):
        return {
            "limit": self.limit,
            "used": self._used,
            "root": self.root.snapshot(),
        }


class DistributedLSRQAOACPU(BaseSolver):
    """Recursive LS-RQAOA with independent mean-field episodes.

    Parameters
    ----------
    config : dict, optional
        Strict configuration with ``messages``, ``partition_builder``,
        ``correlation_source``, ``coverage``, ``selection``, ``update``,
        ``fallback``, ``limits`` and ``checkpoint`` sections.

    Notes
    -----
    Every episode starts from the same parent mean vector and its own
    partition.  Final correlations are measured again at the frozen final
    fields and final angle bank.  Only then is one global reduction applied.
    """

    algorithm_type = "laboratory.distributed_ls_rqaoa_cpu"
    supports_parallel = True
    supports_checkpoint = True
    extra_config_fields = (
        "messages",
        "partition_builder",
        "correlation_source",
        "aggregation",
        "selection",
        "coverage",
        "update",
        "fallback",
        "execution",
        "budget_policy",
        "finisher",
    )
    optional_fields = BaseSolver.optional_fields + (
        "partition_plan",
        "episode_history",
        "final_correlation_tables",
        "transport_history",
        "reduction_history",
        "current_means",
        "reductions_completed",
        "solution_source",
        "needs_repartition",
        "resources",
    )

    def _validate(self, config, graph):
        p = config.setdefault("parameters", {})
        if set(p) - {"finish_at", "max_reductions"}:
            raise ValueError("Unknown distributed parameter.")
        p.setdefault("finish_at", 4)
        p.setdefault("max_reductions", 2)
        positive_int(p["finish_at"], "finish_at", minimum=0)
        positive_int(p["max_reductions"], "max_reductions", minimum=0)

        messages = config.setdefault("messages", {})
        if set(messages) - {"initial", "episode_mean_aggregation", "transport"}:
            raise ValueError("Unknown messages section field.")
        if messages.setdefault("episode_mean_aggregation", "uniform") != "uniform":
            raise ValueError("Only uniform episode mean aggregation is implemented.")
        initial = messages.setdefault("initial", {"type": "biased_spins", "parameters": {"amplitude": 0.1, "spins": {"type": "all_plus"}}})
        self._validate_message_source(graph, initial)
        transport = messages.setdefault("transport", {})
        if set(transport) - {"type", "partition", "block_moments", "angles"}:
            raise ValueError("Unknown transport field.")
        transport.setdefault("type", "carry_survivor_means")
        transport.setdefault("partition", "rebuild")
        transport.setdefault("block_moments", "product_reset")
        transport.setdefault("angles", "restart")
        if transport["type"] not in ("carry_survivor_means", "cold_restart"):
            raise ValueError("Unknown transport type.")
        if transport["partition"] != "rebuild" or transport["block_moments"] != "product_reset" or transport["angles"] != "restart":
            raise ValueError("Transport must rebuild partitions, reset product moments, and restart angles.")

        builder = config.setdefault("partition_builder", {"type": "greedy_abs_weight_partition", "parameters": {"max_size": 3, "rounds": 1, "full_graph_when_fits": True}})
        try:
            build_subgraph_builder(builder, seed=config.get("seed", 0))
        except Exception as error:
            raise ValueError(f"Invalid partition builder: {error}") from error
        if builder["type"] not in ("random_partition", "greedy_abs_weight_partition"):
            raise ValueError("Unsupported partition builder type.")

        source = config.setdefault("correlation_source", {})
        if set(source) - {
            "type", "parameters", "episode_initialization", "episode_execution",
            "mean_field", "initialization", "local_solver", "moments",
            "final_correlations",
        }:
            raise ValueError("Unknown correlation source field.")
        if source.get("type") != "mean_field_episodes_cpu":
            raise ValueError("correlation_source.type must be mean_field_episodes_cpu.")
        sp = source.setdefault("parameters", {})
        sp.setdefault("layers", 1)
        sp.setdefault("max_block_vertices", 18)
        sp.setdefault("dtype", "complex128")
        positive_int(sp["layers"], "layers")
        positive_int(sp["max_block_vertices"], "max_block_vertices")
        if sp["max_block_vertices"] > 24:
            raise ValueError("The block statevector cap is 24 vertices.")
        if sp["dtype"] not in ("complex64", "complex128"):
            raise ValueError("dtype must be complex64 or complex128.")
        if source.setdefault("episode_initialization", "same_parent_means") != "same_parent_means":
            raise ValueError("Only same_parent_means episode initialization is implemented.")
        if source.setdefault("episode_execution", "sequential") != "sequential":
            raise ValueError("Only sequential episode execution is implemented.")
        mean_field = source.setdefault("mean_field", {})
        if set(mean_field) - {"max_sweeps", "schedule", "moment_damping", "angle_damping", "acceptance", "optimizer_state_policy", "stopping"}:
            raise ValueError("Unknown mean_field field.")
        if isinstance(mean_field.setdefault("max_sweeps", 1), bool) or not isinstance(mean_field["max_sweeps"], int) or mean_field["max_sweeps"] < 0:
            raise ValueError("mean_field.max_sweeps must be a nonnegative integer.")
        schedule = mean_field.setdefault("schedule", {"type": "jacobi", "block_order": "as_planned"})
        if set(schedule) - {"type", "block_order"} or schedule.get("type") not in ("jacobi", "gauss_seidel"):
            raise ValueError("Unknown mean-field schedule.")
        if schedule.setdefault("block_order", "as_planned") != "as_planned":
            raise ValueError("Mean-field block_order must be as_planned.")
        for key in ("moment_damping", "angle_damping"):
            val = mean_field.setdefault(key, 0.5)
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)) or not 0 <= float(val) <= 1:
                raise ValueError(f"{key} must be in [0, 1].")
        if mean_field.setdefault("acceptance", "always") != "always":
            raise ValueError("Only acceptance='always' is implemented.")
        if mean_field.setdefault("optimizer_state_policy", "restart_each_update") != "restart_each_update":
            raise ValueError("Only restart_each_update is implemented.")
        if mean_field.setdefault("stopping", {"type": "max_sweeps"}) != {"type": "max_sweeps"}:
            raise ValueError("Only max_sweeps stopping is implemented.")
        source["initialization"] = source.get("initialization", {"type": "fixed_angles", "parameters": {"values": [0.4, 0.2]}})
        sp["layers"] = int(sp["layers"])
        self._validate_local_solver(source)
        finals = source.setdefault("final_correlations", {"mode": "shots", "shots": 16})
        if set(finals) - {"mode", "shots"} or finals.get("mode") not in ("exact", "shots"):
            raise ValueError("Unknown final_correlations field.")
        if finals["mode"] == "shots":
            positive_int(finals.get("shots"), "final shots")
        elif "shots" in finals:
            raise ValueError("Exact final correlations do not take shots.")

        aggregation = config.setdefault("aggregation", {"type": "mean", "parameters": {}})
        if set(aggregation) - {"type", "parameters"} or aggregation.get("type") != "mean" or aggregation.get("parameters") != {}:
            raise ValueError("Only aggregation=mean with empty parameters is implemented.")
        selection = config.setdefault("selection", {"type": "max_abs_correlation", "parameters": {"pairs": "edges", "zero_sign": 1}})
        if set(selection) - {"type", "parameters"} or selection.get("type") != "max_abs_correlation":
            raise ValueError("Only max_abs_correlation selection is implemented.")
        selection.setdefault("parameters", {})
        if set(selection["parameters"]) - {"pairs", "zero_sign"}:
            raise ValueError("Unknown selection parameter.")
        selection["parameters"].setdefault("pairs", "edges")
        selection["parameters"].setdefault("zero_sign", 1)
        if selection["parameters"]["pairs"] not in ("edges", "all_pairs"):
            raise ValueError("selection.pairs must be edges or all_pairs.")
        if selection["parameters"]["zero_sign"] not in (-1, 1) or isinstance(selection["parameters"]["zero_sign"], bool):
            raise ValueError("selection.zero_sign must be +1 or -1.")
        coverage = config.setdefault("coverage", {"require_full": False, "empty_policy": "anchor_partition"})
        if set(coverage) - {"require_full", "empty_policy"}:
            raise ValueError("Unknown coverage field.")
        coverage.setdefault("require_full", False)
        coverage.setdefault("empty_policy", "anchor_partition")
        if type(coverage["require_full"]) is not bool:
            raise ValueError("coverage.require_full must be boolean.")
        if coverage["empty_policy"] not in ("anchor_partition", "error"):
            raise ValueError("coverage.empty_policy must be anchor_partition or error.")
        update = config.setdefault("update", {"type": "refresh_all"})
        if set(update) - {"type"} or update.get("type") != "refresh_all":
            raise ValueError("Only update=refresh_all is implemented.")
        fallback = config.setdefault("fallback", {"type": "sign_parent_means", "zero_sign": 1})
        if set(fallback) - {"type", "zero_sign"} or fallback.get("type") != "sign_parent_means":
            raise ValueError("Only fallback=sign_parent_means is implemented.")
        fallback.setdefault("zero_sign", 1)
        if fallback["zero_sign"] not in (-1, 1) or isinstance(fallback["zero_sign"], bool):
            raise ValueError("fallback.zero_sign must be +1 or -1.")
        execution = config.setdefault("execution", {"backend": "process", "start_method": "spawn", "native_threads": 1})
        if set(execution) - {"backend", "start_method", "native_threads"}:
            raise ValueError("Unknown execution field.")
        if execution.setdefault("backend", "process") != "process" or execution.setdefault("start_method", "spawn") != "spawn":
            raise ValueError("Only process/spawn execution is implemented.")
        positive_int(execution.setdefault("native_threads", 1), "native_threads")
        policy = config.setdefault("budget_policy", {"type": "parent_step_round_robin"})
        if set(policy) - {"type"} or policy.get("type") != "parent_step_round_robin":
            raise ValueError("Only parent_step_round_robin is implemented.")
        limits = config.setdefault("limits", {})
        if set(limits) - {"shots_total", "max_block_updates_total"}:
            raise ValueError("Unknown distributed limit.")
        limits.setdefault("shots_total", None)
        limits.setdefault("max_block_updates_total", None)
        if limits["shots_total"] is not None:
            positive_int(limits["shots_total"], "shots_total", minimum=0)
        if limits["max_block_updates_total"] is not None:
            positive_int(limits["max_block_updates_total"], "max_block_updates_total", minimum=0)
        collect = config.setdefault("collect", {})
        allowed_collect = {
            "reduction_history", "message_transport_history", "episode_history",
            "parameter_trajectory", "local_optimizer_history",
            "final_correlation_tables", "final_reduction_state",
        }
        if set(collect) - allowed_collect:
            raise ValueError(f"Unknown collect field: {set(collect) - allowed_collect}")
        for key in allowed_collect:
            collect.setdefault(key, False)
            if type(collect[key]) is not bool:
                raise ValueError(f"collect.{key} must be boolean.")
        if "finisher" in config:
            finisher = config["finisher"]
            if set(finisher) - {"type", "parameters"} or finisher.get("type") != "exact_cpu":
                raise ValueError("Only exact_cpu finisher is implemented.")
            ExactCPUFinisher(finisher)

    @staticmethod
    def _validate_message_source(graph, source):
        if not isinstance(source, dict) or set(source) - {"type", "parameters"}:
            raise ValueError("initial messages must have type and parameters.")
        kind = source.get("type")
        params = source.setdefault("parameters", {})
        if kind == "zero":
            if params:
                raise ValueError("zero messages take no parameters.")
            return
        if kind == "provided_means":
            if set(params) != {"entries"} or not isinstance(params["entries"], list):
                raise ValueError("provided_means requires entries.")
            seen = set()
            for item in params["entries"]:
                if not isinstance(item, dict) or set(item) != {"node", "mean"}:
                    raise ValueError("Mean entries need node and mean.")
                node = item["node"]
                value = item["mean"]
                if isinstance(node, bool) or not isinstance(node, int) or node not in graph.nodes:
                    raise ValueError("Mean node is not an active graph label.")
                if node in seen or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not -1 <= float(value) <= 1:
                    raise ValueError("Mean entries must be unique and finite in [-1, 1].")
                seen.add(node)
            if seen != set(graph.nodes):
                raise ValueError("provided_means must cover the graph.")
            return
        if kind != "biased_spins":
            raise ValueError("Unknown initial message type.")
        if set(params) - {"amplitude", "spins"}:
            raise ValueError("Unknown biased_spins parameter.")
        amplitude = params.setdefault("amplitude", 0.1)
        if isinstance(amplitude, bool) or not isinstance(amplitude, (int, float)) or not math.isfinite(float(amplitude)) or not 0 <= float(amplitude) <= 1:
            raise ValueError("amplitude must be finite in [0, 1].")
        spins = params.setdefault("spins", {"type": "all_plus"})
        if set(spins) - {"type", "parameters"} or spins.get("type") not in ("all_plus", "provided_spins"):
            raise ValueError("Unknown biased spin source.")
        if spins["type"] == "provided_spins":
            entries = spins.setdefault("parameters", {}).get("entries")
            if not isinstance(entries, list):
                raise ValueError("provided_spins requires entries.")
            seen = set()
            for item in entries:
                if not isinstance(item, dict) or set(item) != {"node", "spin"}:
                    raise ValueError("Spin entries need node and spin.")
                node = item["node"]
                spin = item["spin"]
                if isinstance(node, bool) or not isinstance(node, int) or node not in graph.nodes or node in seen:
                    raise ValueError("Spin node is invalid or duplicated.")
                if spin not in (-1, 1) or isinstance(spin, bool):
                    raise ValueError("Spin must be +1 or -1.")
                seen.add(node)
            if seen != set(graph.nodes):
                raise ValueError("provided_spins must cover the graph.")

    @staticmethod
    def _validate_local_solver(source):
        local = source.setdefault("local_solver", {})
        if set(local) - {"type", "objective", "objective_reference", "estimation", "optimizer"}:
            raise ValueError("Unknown local_solver field.")
        if local.get("type") != "qaoa_ising_statevector_cpu":
            raise ValueError("local_solver.type must be qaoa_ising_statevector_cpu.")
        if local.setdefault("objective_reference", "local_surrogate") != "local_surrogate":
            raise ValueError("Only local_surrogate objective is implemented.")
        from .objectives import validate_objective
        from .components import validate_adam_config, _validate_initialization

        local["objective"] = validate_objective(local.get("objective", {}))
        estimation = local.setdefault("estimation", {})
        if set(estimation) - {"mode", "shots_per_evaluation"}:
            raise ValueError("Unknown local estimation field.")
        estimation.setdefault("mode", "exact")
        estimation.setdefault("shots_per_evaluation", 8)
        if estimation["mode"] not in ("exact", "shots"):
            raise ValueError("Unknown local estimation mode.")
        positive_int(estimation["shots_per_evaluation"], "shots_per_evaluation")
        local["optimizer"] = validate_adam_config(local.get("optimizer", {}))
        moments = source.setdefault("moments", {})
        if set(moments) - {"mode", "shots", "statistics", "shared_sample"}:
            raise ValueError("Unknown moments field.")
        moments.setdefault("mode", "exact")
        moments.setdefault("statistics", ["z_all", "zz_internal_edges"])
        moments.setdefault("shared_sample", True)
        if moments["mode"] not in ("exact", "shots") or moments["statistics"] != ["z_all", "zz_internal_edges"] or moments["shared_sample"] is not True:
            raise ValueError("Unsupported moment protocol.")
        if moments["mode"] == "shots":
            positive_int(moments["shots"], "moments.shots")
        _validate_initialization(source, int(source["parameters"]["layers"]))

    def _initial_means(self, graph, source):
        kind = source["type"]
        params = source["parameters"]
        if kind == "zero":
            return {int(node): 0.0 for node in graph.nodes}
        if kind == "provided_means":
            return {int(item["node"]): float(item["mean"]) for item in params["entries"]}
        amplitude = float(params["amplitude"])
        spins = params["spins"]
        if spins["type"] == "all_plus":
            return {int(node): amplitude for node in graph.nodes}
        return {
            int(item["node"]): amplitude * float(item["spin"])
            for item in spins["parameters"]["entries"]
        }

    def _restart_means(self, graph, source):
        if source["type"] == "provided_means":
            table = {int(item["node"]): float(item["mean"]) for item in source["parameters"]["entries"]}
            missing = set(graph.nodes) - set(table)
            if missing:
                raise ValueError("Cold restart needs initial means for every new label.")
            return {int(node): table[int(node)] for node in graph.nodes}
        if source["type"] == "biased_spins":
            spins = source["parameters"]["spins"]
            if spins["type"] == "provided_spins":
                table = {int(item["node"]): int(item["spin"]) for item in spins["parameters"]["entries"]}
                missing = set(graph.nodes) - set(table)
                if missing:
                    raise ValueError("Cold restart needs initial spins for every new label.")
            else:
                table = {int(node): 1 for node in graph.nodes}
            amplitude = float(source["parameters"]["amplitude"])
            return {int(node): amplitude * table[int(node)] for node in graph.nodes}
        return {int(node): 0.0 for node in graph.nodes}

    def _partitions(self, graph, config, version):
        builder = build_subgraph_builder(
            config["partition_builder"], seed=config.get("seed", 0) + int(version)
        )
        tasks = builder.build(graph, version=version)
        if not tasks:
            return []
        grouped = {}
        for task in tasks:
            grouped.setdefault(int(task["round"]), []).append(tuple(task["nodes"]))
        partitions = []
        for round_index in sorted(grouped):
            blocks = grouped[round_index]
            seen = set()
            for block in blocks:
                if not block or seen.intersection(block):
                    raise ValueError("Partition blocks overlap or are empty.")
                seen.update(block)
            if seen != set(graph.nodes):
                raise ValueError("Partition does not cover the current graph.")
            if len(blocks) < 1:
                raise ValueError("Partition is empty.")
            partitions.append({"partition_id": f"p{int(version)}:r{int(round_index)}", "round": int(round_index), "blocks": [list(block) for block in blocks], "repair": False})
        return partitions

    def _requested_pairs(self, graph, blocks, pairs):
        result = []
        for block in blocks:
            if pairs == "edges":
                result.append([(min(u, v), max(u, v)) for u, v, _ in graph.edges if u in block and v in block])
            else:
                result.append(list(combinations(sorted(block), 2)))
        return result

    def _repair_partition(self, graph, config):
        nonzero = [(min(u, v), max(u, v), w) for u, v, w in graph.edges if w != 0]
        if not nonzero:
            return None
        pair = min((u, v) for u, v, _ in nonzero)
        max_size = int(config["partition_builder"]["parameters"]["max_size"])
        anchor = list(greedy_anchor_block(graph, pair, max_size))
        remaining = [int(node) for node in graph.nodes if node not in set(anchor)]
        blocks = [anchor]
        for start in range(0, len(remaining), max_size):
            blocks.append(remaining[start : start + max_size])
        return {"partition_id": f"repair:{pair[0]}:{pair[1]}", "round": -1, "blocks": blocks, "repair": True, "reason": "anchor_partition"}

    def _episode_config(self, graph, config, partition, means):
        source = config["correlation_source"]
        mean_field = source["mean_field"]
        local = source["local_solver"]
        episode = {
            "schema_version": 1,
            "type": "laboratory.block_mean_field_qaoa_cpu",
            "seed": int(config.get("seed", 0)),
            "num_treads": 1,
            "parameters": {
                "layers": int(source["parameters"]["layers"]),
                "max_sweeps": int(mean_field["max_sweeps"]),
                "max_block_vertices": int(source["parameters"]["max_block_vertices"]),
                "dtype": source["parameters"]["dtype"],
            },
            "partition": {"type": "explicit_partition", "parameters": {"blocks": deepcopy(partition["blocks"])}},
            "initial_messages": {
                "type": "provided_means",
                "parameters": {"entries": [{"node": int(node), "mean": float(means[int(node)])} for node in graph.nodes]},
            },
            "initialization": deepcopy(source["initialization"]),
            "schedule": {"type": mean_field["schedule"]["type"], "block_order": "as_given"},
            "updates": {
                "moment_damping": float(mean_field["moment_damping"]),
                "angle_damping": float(mean_field["angle_damping"]),
                "acceptance": mean_field["acceptance"],
                "optimizer_state_policy": mean_field["optimizer_state_policy"],
            },
            "local_solver": deepcopy(local),
            "moments": deepcopy(source["moments"]),
            "rounding": {"type": "sign_means", "zero_sign": 1},
            "stopping": {"type": "max_sweeps"},
            "execution": deepcopy(config["execution"]),
            "budget_policy": {"type": "sweep_round_robin"},
            "limits": {"shots_total": None, "max_block_updates_total": None},
            "checkpoint": {"enabled": False},
            "collect": {
                "message_history": bool(config["collect"]["episode_history"]),
                "model_history": bool(config["collect"]["episode_history"]),
                "parameter_trajectory": bool(config["collect"]["parameter_trajectory"]),
                "local_optimizer_history": bool(config["collect"]["local_optimizer_history"]),
                "raw_candidate_probabilities": False,
            },
            "output": {"report": 0, "logs": 0, "color": "never"},
        }
        return episode

    def _plan_step(self, root, config, partitions, graph):
        source = config["correlation_source"]
        mean_field = source["mean_field"]
        blocks_per = [len(partition["blocks"]) for partition in partitions]
        sweeps = int(mean_field["max_sweeps"])
        moment = int(source["moments"]["shots"]) if source["moments"]["mode"] == "shots" else 0
        final = int(source["final_correlations"]["shots"]) if source["final_correlations"]["mode"] == "shots" else 0
        final_jobs = 0
        for partition in partitions:
            for pairs in self._requested_pairs(graph, partition["blocks"], config["selection"]["parameters"]["pairs"]):
                if pairs:
                    final_jobs += 1
        inner_min = sum(sweeps * count * moment for count in blocks_per)
        final_min = final_jobs * final
        required = inner_min + final_min
        available = root.available()
        if available is not None and available < required:
            return None, "mandatory_reserve_unavailable", required, final_jobs
        local = source["local_solver"]
        cost = 0
        if local["estimation"]["mode"] == "shots":
            cost = (4 * int(source["parameters"]["layers"]) + 1) * int(local["estimation"]["shots_per_evaluation"])
        updates = []
        for episode_index, count in enumerate(blocks_per):
            for sweep in range(sweeps):
                for block in range(count):
                    updates.append((episode_index, sweep, block))
        max_steps = int(local["optimizer"]["parameters"]["max_steps"])
        allocations = {key: 0 for key in updates}
        if cost == 0:
            for key in updates:
                allocations[key] = max_steps
        else:
            extra = None if available is None else available - required
            if extra is None:
                for key in updates:
                    allocations[key] = max_steps
            else:
                count = min(extra // cost, len(updates) * max_steps)
                for index in range(count):
                    allocations[updates[index]] += 1
        quotas = []
        for episode_index, count in enumerate(blocks_per):
            quota = sweeps * count * moment
            for sweep in range(sweeps):
                for block in range(count):
                    quota += allocations[(episode_index, sweep, block)] * cost
            quotas.append(quota)
        return {
            "quotas": quotas,
            "allocations": allocations,
            "inner_min": inner_min,
            "final_min": final_min,
            "required": required,
            "final_jobs": final_jobs,
        }, None, required, final_jobs

    def _final_correlations(self, graph, episode, config, root, rng):
        source = config["correlation_source"]
        final_cfg = source["final_correlations"]
        layers = int(source["parameters"]["layers"])
        dtype = source["parameters"]["dtype"]
        shots = int(final_cfg["shots"]) if final_cfg["mode"] == "shots" else 0
        pairs_mode = config["selection"]["parameters"]["pairs"]
        tables = []
        for block_index, (block, theta) in enumerate(zip(episode["blocks"], episode["theta"])):
            problem = build_block_problem(graph, block, episode["means"], block_id=block_index)
            pairs = (
                [(min(u, v), max(u, v)) for u, v, _ in graph.edges if u in block and v in block]
                if pairs_mode == "edges"
                else list(combinations(sorted(block), 2))
            )
            if not pairs:
                continue
            state, probabilities = state_at_angles(
                problem.affected_cuts,
                theta,
                layers,
                dtype=dtype,
            )
            norm = float(np.linalg.norm(state))
            if not math.isfinite(norm) or abs(norm - 1.0) > 1e-8:
                raise ValueError("Final conditional state is not normalized.")
            nodes = problem.nodes
            index = np.arange(len(probabilities), dtype=np.uint64)
            shifts = np.arange(len(nodes), dtype=np.uint64)
            bits = ((index[:, None] >> shifts[None, :]) & np.uint64(1)).astype(float)
            spins = 1.0 - 2.0 * bits
            if shots:
                samples = rng.choice(len(probabilities), size=shots, p=probabilities).astype(np.uint64)
                sampled_spins = spins[samples]
                values = {
                    pair: float((sampled_spins[:, nodes.index(pair[0])] * sampled_spins[:, nodes.index(pair[1])]).mean())
                    for pair in pairs
                }
                root.spend(shots)
                origin = "empirical"
            else:
                values = {
                    pair: float(probabilities @ (spins[:, nodes.index(pair[0])] * spins[:, nodes.index(pair[1])]))
                    for pair in pairs
                }
                origin = "exact"
            tables.append(
                {
                    "episode_id": episode["episode_id"],
                    "block_id": block_index,
                    "nodes": [int(node) for node in block],
                    "theta": np.asarray(theta, dtype=float).tolist(),
                    "mode": origin,
                    "shots": shots,
                    "pairs": [[int(u), int(v)] for u, v in pairs],
                    "values": [[int(u), int(v), float(values[(u, v)])] for u, v in pairs],
                }
            )
        return tables

    def _model_sign(self, means, zero_sign):
        return {int(node): 1 if float(value) > 0 else (-1 if float(value) < 0 else int(zero_sign)) for node, value in means.items()}

    def _finish(self, state, means, config, root, rng):
        nodes = list(state.current_nodes)
        if not nodes:
            residual = {}
            source = "empty"
        elif len(nodes) <= int(config["parameters"]["finish_at"]) and "finisher" in config:
            finisher = ExactCPUFinisher(config["finisher"])
            if len(nodes) <= finisher.max_vertices:
                residual, _ = finisher.solve(state.current_graph())
                source = "exact_finisher"
            else:
                residual = self._model_sign(means, int(config["fallback"]["zero_sign"]))
                source = "sign_parent_means"
        else:
            residual = self._model_sign(means, int(config["fallback"]["zero_sign"]))
            source = "sign_parent_means"
        solution = state.restore(residual)
        return solution, source

    def _checkpoint_state(self, state, means, budget, reductions, history, episodes, *, phase, rng, logical_updates=0):
        return {
            "config": deepcopy(self._active_config),
            "graph": state.original_graph.snapshot() if getattr(state, "original_graph", None) is not None else state.current_graph().snapshot(),
            "reduction_state": state.snapshot(),
            "means": {str(node): float(value) for node, value in means.items()},
            "budget": budget.snapshot(),
            "reductions": int(reductions),
            "logical_updates": int(logical_updates),
            "history": deepcopy(history),
            "episodes": deepcopy(episodes),
            "phase": phase,
            "rng_state": deepcopy(rng.bit_generator.state),
        }

    def resume(self, *, control=None):
        """Continue a restored distributed run."""
        if self._resume_state is None:
            raise RuntimeError("resume() requires a solver returned by restore().")
        if self._resume_graph is None:
            raise RuntimeError("The restored checkpoint has no graph.")
        self._resume_mode = True
        return self.solve(self._resume_graph, control=control)

    def _run(self, graph, config, rng):
        self._active_config = config
        state = GraphReducer().start(graph)
        parent_means = self._initial_means(state.current_graph(), config["messages"]["initial"])
        root = ShotBudget(config["limits"]["shots_total"])
        history = []
        episodes = []
        reductions = 0
        logical_updates = 0
        manager = None
        if config.get("checkpoint", {}).get("enabled", False):
            manager = CheckpointManager(
                config["checkpoint"],
                run_id=self._run_id,
                component_id=self.algorithm_type,
                graph=graph,
                config_signature=stable_hash({"config": config, "graph": graph.snapshot()}),
            )
            self._checkpoint_manager = manager
        if self._resume_state:
            saved = self._resume_state
            state = ReductionState.from_snapshot(saved["reduction_state"])
            parent_means = {int(k): float(v) for k, v in saved["means"].items()}
            root = ShotBudget(saved["budget"]["total"])
            if saved["budget"].get("used"):
                root.spend(int(saved["budget"]["used"]))
            reductions = int(saved.get("reductions", 0))
            logical_updates = int(saved.get("logical_updates", 0))
            history = deepcopy(saved.get("history") or [])
            episodes = deepcopy(saved.get("episodes") or [])
        phase = "initialized"
        if manager is not None:
            self.last_checkpoint = manager.write_generation(
                self._checkpoint_state(state, parent_means, root, reductions, history, episodes, phase=phase, rng=rng, logical_updates=logical_updates),
                phase=phase,
            )
        solution_source = "sign_parent_means"
        needs_repartition = False
        final_tables = []
        status = "completed"
        reason = "max_reductions"
        while True:
            current = state.current_graph()
            if self._control is not None and self._control.should_pause("reductions", reductions):
                status = "paused"
                reason = "checkpoint_pause"
                break
            if len(current.nodes) <= int(config["parameters"]["finish_at"]):
                reason = "finish_at"
                break
            if not current.edges:
                reason = "trivial_graph"
                break
            if reductions >= int(config["parameters"]["max_reductions"]):
                reason = "max_reductions"
                break
            partitions = self._partitions(current, config, reductions)
            if config["coverage"]["require_full"]:
                requested = set()
                for partition in partitions:
                    for pair_list in self._requested_pairs(current, partition["blocks"], config["selection"]["parameters"]["pairs"]):
                        requested.update(pair_list)
                if config["selection"]["parameters"]["pairs"] == "edges":
                    required = {(min(u, v), max(u, v)) for u, v, _ in current.edges}
                else:
                    required = set(combinations(sorted(current.nodes), 2))
                if requested != required:
                    status = "stopped"
                    reason = "coverage_incomplete"
                    break
            observed = set()
            for partition in partitions:
                for pair_list in self._requested_pairs(current, partition["blocks"], config["selection"]["parameters"]["pairs"]):
                    observed.update(pair_list)
            nonzero_edges = [(min(u, v), max(u, v)) for u, v, w in current.edges if w != 0]
            if not observed and nonzero_edges and config["coverage"]["empty_policy"] == "anchor_partition":
                repair = self._repair_partition(current, config)
                if repair is not None:
                    partitions.append(repair)
            plan, error, required, final_jobs = self._plan_step(root, config, partitions, current)
            if error is not None:
                status = "stopped"
                reason = error
                break
            total_updates = sum(len(partition["blocks"]) * int(config["correlation_source"]["mean_field"]["max_sweeps"]) for partition in partitions)
            if config["limits"]["max_block_updates_total"] is not None:
                if logical_updates + total_updates > int(config["limits"]["max_block_updates_total"]):
                    status = "stopped"
                    reason = "block_update_limit"
                    break
            step_episodes = []
            final_tables = []
            step_ok = True
            for index, partition in enumerate(partitions):
                episode_config = self._episode_config(current, config, partition, parent_means)
                engine = BlockMeanFieldQAOACPU(episode_config)
                episode_budget = EpisodeBudget(root, plan["quotas"][index])
                try:
                    engine._validate(episode_config, current)
                    result = engine._run(current, episode_config, rng, budget=episode_budget)
                except Exception:
                    step_ok = False
                    break
                if result.get("status") != "completed":
                    step_ok = False
                    break
                means = {int(k): float(v) for k, v in result["means"].items()}
                theta = [np.asarray(value, dtype=float) for value in result["theta_bank"]]
                episode = {
                    "episode_id": partition["partition_id"],
                    "partition": deepcopy(partition),
                    "blocks": [list(block) for block in partition["blocks"]],
                    "means": means,
                    "theta": theta,
                    "model_mean_cut": float(result["model_mean_cut"]),
                    "completed_sweeps": int(result["completed_sweeps"]),
                    "accepted_updates": int(result["accepted_updates"]),
                    "resources": deepcopy(result["resources"]),
                }
                step_episodes.append(episode)
                tables = self._final_correlations(current, episode, config, root, rng)
                final_tables.extend(tables)
                episodes.append(episode)
            if not step_ok:
                status = "stopped"
                reason = "episode_incomplete"
                break
            logical_updates += total_updates
            # Average final c over records that actually queried a pair.
            observations = {}
            for table in final_tables:
                for u, v, value in table["values"]:
                    observations.setdefault((int(u), int(v)), []).append(float(value))
            if not observations:
                status = "stopped"
                reason = "no_correlations"
                break
            correlations = {pair: sum(values) / len(values) for pair, values in observations.items()}
            # Select one observed pair by the declared deterministic rule.
            zero_sign = int(config["selection"]["parameters"]["zero_sign"])
            candidates = []
            for pair, value in correlations.items():
                candidates.append((-abs(value), pair[0], pair[1], value))
            _, u, v, correlation = min(candidates)
            if abs(correlation) <= 1e-15:
                sigma = zero_sign
            else:
                sigma = 1 if correlation > 0 else -1
            removed, retained = (u, v) if u > v else (v, u)
            try:
                event = GraphReducer().reduce(
                    state,
                    removed=removed,
                    retained=retained,
                    sign=sigma,
                    correlation=float(correlation),
                    selection_score=float(abs(correlation)),
                )
            except Exception:
                status = "stopped"
                reason = "reduction_failed"
                break
            reductions += 1
            average_means = {
                int(node): sum(episode["means"][int(node)] for episode in step_episodes) / len(step_episodes)
                for node in current.nodes
            }
            if config["messages"]["transport"]["type"] == "carry_survivor_means":
                next_means = {
                    int(node): float(average_means[int(node)])
                    for node in state.current_nodes
                    if int(node) in average_means
                }
            else:
                next_means = self._restart_means(state.current_graph(), config["messages"]["initial"])
            parent_means = next_means
            needs_repartition = True
            history.append(
                {
                    "reduction": int(reductions),
                    "removed": int(removed),
                    "retained": int(retained),
                    "sign": int(sigma),
                    "correlation": float(correlation),
                    "selection_score": float(abs(correlation)),
                    "episodes": [episode["episode_id"] for episode in step_episodes],
                    "coverage": sorted([list(pair) for pair in observations]),
                    "repair": any(partition.get("repair") for partition in partitions),
                    "cut_offset": float(state.cut_offset),
                    "transport": {
                        "type": config["messages"]["transport"]["type"],
                        "means": {str(node): float(value) for node, value in parent_means.items()},
                        "kind": "means_only_product_reinitialization",
                        "conditional_projection": False,
                    },
                    "event": event,
                }
            )
            phase = "reduction_committed"
            if manager is not None:
                self.last_checkpoint = manager.write_generation(
                    self._checkpoint_state(state, parent_means, root, reductions, history, episodes, phase=phase, rng=rng, logical_updates=logical_updates),
                    phase=phase,
                )
        solution, source = self._finish(state, parent_means, config, root, rng)
        return {
            "solution": solution,
            "status": status,
            "termination_reason": reason,
            "partition_plan": [episode["partition"] for episode in episodes],
            "episode_history": episodes if config["collect"]["episode_history"] else None,
            "final_correlation_tables": final_tables if config["collect"]["final_correlation_tables"] else None,
            "transport_history": [
                row["transport"] for row in history
            ] if config["collect"]["message_transport_history"] else None,
            "reduction_history": history if config["collect"]["reduction_history"] else None,
            "current_means": {str(node): float(value) for node, value in parent_means.items()},
            "reductions_completed": int(reductions),
            "solution_source": source,
            "needs_repartition": bool(needs_repartition),
            "resources": {
                "shots_used": root.used,
                "budget": root.snapshot(),
                "reductions": int(reductions),
                "episodes": len(episodes),
                "logical_inner_updates": int(logical_updates),
                "final_correlation_jobs": int(len(final_tables)),
                "num_treads": int(config.get("num_treads", 1)),
                "num_treads_applied": 1,
                "backend": config["execution"]["backend"],
            },
        }


def main(argv=None):
    from ..tools.demo import run_solver_demo

    return run_solver_demo(DistributedLSRQAOACPU, argv)


if __name__ == "__main__":
    raise SystemExit(main())
