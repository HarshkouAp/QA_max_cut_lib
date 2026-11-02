"""Block mean-field QAOA with Gauss-Seidel or Jacobi message updates."""

from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from ..core.budget import ShotBudget, ShotBudgetExceeded
from ..core.checkpoint import CheckpointManager
from ..core.experiment_ids import stable_hash
from .base import BaseSolver, positive_int
from .block_mean_field import (
    MeanFieldError,
    build_block_problem,
    canonical_edge,
    damped_moments,
    pair_probabilities,
)
from .components import _initial_angles, _validate_initialization, validate_adam_config
from .objectives import objective_value, validate_objective
from .qaoa_state import state_at_angles
from .qaoa_training import train_statevector

__all__ = ["BlockMeanFieldQAOACPU"]


class BlockMeanFieldQAOACPU(BaseSolver):
    """Iterate local QAOA blocks with classical mean-spin messages.

    Parameters
    ----------
    config : dict, optional
        Strict configuration with ``partition``, ``initial_messages``,
        ``schedule``, ``updates``, ``local_solver``, ``moments``,
        ``rounding`` and ``stopping`` sections.

    Notes
    -----
    The block distributions are represented by a compact moment bank
    ``(m_i, q_ij)``.  Local states are full CPU statevectors.  Output is a
    sign-rounded full classical assignment, not a global quantum readout.
    """

    algorithm_type = "laboratory.block_mean_field_qaoa_cpu"
    supports_parallel = True
    supports_checkpoint = True
    extra_config_fields = (
        "partition",
        "initial_messages",
        "schedule",
        "updates",
        "local_solver",
        "moments",
        "rounding",
        "stopping",
        "execution",
        "budget_policy",
    )
    optional_fields = BaseSolver.optional_fields + (
        "means",
        "edge_moments",
        "model_mean_cut",
        "model_energy",
        "model_origin",
        "partition",
        "schedule",
        "completed_sweeps",
        "accepted_updates",
        "completed_local_jobs",
        "partial_sweep",
        "pending_batch",
        "model_history",
        "message_history",
        "theta_bank",
        "initial_means",
        "moment_damping",
        "angle_damping",
        "rounding",
        "resources",
    )

    def _validate(self, config, graph):
        if "readout" in config:
            raise ValueError("Block mean-field output is sign_means; global readout is not supported.")
        if "objective" in config:
            raise ValueError("The objective belongs in local_solver.objective.")
        p = config.setdefault("parameters", {})
        allowed = {"layers", "max_sweeps", "max_block_vertices", "dtype"}
        if set(p) - allowed:
            raise ValueError("Unknown block mean-field parameter.")
        p.setdefault("layers", 1)
        p.setdefault("max_sweeps", 1)
        p.setdefault("max_block_vertices", 18)
        p.setdefault("dtype", "complex128")
        positive_int(p["layers"], "layers")
        positive_int(p["max_block_vertices"], "max_block_vertices")
        if p["max_block_vertices"] > 24:
            raise ValueError("The full-state block cap is 24 vertices.")
        if isinstance(p["max_sweeps"], bool) or not isinstance(p["max_sweeps"], int) or p["max_sweeps"] < 0:
            raise ValueError("max_sweeps must be a nonnegative integer.")
        if p["dtype"] not in ("complex64", "complex128"):
            raise ValueError("dtype must be complex64 or complex128.")

        partition = config.setdefault("partition", {})
        if set(partition) - {"type", "parameters"} or partition.get("type") != "explicit_partition":
            raise ValueError("Only explicit_partition is implemented.")
        pp = partition.setdefault("parameters", {})
        if set(pp) != {"blocks"} or not isinstance(pp["blocks"], list):
            raise ValueError("explicit_partition requires parameters.blocks.")
        self._validate_partition(graph, pp["blocks"], int(p["max_block_vertices"]))

        messages = config.setdefault("initial_messages", {})
        if set(messages) - {"type", "parameters"}:
            raise ValueError("Unknown initial_messages field.")
        if messages.get("type") not in ("provided_means", "biased_spins", "zero"):
            raise ValueError("Unknown initial message type.")
        mp = messages.setdefault("parameters", {})
        if messages["type"] == "zero":
            if set(mp):
                raise ValueError("zero initial messages take no parameters.")
        elif messages["type"] == "provided_means":
            if set(mp) != {"entries"} or not isinstance(mp["entries"], list):
                raise ValueError("provided_means requires entries.")
            self._validate_means_entries(graph, mp["entries"])
        else:
            if set(mp) - {"amplitude", "spins"}:
                raise ValueError("Unknown biased_spins parameter.")
            mp.setdefault("amplitude", 0.1)
            if not isinstance(mp["amplitude"], (int, float)) or isinstance(mp["amplitude"], bool):
                raise ValueError("amplitude must be finite in [0, 1].")
            if not math.isfinite(float(mp["amplitude"])) or not 0 <= float(mp["amplitude"]) <= 1:
                raise ValueError("amplitude must be finite in [0, 1].")
            source = mp.setdefault("spins", {"type": "all_plus"})
            if set(source) - {"type", "parameters"} or source.get("type") not in ("all_plus", "provided_spins"):
                raise ValueError("biased spins source must be all_plus or provided_spins.")
            if source["type"] == "provided_spins":
                sp = source.setdefault("parameters", {})
                if set(sp) != {"entries"} or not isinstance(sp["entries"], list):
                    raise ValueError("provided_spins requires entries.")
                self._validate_spin_entries(graph, sp["entries"])

        initialization = config.setdefault("initialization", {})
        if initialization.get("type") == "angle_bank":
            entries = initialization.setdefault("parameters", {}).get("entries")
            if not isinstance(entries, list) or len(entries) != len(
                config["partition"]["parameters"]["blocks"]
            ):
                raise ValueError("angle_bank requires one entry per block.")
            for index, item in enumerate(entries):
                if not isinstance(item, dict) or set(item) != {"block_id", "values"}:
                    raise ValueError("Each angle bank entry needs block_id and values.")
                if item["block_id"] != index:
                    raise ValueError("angle_bank entries must follow block order.")
                values = np.asarray(item["values"], dtype=float)
                if values.shape != (2 * int(p["layers"]),) or not np.isfinite(values).all():
                    raise ValueError("Each angle bank needs exactly 2*layers finite values.")
        else:
            values = initialization.get("parameters", {}).get("values")
            if (
                initialization.get("type") == "fixed_angles"
                and isinstance(values, list)
                and values
                and isinstance(values[0], (list, tuple))
            ):
                if len(values) != len(config["partition"]["parameters"]["blocks"]):
                    raise ValueError("A nested angle bank needs one row per block.")
                for row in values:
                    array = np.asarray(row, dtype=float)
                    if array.shape != (2 * int(p["layers"]),) or not np.isfinite(array).all():
                        raise ValueError("Each angle row needs exactly 2*layers finite values.")
            else:
                _validate_initialization(config, int(p["layers"]))

        schedule = config.setdefault("schedule", {"type": "jacobi", "block_order": "as_given"})
        if set(schedule) - {"type", "block_order"}:
            raise ValueError("Unknown schedule field.")
        schedule.setdefault("type", "jacobi")
        schedule.setdefault("block_order", "as_given")
        if schedule["type"] not in ("gauss_seidel", "jacobi"):
            raise ValueError("schedule.type must be gauss_seidel or jacobi.")
        if schedule["block_order"] != "as_given":
            raise ValueError("Only block_order='as_given' is implemented.")

        updates = config.setdefault(
            "updates",
            {
                "moment_damping": 0.5,
                "angle_damping": 1.0,
                "acceptance": "always",
                "optimizer_state_policy": "restart_each_update",
            },
        )
        if set(updates) - {"moment_damping", "angle_damping", "acceptance", "optimizer_state_policy"}:
            raise ValueError("Unknown updates field.")
        updates.setdefault("moment_damping", 0.5)
        updates.setdefault("angle_damping", 1.0)
        updates.setdefault("acceptance", "always")
        updates.setdefault("optimizer_state_policy", "restart_each_update")
        for key in ("moment_damping", "angle_damping"):
            value = updates[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise ValueError(f"{key} must be in [0, 1].")
        if updates["acceptance"] != "always":
            raise ValueError("Only acceptance='always' is implemented.")
        if updates["optimizer_state_policy"] != "restart_each_update":
            raise ValueError("Only restart_each_update is implemented.")

        local = config.setdefault("local_solver", {})
        if set(local) - {"type", "objective", "objective_reference", "estimation", "optimizer"}:
            raise ValueError("Unknown local_solver field.")
        if local.get("type") != "qaoa_ising_statevector_cpu":
            raise ValueError("local_solver.type must be qaoa_ising_statevector_cpu.")
        local["objective"] = validate_objective(local.get("objective", {}))
        if local.setdefault("objective_reference", "local_surrogate") != "local_surrogate":
            raise ValueError("Only objective_reference='local_surrogate' is implemented.")
        estimation = local.setdefault("estimation", {})
        if set(estimation) - {"mode", "shots_per_evaluation"}:
            raise ValueError("Unknown local estimation field.")
        estimation.setdefault("mode", "exact")
        estimation.setdefault("shots_per_evaluation", 8)
        if estimation["mode"] not in ("exact", "shots"):
            raise ValueError("local estimation.mode must be exact or shots.")
        positive_int(estimation["shots_per_evaluation"], "shots_per_evaluation")
        local["optimizer"] = validate_adam_config(local.get("optimizer", {}))

        moments = config.setdefault("moments", {})
        if set(moments) - {"mode", "shots", "statistics", "shared_sample"}:
            raise ValueError("Unknown moments field.")
        moments.setdefault("mode", "exact")
        moments.setdefault("statistics", ["z_all", "zz_internal_edges"])
        moments.setdefault("shared_sample", True)
        if moments["mode"] not in ("exact", "shots"):
            raise ValueError("moments.mode must be exact or shots.")
        if moments["statistics"] != ["z_all", "zz_internal_edges"]:
            raise ValueError("moments.statistics must request z_all and zz_internal_edges.")
        if moments["shared_sample"] is not True:
            raise ValueError("moments.shared_sample must be true.")
        if moments["mode"] == "shots":
            positive_int(moments.get("shots"), "moments.shots")
        elif "shots" in moments:
            raise ValueError("Exact moments do not take shots.")

        rounding = config.setdefault("rounding", {"type": "sign_means", "zero_sign": 1})
        if set(rounding) - {"type", "zero_sign"} or rounding.get("type") != "sign_means":
            raise ValueError("Only rounding.type='sign_means' is implemented.")
        rounding.setdefault("zero_sign", 1)
        if rounding["zero_sign"] not in (-1, 1) or isinstance(rounding["zero_sign"], bool):
            raise ValueError("rounding.zero_sign must be +1 or -1.")

        stopping = config.setdefault("stopping", {"type": "max_sweeps"})
        if set(stopping) - {"type"} or stopping.get("type") != "max_sweeps":
            raise ValueError("Only stopping.type='max_sweeps' is implemented.")

        limits = config.setdefault("limits", {})
        if set(limits) - {"shots_total", "max_block_updates_total"}:
            raise ValueError("Unknown block mean-field limit.")
        limits.setdefault("shots_total", None)
        limits.setdefault("max_block_updates_total", None)
        if limits["shots_total"] is not None:
            positive_int(limits["shots_total"], "shots_total", minimum=0)
        if limits["max_block_updates_total"] is not None:
            positive_int(limits["max_block_updates_total"], "max_block_updates_total", minimum=0)

        execution = config.setdefault(
            "execution", {"backend": "process", "start_method": "spawn", "native_threads": 1}
        )
        if set(execution) - {"backend", "start_method", "native_threads"}:
            raise ValueError("Unknown execution field.")
        execution.setdefault("backend", "process")
        execution.setdefault("start_method", "spawn")
        execution.setdefault("native_threads", 1)
        if execution["backend"] != "process" or execution["start_method"] != "spawn":
            raise ValueError("Only process/spawn execution is implemented.")
        positive_int(execution["native_threads"], "native_threads")

        policy = config.setdefault("budget_policy", {"type": "sweep_round_robin"})
        if set(policy) - {"type"} or policy.get("type") != "sweep_round_robin":
            raise ValueError("Only sweep_round_robin is implemented.")

        collect = config.setdefault("collect", {})
        allowed_collect = {
            "message_history",
            "model_history",
            "parameter_trajectory",
            "local_optimizer_history",
            "raw_candidate_probabilities",
        }
        if set(collect) - allowed_collect:
            raise ValueError(f"Unknown block collect field: {set(collect) - allowed_collect}")
        for key in allowed_collect:
            collect.setdefault(key, False)
            if type(collect[key]) is not bool:
                raise ValueError(f"collect.{key} must be boolean.")
        # Raw candidate probabilities are optional and remain tied to the
        # proposal's block/input/angle metadata.

    @staticmethod
    def _validate_partition(graph, blocks, max_size):
        if not isinstance(blocks, list):
            raise ValueError("partition.parameters.blocks must be a list.")
        seen = set()
        parsed = []
        if not graph.nodes:
            if blocks:
                raise ValueError("An empty graph requires an empty partition.")
            return
        if not blocks:
            raise ValueError("A nonempty graph requires a nonempty partition.")
        for block in blocks:
            if not isinstance(block, list) or not block:
                raise ValueError("Each block must be a nonempty ordered list.")
            if len(block) > max_size:
                raise ValueError("A block exceeds max_block_vertices.")
            ordered = []
            for node in block:
                if isinstance(node, bool) or not isinstance(node, int):
                    raise ValueError("Block labels must be integers.")
                if node not in graph.nodes:
                    raise ValueError(f"Block label is not in the graph: {node}")
                if node in seen or node in ordered:
                    raise ValueError(f"Partition overlap or duplicate label: {node}")
                seen.add(node)
                ordered.append(node)
            parsed.append(tuple(ordered))
        if seen != set(graph.nodes):
            raise ValueError("Partition must cover every graph node exactly once.")
        return parsed

    @staticmethod
    def _validate_means_entries(graph, entries):
        if not isinstance(entries, list):
            raise ValueError("entries must be a list.")
        seen = set()
        for item in entries:
            if not isinstance(item, dict) or set(item) != {"node", "mean"}:
                raise ValueError("Each mean entry needs exactly node and mean.")
            node = item["node"]
            value = item["mean"]
            if isinstance(node, bool) or not isinstance(node, int) or node not in graph.nodes:
                raise ValueError("Mean nodes must be graph integer labels.")
            if node in seen:
                raise ValueError(f"Duplicate mean node: {node}")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not -1 <= float(value) <= 1:
                raise ValueError("Mean values must be finite in [-1, 1].")
            seen.add(node)
        if seen != set(graph.nodes):
            raise ValueError("provided_means must cover every graph node.")

    @staticmethod
    def _validate_spin_entries(graph, entries):
        if not isinstance(entries, list):
            raise ValueError("entries must be a list.")
        seen = set()
        for item in entries:
            if not isinstance(item, dict) or set(item) != {"node", "spin"}:
                raise ValueError("Each spin entry needs exactly node and spin.")
            node = item["node"]
            spin = item["spin"]
            if isinstance(node, bool) or not isinstance(node, int) or node not in graph.nodes:
                raise ValueError("Spin nodes must be graph integer labels.")
            if node in seen:
                raise ValueError(f"Duplicate spin node: {node}")
            if spin not in (-1, 1) or isinstance(spin, bool):
                raise ValueError("Spin values must be +1 or -1.")
            seen.add(node)
        if seen != set(graph.nodes):
            raise ValueError("provided_spins must cover every graph node.")

    def _initial_means(self, graph, config):
        messages = config["initial_messages"]
        kind = messages["type"]
        params = messages["parameters"]
        if kind == "zero":
            return {int(node): 0.0 for node in graph.nodes}
        if kind == "provided_means":
            return {int(item["node"]): float(item["mean"]) for item in params["entries"]}
        amplitude = float(params["amplitude"])
        source = params["spins"]
        if source["type"] == "all_plus":
            spins = {int(node): 1 for node in graph.nodes}
        else:
            spins = {
                int(item["node"]): int(item["spin"])
                for item in source["parameters"]["entries"]
            }
        return {int(node): amplitude * float(spins[node]) for node in graph.nodes}

    def _initial_thetas(self, graph, config, rng):
        layers = int(config["parameters"]["layers"])
        initialization = config["initialization"]
        blocks = config["partition"]["parameters"]["blocks"]
        if initialization.get("type") == "angle_bank":
            return [
                np.asarray(item["values"], dtype=float).copy()
                for item in initialization["parameters"]["entries"]
            ], "angle_bank"
        values = initialization.get("parameters", {}).get("values")
        if (
            initialization.get("type") == "fixed_angles"
            and isinstance(values, list)
            and values
            and isinstance(values[0], (list, tuple))
        ):
            return [np.asarray(row, dtype=float).copy() for row in values], "fixed_angles"
        values, source = _initial_angles(config, layers)
        if values is not None:
            return [np.asarray(values, dtype=float).copy() for _ in blocks], source
        return [
            np.r_[
                rng.uniform(-np.pi, np.pi, layers),
                rng.uniform(-np.pi / 2, np.pi / 2, layers),
            ]
            for _ in blocks
        ], source

    @staticmethod
    def _moment_counts(config):
        moments = config["moments"]
        if moments["mode"] == "shots":
            return int(moments["shots"])
        return 0

    @staticmethod
    def _training_cost(config):
        layers = int(config["parameters"]["layers"])
        estimation = config["local_solver"]["estimation"]
        if estimation["mode"] != "shots":
            return 0
        return (4 * layers + 1) * int(estimation["shots_per_evaluation"])

    def _plan_sweep(self, budget, config, block_count):
        required = block_count * self._moment_counts(config)
        available = budget.available()
        max_steps = int(config["local_solver"]["optimizer"]["parameters"]["max_steps"])
        if available is not None and available < required:
            return None, "moment_reserve_unavailable", [], required
        cost = self._training_cost(config)
        allocations = [0] * block_count
        if cost == 0:
            return [max_steps] * block_count, None, [required] * block_count, required
        extra = None if available is None else available - required
        if extra is None:
            return [max_steps] * block_count, None, [
                required + max_steps * cost for _ in range(block_count)
            ], required
        count = min(extra // cost, block_count * max_steps)
        for index in range(count):
            allocations[index % block_count] += 1
        quotas = [required + allocation * cost for allocation in allocations]
        return allocations, None, quotas, required

    def _sample_moments(self, problem, theta, config, rng, *, shots):
        layers = int(config["parameters"]["layers"])
        dtype = config["parameters"]["dtype"]
        state, probabilities = state_at_angles(
            problem.affected_cuts, theta, layers, dtype=dtype
        )
        norm = float(np.linalg.norm(state))
        if not math.isfinite(norm) or abs(norm - 1.0) > 1e-8:
            raise MeanFieldError("The local quantum state is not normalized.")
        nodes = problem.nodes
        n = len(nodes)
        size = 1 << n
        if shots:
            indices = rng.choice(size, size=shots, p=probabilities).astype(np.uint64)
            shifts = np.arange(n, dtype=np.uint64)
            bits = ((indices[:, None] >> shifts[None, :]) & np.uint64(1)).astype(float)
            spins = 1.0 - 2.0 * bits
            z = {int(node): float(spins[:, i].mean()) for i, node in enumerate(nodes)}
            q = {}
            for u, v, _ in problem.internal_edges:
                i = nodes.index(u)
                j = nodes.index(v)
                q[canonical_edge(u, v)] = float((spins[:, i] * spins[:, j]).mean())
            origin = "empirical"
        else:
            indices = np.arange(size, dtype=np.uint64)
            shifts = np.arange(n, dtype=np.uint64)
            bits = ((indices[:, None] >> shifts[None, :]) & np.uint64(1)).astype(float)
            spins = 1.0 - 2.0 * bits
            z = {
                int(node): float(probabilities @ spins[:, i])
                for i, node in enumerate(nodes)
            }
            q = {}
            for u, v, _ in problem.internal_edges:
                i = nodes.index(u)
                j = nodes.index(v)
                q[canonical_edge(u, v)] = float(
                    probabilities @ (spins[:, i] * spins[:, j])
                )
            origin = "exact"
        return z, q, probabilities, origin

    def _update_block(self, block_id, nodes, problems, means, q, theta, config, budget, rng, *, reserved_moments, effective_steps):
        problem = problems[block_id]
        old_theta = np.asarray(theta[block_id], dtype=float).copy()
        optimizer = deepcopy(config["local_solver"]["optimizer"])
        optimizer["parameters"]["max_steps"] = int(effective_steps)
        training = train_statevector(
            problem.affected_cuts,
            int(config["parameters"]["layers"]),
            objective=config["local_solver"]["objective"],
            estimation_mode=config["local_solver"]["estimation"]["mode"],
            shots_per_evaluation=int(config["local_solver"]["estimation"]["shots_per_evaluation"]),
            optimizer=optimizer,
            initial_angles=old_theta,
            dtype=config["parameters"]["dtype"],
            rng=rng,
            budget=budget,
            reserved_shots=int(reserved_moments),
            collect_optimizer_history=bool(config["collect"]["local_optimizer_history"]),
            collect_objective_history=bool(config["collect"]["parameter_trajectory"]),
            score_cuts=problem.affected_cuts,
        )
        candidate = np.asarray(training["angles"], dtype=float).copy()
        kappa = float(config["updates"]["angle_damping"])
        applied = (1.0 - kappa) * old_theta + kappa * candidate
        shots = self._moment_counts(config)
        z, raw_q, probabilities, origin = self._sample_moments(
            problem, applied, config, rng, shots=shots
        )
        if shots:
            budget.spend(shots)
        # Raw moments are from the applied angles, not from the optimizer proposal.
        old_m = {int(node): float(means[node]) for node in nodes}
        old_q = {
            canonical_edge(u, v): float(q[canonical_edge(u, v)])
            for u, v, _ in problem.internal_edges
        }
        new_m, new_q = damped_moments(
            old_m,
            old_q,
            z,
            raw_q,
            float(config["updates"]["moment_damping"]),
        )
        for edge, value in new_q.items():
            probabilities_pair = pair_probabilities(
                new_m[edge[0]], new_m[edge[1]], value
            )
            if min(probabilities_pair.values()) < -1e-9:
                raise MeanFieldError("Damped pair moments are not physically consistent.")
        return {
            "block_id": int(block_id),
            "nodes": [int(node) for node in nodes],
            "old_theta": old_theta.tolist(),
            "candidate_theta": candidate.tolist(),
            "applied_theta": applied.tolist(),
            "old_means": old_m,
            "raw_means": z,
            "new_means": new_m,
            "old_q": old_q,
            "raw_q": raw_q,
            "new_q": new_q,
            "moment_origin": origin,
            "moment_shots": shots,
            "moment_probabilities": (
                probabilities.tolist()
                if config["collect"]["raw_candidate_probabilities"]
                else None
            ),
            "training": {
                "reason": training["reason"],
                "evaluations": int(training["evaluations"]),
                "shots": int(training["shots"]),
                "optimizer_steps": int(training["optimizer_steps"]),
                "best_loss": training["best_loss"],
                "objective_value": training["objective_value"],
                "optimizer_history": training["optimizer_history"],
                "objective_history": training["objective_history"],
            },
            "field_fingerprint": problem.field_fingerprint,
            "field_problem": problem.to_dict(),
            "effective_steps": int(effective_steps),
            "reserved_moments": int(reserved_moments),
        }

    @staticmethod
    def _apply_proposal(means, q, theta, proposal):
        for node in proposal["nodes"]:
            means[int(node)] = float(proposal["new_means"][int(node)])
        for edge, value in proposal["new_q"].items():
            q[tuple(edge)] = float(value)
        theta[int(proposal["block_id"])] = np.asarray(
            proposal["applied_theta"], dtype=float
        )

    def _model_cut(self, graph, blocks, means, q):
        block_of = {}
        for index, block in enumerate(blocks):
            for node in block:
                block_of[int(node)] = index
        total = 0.0
        for u, v, weight in graph.edges:
            y = float(weight)
            if block_of[int(u)] == block_of[int(v)]:
                value = q.get(canonical_edge(u, v), means[int(u)] * means[int(v)])
            else:
                value = means[int(u)] * means[int(v)]
            total += 0.5 * y * (1.0 - value)
        return float(total)

    @staticmethod
    def _rounding(means, zero_sign):
        return {
            int(node): 1 if float(mean) > 0 else (-1 if float(mean) < 0 else int(zero_sign))
            for node, mean in means.items()
        }

    def _result(
        self,
        graph,
        config,
        blocks,
        means,
        q,
        theta,
        *,
        status,
        reason,
        budget,
        completed_sweeps,
        accepted_updates,
        local_jobs,
        history,
        model_history,
        initial_means,
        initial_thetas,
        partial_sweep,
        pending_batch,
    ):
        solution = self._rounding(means, int(config["rounding"]["zero_sign"]))
        model_cut = self._model_cut(graph, blocks, means, q)
        return {
            "solution": solution,
            "status": status,
            "termination_reason": reason,
            "means": {str(node): float(value) for node, value in means.items()},
            "edge_moments": [
                [int(u), int(v), float(q[canonical_edge(u, v)])]
                for block in blocks
                for u, v, _ in graph.edges
                if u in block and v in block
            ],
            "model_mean_cut": float(model_cut),
            "model_energy": float(-model_cut),
            "model_origin": "exact" if config["moments"]["mode"] == "exact" else "empirical",
            "partition": [[int(node) for node in block] for block in blocks],
            "schedule": config["schedule"]["type"],
            "moment_damping": float(config["updates"]["moment_damping"]),
            "angle_damping": float(config["updates"]["angle_damping"]),
            "rounding": deepcopy(config["rounding"]),
            "completed_sweeps": int(completed_sweeps),
            "accepted_updates": int(accepted_updates),
            "completed_local_jobs": int(local_jobs),
            "partial_sweep": bool(partial_sweep),
            "pending_batch": bool(pending_batch),
            "initial_means": {str(node): float(value) for node, value in initial_means.items()},
            "initial_thetas": [np.asarray(value, dtype=float).tolist() for value in initial_thetas],
            "theta_bank": [np.asarray(value, dtype=float).tolist() for value in theta],
            "message_history": history if config["collect"]["message_history"] else None,
            "model_history": model_history if config["collect"]["model_history"] else None,
            "resources": {
                "shots_used": budget.used if budget is not None else 0,
                "training_shots": int(sum(row["proposal"]["training"]["shots"] for row in history)),
                "moment_shots": int(sum(row["proposal"]["moment_shots"] for row in history)),
                "unknown_shots": 0,
                "completed_local_jobs": int(local_jobs),
                "num_treads": int(config.get("num_treads", 1)),
                "num_treads_requested": int(config.get("num_treads", 1)),
                "num_treads_applied": 1,
                "backend": config["execution"]["backend"],
                "native_threads": int(config["execution"]["native_threads"]),
                "budget": budget.snapshot() if budget is not None else None,
            },
        }

    def _checkpoint_state(self, graph, config, blocks, means, q, theta, budget, rng, *, next_sweep, completed_sweeps, accepted_updates, local_jobs, history, model_history, initial_means, initial_thetas, phase):
        """Return one JSON-compatible checkpoint state."""
        import numpy as _np

        return {
            "config": deepcopy(config),
            "graph": graph.snapshot(),
            "rng_state": deepcopy(rng.bit_generator.state),
            "means": {str(node): float(value) for node, value in means.items()},
            "q": [[int(u), int(v), float(value)] for (u, v), value in q.items()],
            "theta": [_np.asarray(value, dtype=float).tolist() for value in theta],
            "budget": None if budget is None else budget.snapshot(),
            "next_sweep": int(next_sweep),
            "completed_sweeps": int(completed_sweeps),
            "accepted_updates": int(accepted_updates),
            "local_jobs": int(local_jobs),
            "history": deepcopy(history),
            "model_history": deepcopy(model_history),
            "initial_means": {str(node): float(value) for node, value in initial_means.items()},
            "initial_thetas": [_np.asarray(value, dtype=float).tolist() for value in initial_thetas],
            "phase": str(phase),
        }

    def _restore_checkpoint(self, graph, config):
        """Restore a compact block model from ``restore`` state."""
        saved = self._resume_state
        if not isinstance(saved, dict):
            return None
        if saved.get("graph", {}).get("fingerprint") != graph.snapshot().get("fingerprint"):
            raise ValueError("Checkpoint graph does not match the restored graph.")
        means = {int(k): float(v) for k, v in saved["means"].items()}
        q = {(int(item[0]), int(item[1])): float(item[2]) for item in saved["q"]}
        theta = [np.asarray(value, dtype=float) for value in saved["theta"]]
        budget_data = saved.get("budget") or {"total": None, "used": 0}
        budget = ShotBudget(budget_data.get("total"))
        if int(budget_data.get("used", 0)):
            budget.spend(int(budget_data["used"]))
        self._checkpoint_next_sweep = int(saved.get("next_sweep", 0))
        return {
            "means": means,
            "q": q,
            "theta": theta,
            "budget": budget,
            "completed_sweeps": int(saved.get("completed_sweeps", 0)),
            "accepted_updates": int(saved.get("accepted_updates", 0)),
            "local_jobs": int(saved.get("local_jobs", 0)),
            "history": deepcopy(saved.get("history") or []),
            "model_history": deepcopy(saved.get("model_history") or []),
            "initial_means": {int(k): float(v) for k, v in saved.get("initial_means", {}).items()},
            "initial_thetas": [np.asarray(value, dtype=float) for value in saved.get("initial_thetas", [])],
        }

    def resume(self, *, control=None):
        """Continue a restored checkpoint without starting a new run."""
        if self._resume_state is None:
            raise RuntimeError("resume() requires a solver returned by restore().")
        if self._resume_graph is None:
            raise RuntimeError("The restored checkpoint has no graph.")
        self._resume_mode = True
        return self.solve(self._resume_graph, control=control)

    def _run(self, graph, config, rng):
        blocks = [tuple(int(node) for node in block) for block in config["partition"]["parameters"]["blocks"]]
        layers = int(config["parameters"]["layers"])
        if not graph.nodes:
            return self._result(
                graph,
                config,
                [],
                {},
                {},
                [],
                status="completed",
                reason="empty_graph",
                budget=ShotBudget(0),
                completed_sweeps=0,
                accepted_updates=0,
                local_jobs=0,
                history=[],
                model_history=[],
                initial_means={},
                initial_thetas=[],
                partial_sweep=False,
                pending_batch=False,
            )
        checkpoint_config = config.get("checkpoint", {})
        manager = None
        if checkpoint_config.get("enabled", False):
            manager = CheckpointManager(
                checkpoint_config,
                run_id=self._run_id,
                component_id=self.algorithm_type,
                graph=graph,
                config_signature=stable_hash(
                    {"config": config, "graph": graph.snapshot()}
                ),
            )
            self._checkpoint_manager = manager
        restored = self._restore_checkpoint(graph, config) if self._resume_state else None
        if restored is not None:
            means = restored["means"]
            q = restored["q"]
            theta = restored["theta"]
            budget = restored["budget"]
            completed_sweeps = restored["completed_sweeps"]
            accepted_updates = restored["accepted_updates"]
            local_jobs = restored["local_jobs"]
            history = restored["history"]
            model_history = restored["model_history"]
            initial_means = restored["initial_means"]
            initial_thetas = restored["initial_thetas"]
            next_sweep = int(getattr(self, "_checkpoint_next_sweep", 0))
        else:
            initial_means = self._initial_means(graph, config)
            means = dict(initial_means)
            q = {}
            for block in blocks:
                active = set(block)
                for u, v, _ in graph.edges:
                    if u in active and v in active:
                        q[canonical_edge(u, v)] = means[int(u)] * means[int(v)]
            theta, angle_source = self._initial_thetas(graph, config, rng)
            initial_thetas = [value.copy() for value in theta]
            budget = ShotBudget(config["limits"]["shots_total"])
            history = []
            model_history = []
            completed_sweeps = 0
            accepted_updates = 0
            local_jobs = 0
            next_sweep = 0
        partial_sweep = False
        pending_batch = False
        status = "completed"
        reason = "max_sweeps"
        if config["parameters"]["max_sweeps"] == 0:
            reason = "zero_sweeps"

        def publish(phase, sweep_number):
            if manager is not None:
                self.last_checkpoint = manager.write_generation(
                    self._checkpoint_state(
                        graph,
                        config,
                        blocks,
                        means,
                        q,
                        theta,
                        budget,
                        rng,
                        next_sweep=int(sweep_number),
                        completed_sweeps=completed_sweeps,
                        accepted_updates=accepted_updates,
                        local_jobs=local_jobs,
                        history=history,
                        model_history=model_history,
                        initial_means=initial_means,
                        initial_thetas=initial_thetas,
                        phase=phase,
                    ),
                    phase=phase,
                    next_operation={"sweep": int(sweep_number)},
                )

        publish("initialized", next_sweep)
        for sweep in range(next_sweep, int(config["parameters"]["max_sweeps"])):
            if self._control is not None and self._control.should_pause(
                "sweeps", completed_sweeps
            ):
                status = "paused"
                reason = "checkpoint_pause"
                publish("paused", next_sweep)
                break
            if config["limits"]["max_block_updates_total"] is not None:
                remaining = int(config["limits"]["max_block_updates_total"]) - local_jobs
                if remaining < len(blocks):
                    status = "stopped"
                    reason = "block_update_limit"
                    break
            allocations, plan_error, quotas, required_moments = self._plan_sweep(
                budget, config, len(blocks)
            )
            if plan_error is not None:
                status = "stopped"
                reason = plan_error
                break
            reserved_remaining = required_moments
            scheme = config["schedule"]["type"]
            if scheme == "gauss_seidel":
                sweep_complete = True
                for block_id, nodes in enumerate(blocks):
                    if config["limits"]["max_block_updates_total"] is not None and local_jobs >= int(config["limits"]["max_block_updates_total"]):
                        status = "stopped"
                        reason = "block_update_limit"
                        sweep_complete = False
                        partial_sweep = accepted_updates > 0
                        break
                    active = set(nodes)
                    problems = {
                        block_id: build_block_problem(
                            graph,
                            nodes,
                            means,
                            block_id=block_id,
                        )
                    }
                    try:
                        proposal = self._update_block(
                            block_id,
                            nodes,
                            problems,
                            means,
                            q,
                            theta,
                            config,
                            budget,
                            rng,
                            reserved_moments=reserved_remaining,
                            effective_steps=allocations[block_id],
                        )
                    except (ShotBudgetExceeded, MeanFieldError):
                        status = "stopped"
                        reason = "shot_budget"
                        sweep_complete = False
                        partial_sweep = accepted_updates > 0
                        break
                    local_jobs += 1
                    self._apply_proposal(means, q, theta, proposal)
                    accepted_updates += 1
                    reserved_remaining = max(0, reserved_remaining - proposal["moment_shots"])
                    history.append(
                        {
                            "sweep": int(sweep),
                            "block_id": int(block_id),
                            "schedule": scheme,
                            "model_version_before": int(sweep),
                            "model_version_after": int(sweep),
                            "proposal": proposal,
                        }
                    )
                    model_history.append(
                        {
                            "sweep": int(sweep),
                            "block_id": int(block_id),
                            "means": {str(node): float(value) for node, value in means.items()},
                            "model_mean_cut": self._model_cut(graph, blocks, means, q),
                        }
                    )
                if not sweep_complete:
                    break
                completed_sweeps += 1
                next_sweep = sweep + 1
                publish("gauss_seidel_sweep_commit", next_sweep)
            else:
                snapshot_means = dict(means)
                snapshot_q = dict(q)
                snapshot_theta = [value.copy() for value in theta]
                problems = {
                    block_id: build_block_problem(
                        graph, nodes, snapshot_means, block_id=block_id
                    )
                    for block_id, nodes in enumerate(blocks)
                }
                proposals = []
                sweep_complete = True
                for block_id, nodes in enumerate(blocks):
                    if config["limits"]["max_block_updates_total"] is not None and local_jobs >= int(config["limits"]["max_block_updates_total"]):
                        status = "stopped"
                        reason = "block_update_limit"
                        sweep_complete = False
                        partial_sweep = len(proposals) > 0
                        pending_batch = True
                        break
                    try:
                        proposal = self._update_block(
                            block_id,
                            nodes,
                            problems,
                            snapshot_means,
                            snapshot_q,
                            snapshot_theta,
                            config,
                            budget,
                            rng,
                            reserved_moments=reserved_remaining,
                            effective_steps=allocations[block_id],
                        )
                    except (ShotBudgetExceeded, MeanFieldError):
                        status = "stopped"
                        reason = "shot_budget"
                        sweep_complete = False
                        partial_sweep = len(proposals) > 0
                        pending_batch = True
                        break
                    local_jobs += 1
                    proposals.append(proposal)
                    reserved_remaining = max(0, reserved_remaining - proposal["moment_shots"])
                if not sweep_complete:
                    break
                # Jacobi publishes one complete coordinated bank at a barrier.
                for proposal in proposals:
                    self._apply_proposal(means, q, theta, proposal)
                    accepted_updates += 1
                    history.append(
                        {
                            "sweep": int(sweep),
                            "block_id": int(proposal["block_id"]),
                            "schedule": scheme,
                            "model_version_before": int(sweep),
                            "model_version_after": int(sweep) + 1,
                            "proposal": proposal,
                        }
                    )
                completed_sweeps += 1
                next_sweep = sweep + 1
                model_history.append(
                    {
                        "sweep": int(sweep),
                        "block_id": None,
                        "means": {str(node): float(value) for node, value in means.items()},
                        "model_mean_cut": self._model_cut(graph, blocks, means, q),
                    }
                )
                publish("jacobi_sweep_commit", next_sweep)
        return self._result(
            graph,
            config,
            blocks,
            means,
            q,
            theta,
            status=status,
            reason=reason,
            budget=budget,
            completed_sweeps=completed_sweeps,
            accepted_updates=accepted_updates,
            local_jobs=local_jobs,
            history=history,
            model_history=model_history,
            initial_means=initial_means,
            initial_thetas=initial_thetas,
            partial_sweep=partial_sweep,
            pending_batch=pending_batch,
        )


def main(argv=None):
    """Run a block mean-field example through the common CLI."""
    from ..tools.demo import run_solver_demo

    return run_solver_demo(BlockMeanFieldQAOACPU, argv)


if __name__ == "__main__":
    raise SystemExit(main())
