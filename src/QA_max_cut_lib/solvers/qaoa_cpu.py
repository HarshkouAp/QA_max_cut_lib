"""Simulate full-state QAOA with explicit bit order and signed energy."""

from copy import deepcopy
import numpy as np
from .base import BaseSolver, positive_int
from .objectives import validate_objective
from .qaoa_state import cut_diagonal, state_at_angles
from .qaoa_training import train_statevector
from ..core.budget import ShotBudget
from ..core.checkpoint import CheckpointManager, CheckpointError
from ..core.experiment_ids import stable_hash


class QAOACPU(BaseSolver):
    """Train a bounded full-state QAOA using central differences and ADAM.

    Parameters
    ----------
    config : dict, optional
        JSON-compatible settings. See docs/API.md for all nested fields.
        This implementation supports the five distribution objectives,
        exact or sampled training, and two final readout modes.

    Attributes
    ----------
    optimal_angles : ndarray, shape (2*p,)
        Best evaluated training angles in gamma-then-beta order.
    optimizer_history : list of dict or None
        Accepted optimization-step observations, when collected.
    objective_history : list of dict or None
        All evaluated points, including gradient probes.
    statevector : ndarray or None
        Final full state, when collected.
    probabilities : ndarray or None
        Final full probability vector, when collected.
    correlations : ndarray or None
        Physical <Z_i Z_j>, indexed by node_order, when collected.

    Notes
    -----
    The state uses O(2**N) memory. The original 1000-node target applies
    to future composite solvers, not this direct simulator.
    Training minimizes minus the selected cut-distribution criterion.
    A sampled objective is not an analytic gradient estimate: central
    differences use separate samples for each probe in this version.
    """

    algorithm_type = "qaoa_cpu"
    checkpoint_component_id = "qaoa_cpu"
    supports_checkpoint = True
    optional_fields = BaseSolver.optional_fields + (
        "checkpoint_info",
        "readout_completed",
        "resume_mode",
        "best_evaluation_id",
    )

    def _validate(self, config, graph):
        check_cfg = config.get("checkpoint", {})
        if check_cfg.get("enabled", False):
            events = set(check_cfg.get("every", {}))
            if events - {"optimizer_iterations"}:
                raise ValueError(
                    f"QAOACPU does not support checkpoint events {sorted(events - {'optimizer_iterations'})}."
                )
        p = config.setdefault("parameters", {})
        if set(p) - {"layers", "max_vertices", "dtype"}:
            raise ValueError("Unknown QAOA parameter.")
        p.setdefault("layers", 1)
        p.setdefault("max_vertices", 20)
        p.setdefault("dtype", "complex128")
        positive_int(p["layers"], "layers")
        positive_int(p["max_vertices"], "max_vertices")
        if not 1 <= len(graph) <= min(p["max_vertices"], 24):
            raise ValueError(
                "Direct QAOA requires 1..max_vertices nodes, with a hard cap of 24."
            )
        if p["dtype"] not in ("complex64", "complex128"):
            raise ValueError("dtype must be complex64 or complex128.")
        config["objective"] = validate_objective(config.get("objective", {}))
        estimation = config.setdefault("estimation", {})
        if set(estimation) - {"mode", "shots_per_evaluation"}:
            raise ValueError("Unknown estimation field.")
        estimation.setdefault("mode", "exact")
        estimation.setdefault("shots_per_evaluation", 256)
        if estimation["mode"] not in ("exact", "shots"):
            raise ValueError("estimation.mode must be exact or shots.")
        positive_int(estimation["shots_per_evaluation"], "shots_per_evaluation")
        optimizer = config.setdefault("optimizer", {})
        if (
            set(optimizer) - {"type", "parameters", "gradient"}
            or optimizer.get("type", "adam") != "adam"
        ):
            raise ValueError(
                "Only the documented ADAM optimizer is implemented in this milestone."
            )
        optimizer["type"] = "adam"
        op = optimizer.setdefault("parameters", {})
        if set(op) - {
            "max_steps",
            "learning_rate",
            "beta1",
            "beta2",
            "epsilon",
            "gradient_tolerance",
        }:
            raise ValueError("Unknown ADAM parameter.")
        defaults = dict(
            max_steps=20,
            learning_rate=0.05,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            gradient_tolerance=0.0,
        )
        for key, value in defaults.items():
            op.setdefault(key, value)
        positive_int(op["max_steps"], "max_steps", minimum=0)
        for key in ["learning_rate", "epsilon"]:
            if not np.isfinite(op[key]) or op[key] <= 0:
                raise ValueError(f"Invalid {key}.")
        for key in ["beta1", "beta2"]:
            if not np.isfinite(op[key]) or not 0 <= op[key] < 1:
                raise ValueError(f"Invalid {key}.")
        if not np.isfinite(op["gradient_tolerance"]) or op["gradient_tolerance"] < 0:
            raise ValueError("Invalid gradient_tolerance.")
        gradient = optimizer.setdefault(
            "gradient", {"type": "central_difference", "parameters": {"step": 0.01}}
        )
        if gradient.get("type") != "central_difference" or set(gradient) - {
            "type",
            "parameters",
        }:
            raise ValueError("Only central_difference is implemented.")
        gp = gradient.setdefault("parameters", {})
        gp.setdefault("step", 0.01)
        if set(gp) != {"step"} or not np.isfinite(gp["step"]) or gp["step"] <= 0:
            raise ValueError("The difference step must be finite and positive.")
        init = config.setdefault("initialization", {})
        if set(init) - {"type", "parameters"}:
            raise ValueError("Unknown initialization field.")
        init.setdefault("type", "fixed_angles")
        ip = init.setdefault("parameters", {})
        if init["type"] == "fixed_angles":
            ip.setdefault("values", [0.5] * p["layers"] + [0.3] * p["layers"])
            if (
                set(ip) != {"values"}
                or np.shape(ip["values"]) != (2 * p["layers"],)
                or not np.isfinite(ip["values"]).all()
            ):
                raise ValueError("Supply exactly 2*layers finite angles.")
        elif init["type"] == "random_angles":
            if ip:
                raise ValueError("random_angles has no parameters in this milestone.")
        else:
            raise ValueError("Unknown initialization type.")
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
            raise ValueError("Unknown limit field.")
        limits.setdefault("shots_total", None)
        if limits["shots_total"] is not None:
            positive_int(limits["shots_total"], "shots_total", minimum=0)
            if readout.get("shots", 0) > limits["shots_total"]:
                raise ValueError("The total shot budget cannot cover final readout.")
        collect = config.setdefault("collect", {})
        if set(collect) - {
            "optimizer_history",
            "objective_history",
            "probabilities",
            "statevector",
            "correlations",
        }:
            raise ValueError("Unknown collect field.")
        for key in [
            "optimizer_history",
            "objective_history",
            "probabilities",
            "statevector",
            "correlations",
        ]:
            collect.setdefault(key, False)
            if type(collect[key]) is not bool:
                raise ValueError(f"collect.{key} must be boolean.")

    def _state_at_angles(self, cuts, angles, layers, dtype):
        """Return the state and probabilities for the active ansatz."""
        return state_at_angles(cuts, angles, layers, dtype)

    def _checkpoint_extra(self):
        """Return component-specific durable checkpoint fields."""
        return {}

    def _run(self, graph, config, rng):
        from ..tools.reporting import log_message

        p = config["parameters"]
        layers = p["layers"]
        cuts = cut_diagonal(graph)
        resume_state = deepcopy(self._resume_state) if self._resume_mode and self._resume_state else None
        if resume_state is not None and self._resume_arrays:
            for name, array in self._resume_arrays.items():
                resume_state[name] = np.asarray(array).tolist()
        initial_angles = np.array(
            config["initialization"]["parameters"].get("values", [0.5] * layers + [0.3] * layers),
            dtype=float,
        )
        if config["initialization"]["type"] != "fixed_angles":
            initial_angles = np.r_[
                rng.uniform(-np.pi, np.pi, layers),
                rng.uniform(-np.pi / 2, np.pi / 2, layers),
            ]
        budget = ShotBudget(config["limits"]["shots_total"])
        used_before = 0 if resume_state is None else int(resume_state.get("budget_used", 0))
        if used_before:
            budget.spend(used_before)
        reserved = int(config["readout"].get("shots", 0))
        manager = None
        checkpoint_config = config.get("checkpoint", {})
        recovery_charge = 0
        recovery_ledger = None
        if checkpoint_config.get("enabled", False):
            manager = CheckpointManager(
                checkpoint_config,
                run_id=self._run_id,
                component_id=self.checkpoint_component_id,
                graph=graph,
                config_signature=stable_hash({"config": config, "graph": graph.snapshot()}),
            )
            manager.acquire()
            self._checkpoint_manager = manager
            if self._resume_mode and manager.journal is not None:
                recovery_ledger = manager.journal.reconcile()
                recovery_charge = int(recovery_ledger.get("uncertain_upper_charge", 0))
                if recovery_charge:
                    budget.spend(recovery_charge)
                    for operation_id in recovery_ledger.get("open_operation_ids", []):
                        manager.journal.resolve(
                            operation_id,
                            charge=recovery_charge
                            if operation_id == recovery_ledger["open_operation_ids"][0]
                            else 0,
                            reason="recovery_conservative_charge",
                        )

        def save_checkpoint(state_record, kind):
            if manager is None or not manager.enabled:
                return
            every = checkpoint_config.get("every", {})
            if kind == "stage_begin":
                if manager.journal is not None:
                    iteration = int(state_record.get("next_iteration", 0))
                    needed = (2 * 2 * layers + 1) * int(config["estimation"].get("shots_per_evaluation", 0)) if config["estimation"]["mode"] == "shots" else 0
                    manager.journal.begin(
                        f"{self._run_id}:iteration:{iteration}",
                        attempt_id=f"attempt:{iteration}:0",
                        upper_bound=needed,
                        kind="optimizer_iteration",
                    )
                return
            if kind == "stage_commit":
                if manager.journal is not None:
                    iteration = int(state_record.get("next_iteration", 0)) - 1
                    needed = (2 * 2 * layers + 1) * int(config["estimation"].get("shots_per_evaluation", 0)) if config["estimation"]["mode"] == "shots" else 0
                    manager.journal.commit(
                        f"{self._run_id}:iteration:{iteration}",
                        used=needed,
                    )
                return
            if kind == "periodic" and "optimizer_iterations" not in every:
                return
            if kind == "phase" and not checkpoint_config.get("at_phase_boundaries", True):
                return
            if kind == "pause" and not checkpoint_config.get("on_pause", True):
                return
            payload = deepcopy(state_record)
            arrays = {}
            for name in ("theta", "moment", "variance", "best_theta"):
                if name in payload:
                    arrays[name] = np.asarray(payload.pop(name), dtype=float)
            payload.update(
                {
                    "component_id": self.checkpoint_component_id,
                    "config": deepcopy(config),
                    "graph": graph.snapshot(),
                    "budget_total": config["limits"]["shots_total"],
                    "budget_used": int(budget.used),
                    "readout_reserved": reserved,
                    "control": self._control.snapshot() if self._control is not None else None,
                    "checkpoint_policy": deepcopy(checkpoint_config),
                    "recovery_charge": recovery_charge,
                    "recovery_ledger": recovery_ledger,
                }
            )
            payload.update(self._checkpoint_extra())
            ref = manager.write_generation(
                payload,
                arrays,
                parent=None if manager.last_ref is None else manager.last_ref.generation,
                phase=payload.get("phase"),
                next_operation={"optimizer_iteration": payload.get("next_iteration")},
            )
            self.last_checkpoint = ref
            self.checkpoint_info = ref.as_dict()

        if resume_state is not None and resume_state.get("phase") == "readout_done":
            best_theta = np.asarray(resume_state["best_theta"], dtype=float)
            probabilities = self._state_at_angles(cuts, best_theta, layers, p["dtype"])[1]
            saved = resume_state.get("readout", {}).get("solution")
            if saved is None:
                raise CheckpointError("A completed readout checkpoint has no solution.")
            chosen = int(saved)
            training = {
                "angles": best_theta,
                "state": self._state_at_angles(cuts, best_theta, layers, p["dtype"])[0],
                "probabilities": probabilities,
                "best_loss": resume_state.get("best_loss"),
                "objective_value": None if resume_state.get("best_loss") is None else -resume_state["best_loss"],
                "optimizer_history": resume_state.get("optimizer_history"),
                "objective_history": resume_state.get("objective_history"),
                "evaluations": int(resume_state.get("evaluations", 0)),
                "shots": int(resume_state.get("training_shots", 0)),
                "optimizer_steps": int(resume_state.get("optimizer_steps", 0)),
                "reason": resume_state.get("reason", "completed"),
                "paused": False,
                "phase": "readout_done",
            }
            readout_completed = True
        else:
            training = train_statevector(
                cuts,
                layers,
                objective=config["objective"],
                estimation_mode=config["estimation"]["mode"],
                shots_per_evaluation=config["estimation"]["shots_per_evaluation"],
                optimizer=config["optimizer"],
                initial_angles=initial_angles,
                dtype=p["dtype"],
                rng=rng,
                budget=budget,
                reserved_shots=reserved,
                collect_optimizer_history=config["collect"]["optimizer_history"],
                collect_objective_history=config["collect"]["objective_history"],
                log=lambda message, detail: log_message(config, message, detail=detail),
                state=resume_state,
                control=self._control,
                checkpoint_callback=save_checkpoint,
                state_function=lambda angles: self._state_at_angles(cuts, angles, layers, p["dtype"]),
            )
            best_theta = training["angles"]
            probabilities = training["probabilities"]
            readout_completed = False
            if training["paused"]:
                chosen = int(np.argmax(probabilities))
            else:
                if resume_state is not None and resume_state.get("phase") == "training_done":
                    # A training_done generation may contain a readout token;
                    # it is restored once, not reserved again.
                    pass
                if reserved:
                    if budget.available() is None or budget.available() >= reserved:
                        if manager is not None and manager.journal is not None:
                            manager.journal.begin(
                                f"{self._run_id}:readout",
                                attempt_id=f"readout:{manager.list_generations()[-1] if manager.list_generations() else 0}",
                                upper_bound=reserved,
                                kind="readout",
                            )
                        readout = rng.choice(len(cuts), size=reserved, p=probabilities)
                        chosen = int(readout[int(np.argmax(cuts[readout]))])
                        budget.spend(reserved)
                        readout_completed = True
                        if manager is not None and manager.journal is not None:
                            manager.journal.commit(f"{self._run_id}:readout", used=reserved)
                    else:
                        chosen = int(np.argmax(probabilities))
                else:
                    chosen = int(np.argmax(probabilities))
                    readout_completed = True
        state = training["state"]
        solution = {node: 1 - 2 * ((chosen >> i) & 1) for i, node in enumerate(graph.nodes)}
        correlations = None
        if config["collect"]["correlations"]:
            from ..tools.numerics import correlation_matrix

            correlations = correlation_matrix(probabilities)
        paused = bool(training.get("paused", False))
        already_readout = bool(resume_state is not None and resume_state.get("phase") == "readout_done")
        if not paused and readout_completed and manager is not None and not already_readout:
            final_state = dict(training.get("state_record") or {})
            final_state.update(
                {
                    "component_id": self.checkpoint_component_id,
                    "config": deepcopy(config),
                    "graph": graph.snapshot(),
                    "budget_total": config["limits"]["shots_total"],
                    "budget_used": int(budget.used),
                    "readout_reserved": 0,
                    "readout": {"completed": True, "shots": reserved, "solution": chosen},
                    "solution": deepcopy(solution),
                    "control": self._control.snapshot() if self._control is not None else None,
                    "phase": "readout_done",
                    "rng_state": rng.bit_generator.state,
                }
            )
            final_state.update(self._checkpoint_extra())
            arrays = {}
            for name in ("theta", "moment", "variance", "best_theta"):
                if name in final_state:
                    arrays[name] = np.asarray(final_state.pop(name), dtype=float)
            try:
                ref = manager.write_generation(
                    final_state,
                    arrays,
                    parent=None if manager.last_ref is None else manager.last_ref.generation,
                    phase="readout_done",
                    next_operation={"readout": "done"},
                )
                self.last_checkpoint = ref
                self.checkpoint_info = ref.as_dict()
            except Exception:
                if checkpoint_config.get("on_write_error") == "raise":
                    raise
                raise CheckpointError("Final checkpoint publication failed.")
        log_message(
            config,
            f"QAOA stops: {training['reason']}; total shots={budget.used}.",
            detail=False,
        )
        status = "paused" if paused else ("stopped" if training["reason"] == "shot_budget" else "completed")
        final_reason = "checkpoint_pause" if paused else training["reason"]
        if not paused and reserved and not readout_completed:
            status = "stopped"
            final_reason = "readout_resource_stopped"
        return {
            "solution": solution,
            "status": status,
            "termination_reason": final_reason,
            "expected_cut": float(cuts @ probabilities),
            "expected_energy": -float(cuts @ probabilities),
            "expected_cut_source": "exact_final_statevector",
            "objective_value": training["objective_value"],
            "optimizer_loss": training["best_loss"],
            "best_evaluation_id": training.get("best_evaluation_id", training.get("state_record", {}).get("best_evaluation_id")),
            "objective_source": (
                "best_training_evaluation"
                if training["best_loss"] is not None
                else "not_evaluated"
            ),
            "optimal_angles": best_theta,
            "optimizer_history": training["optimizer_history"],
            "objective_history": training["objective_history"],
            "probabilities": probabilities if config["collect"]["probabilities"] else None,
            "statevector": state if config["collect"]["statevector"] else None,
            "correlations": correlations,
            "readout_completed": readout_completed,
            "resume_mode": bool(self._resume_mode),
            "checkpoint_info": None if self.last_checkpoint is None else self.last_checkpoint.as_dict(),
            "resources": {
                "shots_used": budget.used,
                "known_spent": int(budget.used) - int(recovery_charge),
                "uncertain_upper_charge": int(recovery_charge),
                "total_charged": int(budget.used),
                "training_shots": training["shots"],
                "logical_training_shots": training["shots"],
                "readout_shots": reserved if readout_completed and not paused else 0,
                "readout_reserved": reserved if not readout_completed else 0,
                "shots_available": budget.available(reserved=reserved if not readout_completed else 0),
                "objective_evaluations": training["evaluations"],
                "optimizer_steps": training["optimizer_steps"],
                "num_stages": training["optimizer_steps"] + 1,
                "num_treads": 1,
                "checkpoint_writes": 0 if manager is None else len(manager.list_generations()),
            },
            "data_status": {
                key: "collected" if value else "not_collected"
                for key, value in config["collect"].items()
            },
        }

    def resume(self, *, control=None):
        """Continue the restored run through the next supported boundary."""
        if self._resume_state is None:
            raise RuntimeError("Call solvers.restore(path) before resume().")
        if control is not None and not hasattr(control, "should_pause"):
            raise TypeError("control must be a RunControl.")
        self._resume_mode = True
        if control is not None:
            self._resume_control = control
        return self.solve(self._resume_graph, report=0, logs=0, control=self._resume_control)

def main(argv=None):
    """Run a bounded QAOA example through the common interface."""
    from ..tools.demo import run_solver_demo

    return run_solver_demo(QAOACPU, argv)


if __name__ == "__main__":
    raise SystemExit(main())
