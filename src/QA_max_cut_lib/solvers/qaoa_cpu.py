"""Simulate full-state QAOA with explicit bit order and signed energy."""

from copy import deepcopy
import numpy as np
from .base import BaseSolver, positive_int
from .objectives import validate_objective
from .qaoa_state import cut_diagonal, state_at_angles
from .qaoa_training import train_statevector
from ..core.budget import ShotBudget


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

    def _validate(self, config, graph):
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

    def _run(self, graph, config, rng):
        from ..tools.reporting import log_message

        p = config["parameters"]
        layers = p["layers"]
        cuts = cut_diagonal(graph)
        if config["initialization"]["type"] == "fixed_angles":
            theta = np.array(
                config["initialization"]["parameters"]["values"], dtype=float
            )
        else:
            # This finite initial window is a heuristic, not a claim that
            # arbitrary weighted graphs have a universal gamma period.
            theta = np.r_[
                rng.uniform(-np.pi, np.pi, layers),
                rng.uniform(-np.pi / 2, np.pi / 2, layers),
            ]
        budget = ShotBudget(config["limits"]["shots_total"])
        reserved = config["readout"].get("shots", 0)
        training = train_statevector(
            cuts,
            layers,
            objective=config["objective"],
            estimation_mode=config["estimation"]["mode"],
            shots_per_evaluation=config["estimation"]["shots_per_evaluation"],
            optimizer=config["optimizer"],
            initial_angles=theta,
            dtype=p["dtype"],
            rng=rng,
            budget=budget,
            reserved_shots=reserved,
            collect_optimizer_history=config["collect"]["optimizer_history"],
            collect_objective_history=config["collect"]["objective_history"],
            log=lambda message, detail: log_message(
                config, message, detail=detail
            ),
        )
        state = training["state"]
        probabilities = training["probabilities"]
        best_theta = training["angles"]
        if reserved:
            readout = rng.choice(len(cuts), size=reserved, p=probabilities)
            chosen = int(readout[int(np.argmax(cuts[readout]))])
            budget.spend(reserved)
        else:
            chosen = int(np.argmax(probabilities))
        solution = {
            node: 1 - 2 * ((chosen >> i) & 1) for i, node in enumerate(graph.nodes)
        }
        correlations = None
        if config["collect"]["correlations"]:
            from ..tools.numerics import correlation_matrix

            correlations = correlation_matrix(probabilities)
        log_message(
            config,
            f"QAOA stops: {training['reason']}; total shots={budget.used}.",
            detail=False,
        )
        return {
            "solution": solution,
            "status": "stopped" if training["reason"] == "shot_budget" else "completed",
            "termination_reason": training["reason"],
            "expected_cut": float(cuts @ probabilities),
            "expected_energy": -float(cuts @ probabilities),
            "expected_cut_source": "exact_final_statevector",
            "objective_value": training["objective_value"],
            "optimizer_loss": training["best_loss"],
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
            "resources": {
                "shots_used": budget.used,
                "training_shots": training["shots"],
                "readout_shots": reserved,
                "objective_evaluations": training["evaluations"],
                "optimizer_steps": training["optimizer_steps"],
                "num_stages": training["optimizer_steps"] + 1,
                "num_treads": 1,
            },
            "data_status": {
                key: "collected" if value else "not_collected"
                for key, value in config["collect"].items()
            },
        }

def main(argv=None):
    """Run a bounded QAOA example through the common interface."""
    from ..tools.demo import run_solver_demo

    return run_solver_demo(QAOACPU, argv)


if __name__ == "__main__":
    raise SystemExit(main())
