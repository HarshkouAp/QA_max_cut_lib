"""Simulate full-state QAOA with explicit bit order and signed energy."""

from copy import deepcopy
import numpy as np
from .base import BaseSolver, positive_int
from .objectives import validate_objective, objective_value


def cut_diagonal(graph):
    """Return C(s) for basis states ordered by little-endian node bits."""
    n = len(graph)
    indices = np.arange(1 << n, dtype=np.uint64)
    mapping = {v: i for i, v in enumerate(graph.nodes)}
    cuts = np.zeros(1 << n, dtype=float)
    for u, v, weight in graph.edges:
        cuts += weight * (
            ((indices >> np.uint64(mapping[u])) ^ (indices >> np.uint64(mapping[v])))
            & np.uint64(1)
        )
    if not np.isfinite(cuts).all():
        raise ValueError("Cut diagonal overflow.")
    return cuts


def state_at_angles(cuts, angles, layers, dtype="complex128"):
    """Apply exp(-i beta sum X) exp(-i gamma E), with E=-C.

    Parameters
    ----------
    cuts : ndarray, shape (2**N,)
        Cut values in the node order of the input graph.
    angles : array_like, shape (2*layers,)
        gamma_0,...,gamma_(p-1), beta_0,...,beta_(p-1).
    layers : int
        Number of cost and mixer layers.
    dtype : {'complex128', 'complex64'}
        Storage precision of the state vector.

    Returns
    -------
    state : ndarray, shape (2**N,)
        Full normalized state, without symmetry compression.
    probabilities : ndarray, shape (2**N,)
        Full basis-state probabilities. Index bit i belongs to node i.
    """
    n = (len(cuts) - 1).bit_length()
    angles = np.asarray(angles, dtype=float)
    if (
        len(cuts) != 1 << n
        or angles.shape != (2 * layers,)
        or not np.isfinite(angles).all()
    ):
        raise ValueError("Invalid state size or angle vector.")
    state = np.full(len(cuts), 1 / np.sqrt(len(cuts)), dtype=dtype)
    for layer in range(layers):
        # H=-C. Thus exp(-i*gamma*H) equals exp(+i*gamma*C).
        state *= np.exp(1j * angles[layer] * cuts).astype(dtype)
        beta = angles[layers + layer]
        cosine, sine = np.cos(beta), np.sin(beta)
        for qubit in range(n):
            blocks = state.reshape(-1, 2, 1 << qubit)
            first = blocks[:, 0, :].copy()
            second = blocks[:, 1, :].copy()
            # Each pair differs only in this qubit. Apply cos(beta) I
            # minus i sin(beta) X to both amplitudes from the old state.
            blocks[:, 0, :] = cosine * first - 1j * sine * second
            blocks[:, 1, :] = cosine * second - 1j * sine * first
    probabilities = state.real.astype(float) ** 2 + state.imag.astype(float) ** 2
    norm = float(probabilities.sum())
    if not np.isfinite(norm) or abs(norm - 1) > (
        1e-4 if dtype == "complex64" else 1e-9
    ):
        raise ArithmeticError("State normalization failed.")
    # Remove only the accumulated floating-point normalization error.
    state /= np.sqrt(norm)
    probabilities /= norm
    return state, probabilities


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
        op = config["optimizer"]["parameters"]
        collect = config["collect"]
        step_size = config["optimizer"]["gradient"]["parameters"]["step"]
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
        best_theta = theta.copy()
        best_loss = None
        evaluation_history = []
        history = []
        evaluations = 0
        shots = 0
        mode = config["estimation"]["mode"]
        shots_each = (
            config["estimation"]["shots_per_evaluation"] if mode == "shots" else 0
        )
        reserved = config["readout"].get("shots", 0)
        budget = config["limits"]["shots_total"]
        moment = np.zeros_like(theta)
        variance = np.zeros_like(theta)
        reason = "max_steps"

        def evaluate(angles, kind, iteration):
            nonlocal evaluations, shots, best_theta, best_loss
            _, probabilities = state_at_angles(cuts, angles, layers, p["dtype"])
            if shots_each:
                indices = rng.choice(len(cuts), size=shots_each, p=probabilities)
                masses = np.bincount(indices, minlength=len(cuts)) / shots_each
                shots += shots_each
            else:
                masses = probabilities
            value = objective_value(cuts, masses, config["objective"])
            loss = -value
            evaluations += 1
            if not np.isfinite(loss):
                raise ArithmeticError("The objective is not finite.")
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_theta = angles.copy()
            if collect["objective_history"]:
                evaluation_history.append(
                    {
                        "evaluation_id": evaluations,
                        "optimizer_step": iteration,
                        "kind": kind,
                        "objective_value": value,
                        "optimizer_loss": loss,
                        "angles": angles.tolist(),
                        "shots_used": shots,
                    }
                )
            return loss

        log_message(config, "QAOA training starts.", detail=False)
        for iteration in range(1, op["max_steps"] + 1):
            # Reserve all probes and the central evaluation as one stage.
            # Keep the final readout budget available before this stage.
            needed = (2 * len(theta) + 1) * shots_each
            if budget is not None and shots + needed + reserved > budget:
                reason = "shot_budget"
                break
            gradient = np.zeros_like(theta)
            for index in range(len(theta)):
                delta = np.zeros_like(theta)
                delta[index] = step_size
                plus = evaluate(theta + delta, "gradient_plus", iteration)
                minus = evaluate(theta - delta, "gradient_minus", iteration)
                gradient[index] = (plus - minus) / (2 * step_size)
            loss = evaluate(theta, "central", iteration)
            row = {
                "optimizer_step": iteration,
                "objective_evaluations": evaluations,
                "objective_value": -loss,
                "optimizer_loss": loss,
                "gradient_norm": float(np.linalg.norm(gradient)),
                "angles": theta.tolist(),
                "shots_used": shots,
            }
            history.append(row)
            log_message(
                config,
                f"Step {iteration}: objective={-loss:.8g}; training shots={shots}",
                detail=True,
            )
            if (
                op["gradient_tolerance"]
                and row["gradient_norm"] <= op["gradient_tolerance"]
            ):
                reason = "gradient_tolerance"
                break
            moment = op["beta1"] * moment + (1 - op["beta1"]) * gradient
            variance = op["beta2"] * variance + (1 - op["beta2"]) * gradient**2
            corrected_m = moment / (1 - op["beta1"] ** iteration)
            corrected_v = variance / (1 - op["beta2"] ** iteration)
            theta -= (
                op["learning_rate"]
                * corrected_m
                / (np.sqrt(corrected_v) + op["epsilon"])
            )
        state, probabilities = state_at_angles(cuts, best_theta, layers, p["dtype"])
        if reserved:
            readout = rng.choice(len(cuts), size=reserved, p=probabilities)
            chosen = int(readout[int(np.argmax(cuts[readout]))])
        else:
            chosen = int(np.argmax(probabilities))
        solution = {
            node: 1 - 2 * ((chosen >> i) & 1) for i, node in enumerate(graph.nodes)
        }
        correlations = None
        if collect["correlations"]:
            from ..tools.numerics import correlation_matrix

            correlations = correlation_matrix(probabilities)
        log_message(
            config,
            f"QAOA stops: {reason}; total shots={shots + reserved}.",
            detail=False,
        )
        return {
            "solution": solution,
            "status": "stopped" if reason == "shot_budget" else "completed",
            "termination_reason": reason,
            "expected_cut": float(cuts @ probabilities),
            "expected_energy": -float(cuts @ probabilities),
            "expected_cut_source": "exact_final_statevector",
            "objective_value": None if best_loss is None else -best_loss,
            "optimizer_loss": best_loss,
            "objective_source": (
                "best_training_evaluation" if best_loss is not None else "not_evaluated"
            ),
            "optimal_angles": best_theta,
            "optimizer_history": history if collect["optimizer_history"] else None,
            "objective_history": (
                evaluation_history if collect["objective_history"] else None
            ),
            "probabilities": probabilities if collect["probabilities"] else None,
            "statevector": state if collect["statevector"] else None,
            "correlations": correlations,
            "resources": {
                "shots_used": shots + reserved,
                "training_shots": shots,
                "readout_shots": reserved,
                "objective_evaluations": evaluations,
                "optimizer_steps": len(history),
                "num_stages": len(history) + 1,
                "num_treads": 1,
            },
            "data_status": {
                key: "collected" if value else "not_collected"
                for key, value in collect.items()
            },
        }


def main(argv=None):
    """Run a bounded QAOA example through the common interface."""
    from ..tools.demo import run_solver_demo

    return run_solver_demo(QAOACPU, argv)


if __name__ == "__main__":
    raise SystemExit(main())
