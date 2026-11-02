"""Shared QAOA state-vector training loop.

The public ``QAOACPU`` solver and the RQAOA correlation source use this
engine.  The engine returns angles, the final state, and resource
counters.  It does not perform a discrete readout.
"""

from __future__ import annotations

import math
import numpy as np

from .objectives import objective_value
from .qaoa_state import state_at_angles

__all__ = ["train_statevector"]


def train_statevector(
    cuts,
    layers,
    *,
    objective,
    estimation_mode,
    shots_per_evaluation,
    optimizer,
    initial_angles,
    dtype,
    rng,
    budget,
    reserved_shots=0,
    collect_optimizer_history=False,
    collect_objective_history=False,
    log=None,
):
    """Train QAOA at fixed cuts and return the best evaluated angles.

    Parameters
    ----------
    cuts : ndarray, shape (2**N,)
        Cut values in the current bit order.
    layers : int
        Number of QAOA layers ``p``.
    objective : dict
        Resolved criterion from ``validate_objective``.
    estimation_mode : {'exact', 'shots'}
        Criterion estimation mode.
    shots_per_evaluation : int
        Shots per criterion evaluation when mode is ``shots``.
    optimizer : dict
        Resolved ADAM configuration with ``parameters`` and ``gradient``.
    initial_angles : array_like, shape (2*p,)
        Initial gamma-then-beta vector.
    dtype : {'complex64', 'complex128'}
        State-vector precision.
    rng : numpy.random.Generator
        Random source for shots and random angle initialization is done
        by the caller.  The engine uses this source for criterion shots.
    budget : ShotBudget
        Root shot budget.
    reserved_shots : int, default 0
        Shots that must stay available for a later measurement.
    collect_optimizer_history, collect_objective_history : bool
        Store the corresponding lists in the result.
    log : callable or None, optional
        Callback ``log(message, detail)`` for progress text.

    Returns
    -------
    dict
        Keys: ``angles``, ``state``, ``probabilities``, ``best_loss``,
        ``objective_value``, ``optimizer_history``, ``objective_history``,
        ``evaluations``, ``shots``, ``optimizer_steps``, ``reason``,
        ``initial_angles``.

    Notes
    -----
    A full ADAM step with central differences evaluates the criterion
    ``2*d + 1 = 4*p + 1`` times.  The engine checks the whole stage
    against the budget before it starts the stage.  The final discrete
    bit string is not produced here.
    """
    theta = np.asarray(initial_angles, dtype=float).copy()
    if theta.shape != (2 * layers,) or not np.isfinite(theta).all():
        raise ValueError("initial_angles must contain 2*layers finite values.")
    op = optimizer["parameters"]
    step_size = optimizer["gradient"]["parameters"]["step"]
    shots_each = int(shots_per_evaluation) if estimation_mode == "shots" else 0
    if estimation_mode not in ("exact", "shots"):
        raise ValueError("estimation_mode must be exact or shots.")
    if shots_each < 0:
        raise ValueError("shots_per_evaluation must be positive.")

    best_theta = theta.copy()
    best_loss = None
    evaluations = 0
    shots = 0
    history = []
    evaluation_history = []
    moment = np.zeros_like(theta)
    variance = np.zeros_like(theta)
    reason = "max_steps"

    def emit(message, detail):
        if log is not None:
            log(message, detail)

    def evaluate(angles, kind, iteration):
        nonlocal evaluations, shots, best_theta, best_loss
        _, probabilities = state_at_angles(cuts, angles, layers, dtype)
        if shots_each:
            indices = rng.choice(len(cuts), size=shots_each, p=probabilities)
            masses = np.bincount(indices, minlength=len(cuts)) / shots_each
            budget.spend(shots_each, reserved=reserved_shots)
            shots += shots_each
        else:
            masses = probabilities
        value = objective_value(cuts, masses, objective)
        loss = -value
        evaluations += 1
        if not math.isfinite(loss):
            raise ArithmeticError("The objective is not finite.")
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_theta = np.asarray(angles, dtype=float).copy()
        if collect_objective_history:
            evaluation_history.append(
                {
                    "evaluation_id": evaluations,
                    "optimizer_step": iteration,
                    "kind": kind,
                    "objective_value": value,
                    "optimizer_loss": loss,
                    "angles": np.asarray(angles, dtype=float).tolist(),
                    "shots_used": shots,
                }
            )
        return loss

    emit("QAOA training starts.", False)
    for iteration in range(1, op["max_steps"] + 1):
        # Reserve all probes and the central evaluation as one stage.
        # The caller's reserved shots stay untouched.
        needed = (2 * len(theta) + 1) * shots_each
        if not budget.can_spend(needed, reserved=reserved_shots):
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
        emit(
            f"Step {iteration}: objective={-loss:.8g}; training shots={shots}",
            True,
        )
        if op["gradient_tolerance"] and row["gradient_norm"] <= op["gradient_tolerance"]:
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

    state, probabilities = state_at_angles(cuts, best_theta, layers, dtype)
    return {
        "angles": best_theta,
        "state": state,
        "probabilities": probabilities,
        "best_loss": best_loss,
        "objective_value": None if best_loss is None else -best_loss,
        "optimizer_history": history if collect_optimizer_history else None,
        "objective_history": (
            evaluation_history if collect_objective_history else None
        ),
        "evaluations": evaluations,
        "shots": shots,
        "optimizer_steps": len(history),
        "reason": reason,
        "initial_angles": np.asarray(initial_angles, dtype=float).copy(),
    }
