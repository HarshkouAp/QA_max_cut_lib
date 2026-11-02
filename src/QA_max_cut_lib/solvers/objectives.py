"""Maximize distribution criteria in cut units, or in probability units."""

import numpy as np
from scipy.special import logsumexp


def validate_objective(config):
    """Validate the criterion and return its resolved configuration."""
    if set(config) - {"type", "parameters"}:
        raise ValueError("Unknown objective configuration field.")
    kind = config.get("type", "mean_cut")
    p = dict(config.get("parameters", {}))
    keys = {
        "mean_cut": set(),
        "cvar": {"alpha"},
        "gibbs": {"eta"},
        "best_of_k": {"k"},
        "threshold_probability": {"threshold"},
    }
    if kind not in keys or set(p) != keys[kind]:
        raise ValueError(f"Invalid objective or parameter names for {kind}.")
    if kind == "cvar" and not (np.isfinite(p["alpha"]) and 0 < p["alpha"] <= 1):
        raise ValueError("CVaR alpha must be in (0, 1].")
    if kind == "gibbs" and not (np.isfinite(p["eta"]) and p["eta"] > 0):
        raise ValueError("Gibbs eta must be finite and positive.")
    if kind == "best_of_k" and (type(p["k"]) is not int or p["k"] < 1):
        raise ValueError("Best-of-K k must be a positive integer.")
    if kind == "threshold_probability" and not np.isfinite(p["threshold"]):
        raise ValueError("The cut threshold must be finite.")
    return {"type": kind, "parameters": p}


def objective_value(cuts, probabilities, config):
    """Return a criterion to maximize on a normalized cut distribution.

    Parameters
    ----------
    cuts : array_like, shape (S,)
        Cut values C(s), including negative values when present.
    probabilities : array_like, shape (S,)
        Nonnegative masses with sum one.
    config : dict
        A resolved criterion configuration from validate_objective.

    Returns
    -------
    float
        Mean cut, upper-tail CVaR, log-Gibbs value, expected best of K,
        or probability of reaching an absolute cut threshold.

    Notes
    -----
    The optimizer minimizes the negative of this value. This sign does
    not change the cost Hamiltonian E=-C in the quantum circuit.
    """
    cuts = np.asarray(cuts, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if cuts.ndim != 1 or p.shape != cuts.shape or not np.isfinite(cuts).all():
        raise ValueError("cuts and probabilities must be matching finite vectors.")
    if (
        not np.isfinite(p).all()
        or np.any(p < 0)
        or not np.isclose(p.sum(), 1, atol=1e-9, rtol=1e-9)
    ):
        raise ValueError("Probabilities must be finite, nonnegative, and normalized.")
    kind = config["type"]
    pars = config["parameters"]
    if kind == "mean_cut":
        return float(cuts @ p)
    if kind == "threshold_probability":
        return float(p[cuts >= pars["threshold"]].sum())
    if kind == "gibbs":
        keep = p > 0
        eta = pars["eta"]
        return float(logsumexp(np.log(p[keep]) + eta * cuts[keep]) / eta)
    order = np.argsort(cuts)
    if kind == "cvar":
        # Retain exactly alpha of the best probability mass.
        # Split a boundary state's mass instead of taking the whole state.
        order = order[::-1]
        masses = p[order]
        before = np.cumsum(masses) - masses
        selected = np.minimum(masses, np.maximum(pars["alpha"] - before, 0))
        return float(cuts[order] @ selected / pars["alpha"])
    if kind == "best_of_k":
        # For independent draws, the maximum has CDF F(c)**K.
        cdf = np.clip(np.cumsum(p[order]), 0, 1)
        cdf[-1] = 1.0
        maximum_mass = np.diff(np.r_[0.0, cdf ** pars["k"]])
        return float(cuts[order] @ maximum_mass)
    raise ValueError(f"Unknown objective: {kind}")
