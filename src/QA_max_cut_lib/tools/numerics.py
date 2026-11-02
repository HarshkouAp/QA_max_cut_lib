"""Compute physical moments from a full probability distribution."""

import numpy as np


def validate_probabilities(probabilities):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or len(p) < 1 or len(p) & (len(p) - 1):
        raise ValueError("A full state distribution must have length 2**N.")
    if (
        not np.isfinite(p).all()
        or np.any(p < 0)
        or not np.isclose(p.sum(), 1, atol=1e-9, rtol=1e-9)
    ):
        raise ValueError("Supply a normalized full probability vector.")
    return p


def correlation_matrix(probabilities):
    """Calculate physical <Z_i Z_j> in the recorded node order.

    Parameters
    ----------
    probabilities : array_like, shape (2**N,)
        Full normalized probabilities. Basis index bit i is spin i.

    Returns
    -------
    ndarray, shape (N, N)
        Symmetric correlations in [-1, 1], with diagonal entries one.

    Notes
    -----
    Z_i Z_j is +1 for equal bits and -1 for different bits.
    Thus its expectation is 1 - 2 P(bits differ). A zero entry is a
    computed correlation, not a missing pair.
    """
    p = validate_probabilities(probabilities)
    n = (len(p) - 1).bit_length()
    states = np.arange(len(p), dtype=np.uint64)
    result = np.eye(n)
    for i in range(n):
        for j in range(i):
            different = (
                (states >> np.uint64(i)) ^ (states >> np.uint64(j))
            ) & np.uint64(1)
            result[i, j] = result[j, i] = 1 - 2 * float(p @ different)
    return result


def quality_score(cut_value, optimum, worst):
    """Return (C-W)/(C_star-W), or None if all cut values coincide.

    References must belong to the same graph and use cut units. This
    function does not compute a reference or assume that GW is exact.
    """
    values = np.array([cut_value, optimum, worst], dtype=float)
    if not np.isfinite(values).all() or optimum < worst:
        raise ValueError("Supply finite, correctly ordered reference cuts.")
    return None if optimum == worst else float((cut_value - worst) / (optimum - worst))
