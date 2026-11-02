"""Represent a full QAOA state vector and its bit order.

This module has no optimizer code.  Both the public QAOA solver and the
RQAOA correlation source use these functions.
"""

from __future__ import annotations

import numpy as np

__all__ = ["cut_diagonal", "state_at_angles"]


def cut_diagonal(graph):
    """Return C(s) for basis states ordered by little-endian node bits.

    Parameters
    ----------
    graph : WeightedGraph
        The weighted graph.  Node order defines the bit order.

    Returns
    -------
    ndarray, shape (2**N,)
        The cut value for each basis state.

    Notes
    -----
    Bit ``k`` of a basis index belongs to ``graph.nodes[k]``.  The spin
    is ``1 - 2 * bit``.  The cost is O(|E| * 2**N).
    """
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
    dtype : {'complex64', 'complex128'}
        Storage precision of the state vector.

    Returns
    -------
    state : ndarray, shape (2**N,)
        Full normalized state, without symmetry compression.
    probabilities : ndarray, shape (2**N,)
        Full basis-state probabilities.  Index bit i belongs to node i.

    Raises
    ------
    ValueError
        The state size, angle shape, or angle values are invalid.
    ArithmeticError
        The state does not keep unit normalization.

    Notes
    -----
    H = -C.  Thus exp(-i*gamma*H) equals exp(+i*gamma*C).  The mixer
    acts on one qubit at a time because the X operators commute.  Both
    new amplitudes use the old pair.  The cost is O(p*N*2**N).
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
            # Each pair differs only in the selected qubit.  Apply
            # cos(beta) I - i sin(beta) X to both old amplitudes.
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
