"""Mutual-information and small density-matrix diagnostics.

The public tools separate three objects that must not be mixed:

* quantum mutual information of a full CPU statevector;
* classical Z-basis mutual information of one measured distribution/batch;
* a derived legacy ``half-nats`` view.

The implementation never builds a global ``D x D`` density matrix.  It
forms small one- and two-qubit reduced matrices from a coefficient tensor.
"""

from __future__ import annotations

from itertools import combinations
import math

import numpy as np

__all__ = [
    "MutualInformationError",
    "validate_statevector",
    "validate_node_order",
    "reduced_density_matrix",
    "von_neumann_entropy",
    "quantum_mutual_information",
    "classical_mutual_information",
    "classical_from_probabilities",
    "classical_from_counts",
    "classical_from_moments",
    "pair_scope",
    "map_summary",
    "hop_distances",
    "partial_trace",
    "entropy",
    "quantum_mi",
    "classical_mi",
    "MI_DEFINITIONS",
]


class MutualInformationError(ValueError):
    """Raised for an invalid state, pair, matrix, or information input."""


MI_DEFINITIONS = {
    "quantum": "I_Q(i:j)=S(rho_i)+S(rho_j)-S(rho_ij)",
    "classical_z": "I_Z(i:j)=sum p(a,c) log_b(p(a,c)/(p(a)p(c)))",
    "legacy_half_nats": "0.5 * I_Q in natural-log units",
}


def validate_node_order(node_order, n):
    """Validate a frozen little-endian label order."""
    if not isinstance(node_order, (list, tuple)):
        raise MutualInformationError("node_order must be a sequence.")
    if len(node_order) != int(n):
        raise MutualInformationError("node_order length must equal the state rank.")
    result = []
    for node in node_order:
        if isinstance(node, bool) or not isinstance(node, int):
            raise MutualInformationError("node_order labels must be integers.")
        if node in result:
            raise MutualInformationError("node_order labels must be unique.")
        result.append(int(node))
    return tuple(result)


def validate_statevector(statevector, node_order=None, *, atol=1e-9):
    """Validate and return a normalized complex statevector."""
    state = np.asarray(statevector)
    if state.ndim != 1 or state.size < 1:
        raise MutualInformationError("statevector must be a nonempty vector.")
    if state.size & (state.size - 1):
        raise MutualInformationError("statevector length must be a power of two.")
    if not np.issubdtype(state.dtype, np.complexfloating):
        state = state.astype(np.complex128)
    if not np.isfinite(state.real).all() or not np.isfinite(state.imag).all():
        raise MutualInformationError("statevector contains non-finite values.")
    n = int(round(math.log2(state.size)))
    if node_order is not None:
        validate_node_order(node_order, n)
    norm = float(np.vdot(state, state).real)
    if not math.isfinite(norm) or norm <= 0:
        raise MutualInformationError("statevector norm must be positive and finite.")
    if abs(norm - 1.0) > atol:
        raise MutualInformationError("statevector norm is not one within tolerance.")
    return state.copy()


def _axes_for_positions(n, positions):
    return tuple(n - 1 - int(position) for position in positions)


def reduced_density_matrix(statevector, positions, node_order=None, *, atol=1e-9):
    """Return the small RDM for selected positions in ``node_order``.

    The returned matrix uses the selected order as its row order.  For a
    pair ``(i, j)`` the rows are ``00, 01, 10, 11`` with the first label
    as the most significant bit of the small matrix.
    """
    state = validate_statevector(statevector, node_order, atol=atol)
    n = int(round(math.log2(state.size)))
    if not isinstance(positions, (list, tuple)):
        raise MutualInformationError("positions must be a sequence.")
    selected = []
    for position in positions:
        if isinstance(position, bool) or not isinstance(position, int):
            raise MutualInformationError("positions must be integers.")
        if not 0 <= position < n:
            raise MutualInformationError("position is outside node_order.")
        if position in selected:
            raise MutualInformationError("positions must be unique.")
        selected.append(int(position))
    selected = tuple(selected)
    if not selected:
        raise MutualInformationError("at least one position is required.")
    shape = (2,) * n
    tensor = state.reshape(shape)
    axes = _axes_for_positions(n, selected)
    remaining = tuple(axis for axis in range(n) if axis not in axes)
    tensor = np.transpose(tensor, axes + remaining)
    matrix = tensor.reshape((1 << len(selected), -1))
    rho = matrix @ matrix.conj().T
    rho = (rho + rho.conj().T) / 2.0
    trace = np.trace(rho)
    if not np.isfinite(rho).all() or abs(complex(trace).real - 1.0) > 1e-8:
        raise MutualInformationError("RDM trace is invalid.")
    return rho


def _base_log(base):
    if base == 2:
        return np.log2
    if base == "e" or base == math.e:
        return np.log
    raise MutualInformationError("log base must be 2 or e.")


def von_neumann_entropy(rho, *, base=2, atol=1e-10):
    """Return von Neumann entropy from Hermitian eigenvalues."""
    matrix = np.asarray(rho)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise MutualInformationError("density matrix must be square.")
    if not np.isfinite(matrix.real).all() or not np.isfinite(matrix.imag).all():
        raise MutualInformationError("density matrix contains non-finite values.")
    if not np.allclose(matrix, matrix.conj().T, atol=1e-9, rtol=1e-9):
        raise MutualInformationError("density matrix is not Hermitian.")
    trace = complex(np.trace(matrix))
    if abs(trace.real - 1.0) > 1e-8 or abs(trace.imag) > 1e-8:
        raise MutualInformationError("density matrix trace is not one.")
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.conj().T) / 2.0)
    if eigenvalues.min() < -atol:
        raise MutualInformationError("density matrix is not positive semidefinite.")
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    positive = eigenvalues[eigenvalues > 0]
    log = _base_log(base)
    return float(-np.sum(positive * log(positive)))


def _pair_positions(node_order, pair):
    labels = validate_node_order(node_order, len(node_order))
    try:
        i = labels.index(int(pair[0]))
        j = labels.index(int(pair[1]))
    except (ValueError, IndexError) as error:
        raise MutualInformationError("pair labels are not in node_order.") from error
    if i == j:
        raise MutualInformationError("pair labels must be different.")
    return i, j


def quantum_mutual_information(statevector, node_order, pair, *, base=2, atol=1e-9):
    """Return standard quantum mutual information for one label pair."""
    state = validate_statevector(statevector, node_order, atol=atol)
    i, j = _pair_positions(node_order, pair)
    rho_i = reduced_density_matrix(state, [i], node_order)
    rho_j = reduced_density_matrix(state, [j], node_order)
    rho_ij = reduced_density_matrix(state, [i, j], node_order)
    value = (
        von_neumann_entropy(rho_i, base=base)
        + von_neumann_entropy(rho_j, base=base)
        - von_neumann_entropy(rho_ij, base=base)
    )
    max_value = 2.0 * min(
        von_neumann_entropy(rho_i, base=base),
        von_neumann_entropy(rho_j, base=base),
    )
    if value < -1e-8:
        raise MutualInformationError("QMI is significantly negative.")
    if value > max_value + 1e-8:
        raise MutualInformationError("QMI exceeds the entropy upper bound.")
    return float(max(0.0, value))


def _normalized_probabilities(probabilities):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or p.size < 1 or p.size & (p.size - 1):
        raise MutualInformationError("probabilities must have power-of-two length.")
    if not np.isfinite(p).all() or np.any(p < 0):
        raise MutualInformationError("probabilities must be finite and nonnegative.")
    total = float(p.sum())
    if total <= 0 or abs(total - 1.0) > 1e-9:
        raise MutualInformationError("probabilities must sum to one.")
    return p


def _warn_or_error_negative(value, tolerance, message):
    if value < -tolerance:
        raise MutualInformationError(message)
    return max(0.0, value)


def classical_mutual_information(pair_probabilities, *, base=2, tolerance=1e-9):
    """Return classical Z MI from one 2x2 pair table in bit order."""
    table = np.asarray(pair_probabilities, dtype=float)
    if table.shape != (2, 2):
        raise MutualInformationError("pair marginal must have shape (2, 2).")
    if not np.isfinite(table).all() or np.any(table < -tolerance):
        raise MutualInformationError("pair marginal contains invalid mass.")
    if abs(float(table.sum()) - 1.0) > 1e-8:
        raise MutualInformationError("pair marginal does not sum to one.")
    table = np.clip(table, 0.0, None)
    rows = table.sum(axis=1)
    cols = table.sum(axis=0)
    log = _base_log(base)
    value = 0.0
    for a in range(2):
        for b in range(2):
            joint = float(table[a, b])
            if joint <= 0:
                continue
            if rows[a] <= 0 or cols[b] <= 0:
                raise MutualInformationError("positive joint mass has zero marginal.")
            value += joint * log(joint / (rows[a] * cols[b]))
    value = _warn_or_error_negative(value, 1e-8, "classical MI is significantly negative.")
    if abs(value) < 1e-12:
        value = 0.0
    if value > 1.0 + 1e-8:
        raise MutualInformationError("classical binary MI exceeds one bit.")
    return float(value)


def classical_from_probabilities(probabilities, node_order, pair, *, base=2):
    """Return exact Z MI from a full probability vector."""
    p = _normalized_probabilities(probabilities)
    n = int(round(math.log2(p.size)))
    labels = validate_node_order(node_order, n)
    i, j = _pair_positions(labels, pair)
    table = np.zeros((2, 2), dtype=float)
    indices = np.arange(p.size, dtype=np.uint64)
    bits_i = ((indices >> np.uint64(i)) & np.uint64(1)).astype(np.int64)
    bits_j = ((indices >> np.uint64(j)) & np.uint64(1)).astype(np.int64)
    np.add.at(table, (bits_i, bits_j), p)
    return classical_mutual_information(table, base=base)


def classical_from_counts(counts, *, base=2):
    """Return empirical Z MI from a 2x2 integer/float count table."""
    table = np.asarray(counts, dtype=float)
    if table.shape != (2, 2) or not np.isfinite(table).all() or np.any(table < 0):
        raise MutualInformationError("counts must be a finite nonnegative 2x2 table.")
    total = float(table.sum())
    if total <= 0:
        raise MutualInformationError("counts are empty.")
    return classical_mutual_information(table / total, base=base)


def classical_from_moments(m_i, m_j, q_ij, *, base=2, tolerance=1e-9):
    """Return exact Z MI from binary moments and one ZZ moment."""
    values = (float(m_i), float(m_j), float(q_ij))
    if any(not math.isfinite(value) or abs(value) > 1.0 + tolerance for value in values):
        raise MutualInformationError("moments are outside [-1, 1].")
    table = np.array(
        [
            [
                (1 + m_i + m_j + q_ij) / 4.0,
                (1 + m_i - m_j - q_ij) / 4.0,
            ],
            [
                (1 - m_i + m_j - q_ij) / 4.0,
                (1 - m_i - m_j + q_ij) / 4.0,
            ],
        ],
        dtype=float,
    )
    if table.min() < -tolerance:
        raise MutualInformationError("moments produce a negative pair probability.")
    table = np.clip(table, 0.0, None)
    total = float(table.sum())
    if abs(total - 1.0) > 1e-8:
        raise MutualInformationError("moment pair table does not sum to one.")
    return classical_mutual_information(table / total, base=base)


def pair_scope(node_order, graph, kind="all_pairs", explicit=None):
    """Return canonical unique unordered pairs for one scope."""
    labels = validate_node_order(node_order, len(node_order))
    if kind == "all_pairs":
        return tuple(combinations(labels, 2))
    if kind == "edges":
        pairs = set()
        for u, v, _ in graph.edges:
            if u not in labels or v not in labels:
                continue
            pairs.add(tuple(sorted((int(u), int(v)))))
        return tuple(sorted(pairs))
    if kind == "explicit":
        if explicit is None:
            raise MutualInformationError("explicit pairs are required.")
        pairs = set()
        for item in explicit:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise MutualInformationError("explicit pairs need two labels.")
            u, v = item
            if u not in labels or v not in labels or u == v:
                raise MutualInformationError("explicit pair labels are invalid.")
            pairs.add(tuple(sorted((int(u), int(v)))))
        return tuple(sorted(pairs))
    raise MutualInformationError("unknown pair scope.")


def map_summary(values, valid_mask, *, count=None):
    """Return honest statistics over valid unique pair values."""
    vals = np.asarray(values, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool)
    if vals.shape != mask.shape:
        raise MutualInformationError("values and mask must have the same shape.")
    selected = vals[mask & np.isfinite(vals)]
    n = int(selected.size if count is None else min(int(count), selected.size))
    if selected.size:
        selected = selected[:n]
    if n == 0:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "max": None,
            "population_std": None,
            "values": [],
        }
    return {
        "count": n,
        "mean": float(selected.mean()),
        "min": float(selected.min()),
        "max": float(selected.max()),
        "population_std": float(selected.std(ddof=0)),
        "values": [float(value) for value in selected],
    }


def hop_distances(graph, node_order=None):
    """Return unweighted hop distances and disconnected labels."""
    labels = tuple(int(node) for node in (graph.nodes if node_order is None else node_order))
    allowed = set(labels)
    adjacency = {node: set() for node in labels}
    for u, v, _ in graph.edges:
        if u in allowed and v in allowed:
            adjacency[u].add(v)
            adjacency[v].add(u)
    distances = {}
    for source in labels:
        queue = [source]
        seen = {source: 0}
        for node in queue:
            for neighbour in sorted(adjacency[node]):
                if neighbour not in seen:
                    seen[neighbour] = seen[node] + 1
                    queue.append(neighbour)
        for target in labels:
            key = tuple(sorted((source, target)))
            if source != target and target in seen:
                distances[key] = seen[target]
            elif source != target:
                distances[key] = None
    return distances, tuple(
        pair for pair, distance in distances.items() if distance is None
    )


def partial_trace(statevector, positions, node_order=None, *, atol=1e-9):
    """Alias for :func:`reduced_density_matrix`."""
    return reduced_density_matrix(statevector, positions, node_order, atol=atol)


def entropy(rho, *, base=2, atol=1e-10):
    """Alias for :func:`von_neumann_entropy`."""
    return von_neumann_entropy(rho, base=base, atol=atol)


def quantum_mi(statevector, node_order, pair, *, base=2, atol=1e-9):
    """Alias for :func:`quantum_mutual_information`."""
    return quantum_mutual_information(statevector, node_order, pair, base=base, atol=atol)


def classical_mi(pair_probabilities, *, base=2, tolerance=1e-9):
    """Alias for :func:`classical_mutual_information`."""
    return classical_mutual_information(pair_probabilities, base=base, tolerance=tolerance)
