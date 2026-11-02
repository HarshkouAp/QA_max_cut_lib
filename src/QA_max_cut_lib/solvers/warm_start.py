"""Warm-start preparation and mixer gates for the first Laboratory transfer.

The implementation uses the package's full CPU statevector convention:
node position ``k`` is bit ``k`` and spin ``s = 1 - 2*b``.  It deliberately
does not reuse the legacy big-endian arrays or the symmetric chi path.
"""

from __future__ import annotations

from copy import deepcopy
import math
import numpy as np

__all__ = [
    "WarmStartError",
    "validate_warm_start_section",
    "prepare_warm_start",
    "product_state",
    "state_at_warm_angles",
    "mixer_unitary",
    "clipped_probability",
    "source_spins_to_probabilities",
]

MIXERS = ("warm_aligned", "rounded_modified", "x")
SOURCE_TYPES = ("provided_spins", "provided_probabilities", "classical_solver")
CLASSICAL_TYPES = ("gw_cpu", "simulated_annealing_cpu", "exact_cpu")


class WarmStartError(ValueError):
    """Raised when a warm-start recipe is invalid or unavailable."""


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WarmStartError(f"{name} must be a finite number.")
    value = float(value)
    if not math.isfinite(value):
        raise WarmStartError(f"{name} must be finite.")
    return value


def _node(value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise WarmStartError("Source nodes must be integer labels.")
    return value


def _labels(graph):
    return tuple(graph.nodes)


def _entries(source, graph):
    """Validate a complete node entry list and return a label -> value map."""
    params = source.get("parameters", {})
    if not isinstance(params, dict):
        raise WarmStartError("source.parameters must be an object.")
    entries = params.get("entries")
    if not isinstance(entries, list):
        raise WarmStartError("provided source parameters.entries must be a list.")
    labels = _labels(graph)
    expected = set(labels)
    found = {}
    for item in entries:
        if not isinstance(item, dict):
            raise WarmStartError("Each source entry must be an object.")
        if "node" not in item:
            raise WarmStartError("Each source entry needs a node.")
        node = _node(item["node"])
        if node not in expected:
            raise WarmStartError(f"Source node is not in the graph: {node}")
        if node in found:
            raise WarmStartError(f"Duplicate source node: {node}")
        found[node] = item
    if set(found) != expected:
        missing = sorted(expected - set(found))
        raise WarmStartError(f"Source does not cover graph nodes: {missing}")
    return found


def validate_warm_start_section(section, graph):
    """Resolve and validate a warm-start recipe before expensive work."""
    if not isinstance(section, dict):
        raise WarmStartError("warm_start must be an object.")
    if set(section) - {"epsilon", "source", "mixer", "variant_id"}:
        raise WarmStartError(f"Unknown warm_start fields: {set(section) - {'epsilon', 'source', 'mixer', 'variant_id'}}")
    epsilon = _finite(section.get("epsilon", 0.25), "warm_start.epsilon")
    if not 0.0 <= epsilon <= 0.5:
        raise WarmStartError("warm_start.epsilon must be in [0, 0.5].")
    source = section.get("source")
    if not isinstance(source, dict):
        raise WarmStartError("warm_start.source must be an object.")
    if set(source) - {"type", "parameters"}:
        raise WarmStartError("Unknown source fields.")
    source_type = source.get("type")
    if source_type not in SOURCE_TYPES:
        raise WarmStartError(
            f"warm_start.source.type must be one of {SOURCE_TYPES}."
        )
    params = source.get("parameters", {})
    if not isinstance(params, dict):
        raise WarmStartError("source.parameters must be an object.")
    if source_type in ("provided_spins", "provided_probabilities"):
        if set(params) != {"entries"}:
            raise WarmStartError("Provided sources require exactly parameters.entries.")
        entries = _entries(source, graph)
        if source_type == "provided_spins":
            for item in entries.values():
                if set(item) - {"node", "spin"}:
                    raise WarmStartError("provided_spins entries need node and spin.")
                spin = item.get("spin")
                if isinstance(spin, bool) or spin not in (-1, 1):
                    raise WarmStartError("provided_spins spins must be integer +1 or -1.")
        else:
            for item in entries.values():
                if set(item) - {"node", "probability"}:
                    raise WarmStartError(
                        "provided_probabilities entries need node and probability."
                    )
                q = _finite(item.get("probability"), "probability")
                if not 0.0 <= q <= 1.0:
                    raise WarmStartError("provided probabilities must be in [0, 1].")
    else:
        if set(params) - {"solver_type", "config"}:
            raise WarmStartError("classical_solver parameters allow solver_type and config.")
        solver_type = params.get("solver_type")
        if solver_type not in CLASSICAL_TYPES:
            raise WarmStartError(
                f"classical_solver.solver_type must be one of {CLASSICAL_TYPES}."
            )
        solver_config = params.get("config", {})
        if not isinstance(solver_config, dict):
            raise WarmStartError("classical_solver.config must be an object.")
        if solver_config.get("type", solver_type) != solver_type:
            raise WarmStartError("The child classical solver type is inconsistent.")
        if "checkpoint" in solver_config:
            raise WarmStartError("A classical warm-start child cannot hold a checkpoint.")
    mixer = section.get("mixer", {})
    if not isinstance(mixer, dict) or set(mixer) - {"type"}:
        raise WarmStartError("warm_start.mixer must be an object with type.")
    if mixer.get("type") not in MIXERS:
        raise WarmStartError(f"warm_start.mixer.type must be one of {MIXERS}.")
    return {
        "epsilon": epsilon,
        "source": deepcopy(source),
        "mixer": {"type": mixer["type"]},
    }


def clipped_probability(q, epsilon):
    """Return clipped q; reject non-finite or out-of-range input first."""
    values = np.asarray(q, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise WarmStartError("Warm-start probabilities must be a finite vector.")
    if (values < 0.0).any() or (values > 1.0).any():
        raise WarmStartError("Warm-start probabilities must be in [0, 1].")
    eps = _finite(epsilon, "epsilon")
    if not 0.0 <= eps <= 0.5:
        raise WarmStartError("epsilon must be in [0, 0.5].")
    return np.clip(values, eps, 1.0 - eps)


def source_spins_to_probabilities(spins):
    """Map integer +/-1 spins to little-endian bit probabilities."""
    values = np.asarray(list(spins), dtype=float)
    if not np.isfinite(values).all() or not np.isin(values, (-1.0, 1.0)).all():
        raise WarmStartError("Seed spins must be finite +1/-1 values.")
    return (1.0 - values) / 2.0


def product_state(probabilities, dtype="complex128"):
    """Build a normalized product state without a 2**N by N table."""
    q = np.asarray(probabilities, dtype=float)
    if not np.isfinite(q).all() or (q < 0).any() or (q > 1).any():
        raise WarmStartError("product_state needs finite probabilities in [0, 1].")
    state = np.ones(1, dtype=dtype)
    for value in q:
        amplitude = np.array(
            [math.sqrt(1.0 - float(value)), math.sqrt(float(value))], dtype=dtype
        )
        new = np.concatenate(
            (state * amplitude[0], state * amplitude[1])
        ).astype(dtype, copy=False)
        state = new
    norm = float(np.vdot(state, state).real)
    if not math.isfinite(norm) or abs(norm - 1.0) > 1e-9:
        raise ArithmeticError("Warm product state normalization failed.")
    return state / math.sqrt(norm)


def mixer_unitary(beta, q, mixer, dtype="complex128"):
    """Return one 2x2 mixer matrix using the package spin/bit convention."""
    beta = _finite(beta, "beta")
    value = float(q)
    if not 0.0 <= value <= 1.0 or not math.isfinite(value):
        raise WarmStartError("q must be finite and in [0, 1].")
    c = math.cos(beta)
    s = math.sin(beta)
    if mixer == "x":
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=dtype)
    if mixer not in ("warm_aligned", "rounded_modified"):
        raise WarmStartError(f"Unknown mixer: {mixer}")
    cos_alpha = 1.0 - 2.0 * value
    sin_alpha = 2.0 * math.sqrt(max(0.0, value * (1.0 - value)))
    # The two axes differ only by the sign of the off-diagonal term.
    sign = 1.0 if mixer == "warm_aligned" else -1.0
    return np.array(
        [
            [c + 1j * s * cos_alpha, 1j * sign * s * sin_alpha],
            [1j * sign * s * sin_alpha, c - 1j * s * cos_alpha],
        ],
        dtype=dtype,
    )


def _apply_gate(state, matrix, qubit):
    blocks = state.reshape(-1, 2, 1 << qubit)
    first = blocks[:, 0, :].copy()
    second = blocks[:, 1, :].copy()
    blocks[:, 0, :] = matrix[0, 0] * first + matrix[0, 1] * second
    blocks[:, 1, :] = matrix[1, 0] * first + matrix[1, 1] * second
    return state


def state_at_warm_angles(cuts, angles, layers, probabilities, mixer, dtype="complex128", initial_state=None):
    """Apply p cost layers and one selected warm mixer.

    The cost phase is ``exp(+i*gamma*C)`` because ``H_C=-C``.  The returned
    array is a full little-endian statevector, not a compressed chi state.
    """
    n = (len(cuts) - 1).bit_length()
    values = np.asarray(angles, dtype=float)
    if len(cuts) != 1 << n or values.shape != (2 * layers,) or not np.isfinite(values).all():
        raise WarmStartError("Invalid warm state size or angle vector.")
    q = np.asarray(probabilities, dtype=float)
    if len(q) != n:
        raise WarmStartError("probabilities must have one value per qubit.")
    state = product_state(q, dtype=dtype) if initial_state is None else np.asarray(initial_state, dtype=dtype).copy()
    if len(state) != len(cuts):
        raise WarmStartError("Initial state has the wrong size.")
    for layer in range(layers):
        state *= np.exp(1j * values[layer] * cuts).astype(dtype)
        for qubit, value in enumerate(q):
            state = _apply_gate(
                state, mixer_unitary(values[layers + layer], value, mixer, dtype), qubit
            )
    probabilities_out = state.real.astype(float) ** 2 + state.imag.astype(float) ** 2
    norm = float(probabilities_out.sum())
    if not math.isfinite(norm) or abs(norm - 1.0) > (1e-4 if dtype == "complex64" else 1e-9):
        raise ArithmeticError("Warm state normalization failed.")
    state /= math.sqrt(norm)
    probabilities_out /= norm
    return state, probabilities_out


def _provided_spins(graph, source):
    entries = _entries(source, graph)
    spins = {node: int(entries[node]["spin"]) for node in graph.nodes}
    q_raw = source_spins_to_probabilities([spins[node] for node in graph.nodes])
    return q_raw, spins, graph.cut_value(spins)


def _provided_probabilities(graph, source):
    entries = _entries(source, graph)
    q_raw = np.array(
        [float(entries[node]["probability"]) for node in graph.nodes], dtype=float
    )
    return q_raw, None, None


def _classical(graph, source, *, seed=None):
    from ..core.factory import SolverFactory

    params = source["parameters"]
    solver_type = params["solver_type"]
    config = deepcopy(params.get("config", {}))
    config["type"] = solver_type
    if solver_type not in CLASSICAL_TYPES:
        raise WarmStartError("Unsupported classical source.")
    if config.get("seed") is None and seed is not None:
        config["seed"] = int(seed)
    child = SolverFactory.from_config(config)
    solution = child.solve(graph)
    resources = deepcopy(getattr(child.result, "get", lambda key, default=None: None)("resources", {}) or {})
    return solution, resources


def prepare_warm_start(graph, section, *, seed=None):
    """Prepare q, clipped q, preparation angles, and source provenance."""
    resolved = validate_warm_start_section(section, graph)
    source = resolved["source"]
    source_type = source["type"]
    source_cost = None
    source_run = None
    if source_type == "provided_spins":
        q_raw, spins, source_cut = _provided_spins(graph, source)
    elif source_type == "provided_probabilities":
        q_raw, spins, source_cut = _provided_probabilities(graph, source)
    else:
        solution, resources = _classical(graph, source, seed=seed)
        q_raw = source_spins_to_probabilities([solution[node] for node in graph.nodes])
        spins = {int(node): int(spin) for node, spin in solution.items()}
        source_cut = graph.cut_value(spins)
        source_cost = resources
        source_run = {
            "solver_type": source["parameters"]["solver_type"],
            "solution": spins,
            "seed": source["parameters"].get("config", {}).get("seed", seed),
        }
    q = clipped_probability(q_raw, resolved["epsilon"])
    alpha = 2.0 * np.arcsin(np.sqrt(q))
    product = product_state(q, "complex128")
    from .qaoa_state import cut_diagonal

    cuts = cut_diagonal(graph)
    initial_cut = float(cuts @ (np.abs(product) ** 2))
    return {
        "schema_version": 1,
        "variant_id": "warm_start_product_qaoa_cpu_v1",
        "epsilon": resolved["epsilon"],
        "mixer": resolved["mixer"]["type"],
        "source_type": source_type,
        "source": deepcopy(source),
        "node_order": list(graph.nodes),
        "q_raw": [float(value) for value in q_raw],
        "q_clipped": [float(value) for value in q],
        "warm_start_angles": [float(value) for value in alpha],
        "source_cut": None if source_cut is None else float(source_cut),
        "discrete_seed": None if spins is None else {int(k): int(v) for k, v in spins.items()},
        "source_cost": source_cost,
        "source_cost_status": "measured" if source_cost is not None else "external_unaccounted",
        "source_run": source_run,
        "initial_expected_cut": initial_cut,
        "initial_expected_energy": None if initial_cut is None else -initial_cut,
    }
