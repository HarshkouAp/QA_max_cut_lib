"""GPU QAOA state, sampler, criteria, and training helpers.

CuPy is imported only when a GPU path is requested.  The module is also
readable without CuPy so CPU-only installations can import the package.
"""

from __future__ import annotations

from copy import deepcopy
import math
from numbers import Real

import numpy as np

__all__ = [
    "gpu_available",
    "gpu_device_info",
    "validate_gpu_config",
    "cut_diagonal_gpu",
    "state_at_angles_gpu",
    "forward_states_gpu",
    "restore_full_gpu",
    "probabilities_gpu",
    "correlations_gpu",
    "objective_value_gpu",
    "objective_from_samples_gpu",
    "sample_full_indices_gpu",
    "adjoint_mean_cut_gradient_gpu",
    "train_statevector_gpu",
]


def _missing_cupy():
    raise RuntimeError(
        "CuPy/NVIDIA is not available. Install one GPU extra and check the driver; "
        "the CPU solvers remain usable."
    )


def _cupy():
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - depends on the host
        raise RuntimeError(
            "CuPy/NVIDIA is not available. Install one GPU extra and check the driver; "
            "the CPU solvers remain usable."
        ) from exc
    return cp


def gpu_available():
    """Return True when CuPy imports and at least one CUDA device exists."""
    try:
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def gpu_device_info(device_id=0):
    """Return runtime fields for one device; raise a clear error if absent."""
    cp = _cupy()
    try:
        device = cp.cuda.Device(int(device_id))
        device.use()
        props = cp.cuda.runtime.getDeviceProperties(int(device_id))
        free, total = device.mem_info
        return {
            "device_id": int(device_id),
            "name": props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"]),
            "compute_capability": [int(props["major"]), int(props["minor"])],
            "multiprocessors": int(props["multiProcessorCount"]),
            "memory_free": int(free),
            "memory_total": int(total),
            "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "cupy_version": str(cp.__version__),
        }
    except Exception as exc:
        raise RuntimeError(f"GPU device {device_id!r} is not available: {exc}") from exc


def _positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def validate_gpu_config(config, *, memory_buffers=4):
    """Validate the ``gpu`` section and return a resolved copy.

    Parameters
    ----------
    config : dict, optional
        GPU configuration section.
    memory_buffers : int, optional
        Number of complex arrays expected during one state evaluation.
    """
    config = deepcopy(config or {})
    allowed = {
        "device_id",
        "diagonal_dtype",
        "accumulation_dtype",
        "memory",
        "kernels",
        "state_batch_size",
    }
    if set(config) - allowed:
        raise ValueError(f"Unknown GPU field: {set(config) - allowed}")
    config.setdefault("device_id", 0)
    _positive_int(config["device_id"], "gpu.device_id", minimum=0)
    config.setdefault("diagonal_dtype", "float64")
    config.setdefault("accumulation_dtype", "float64")
    if config["diagonal_dtype"] not in ("float32", "float64"):
        raise ValueError("gpu.diagonal_dtype must be float32 or float64.")
    if config["accumulation_dtype"] not in ("float32", "float64"):
        raise ValueError("gpu.accumulation_dtype must be float32 or float64.")
    memory = config.setdefault("memory", {})
    if set(memory) - {"fraction_of_available", "reserve_bytes"}:
        raise ValueError("Unknown gpu.memory field.")
    memory.setdefault("fraction_of_available", 0.8)
    memory.setdefault("reserve_bytes", 268435456)
    fraction = _finite(memory["fraction_of_available"], "fraction_of_available")
    if not 0 < fraction <= 1:
        raise ValueError("fraction_of_available must be in (0, 1].")
    reserve = memory["reserve_bytes"]
    if isinstance(reserve, bool) or not isinstance(reserve, int) or reserve < 0:
        raise ValueError("reserve_bytes must be a nonnegative integer.")
    kernels = config.setdefault("kernels", {})
    if set(kernels) - {"mixer", "cuda_graphs"}:
        raise ValueError("Unknown gpu.kernels field.")
    kernels.setdefault("mixer", "pairwise")
    kernels.setdefault("cuda_graphs", False)
    if kernels["mixer"] != "pairwise":
        raise ValueError("Only kernels.mixer='pairwise' is implemented.")
    if kernels["cuda_graphs"] is not False:
        raise ValueError("CUDA Graphs are not implemented in this version.")
    config.setdefault("state_batch_size", 1)
    _positive_int(config["state_batch_size"], "state_batch_size")
    if config["state_batch_size"] != 1:
        raise ValueError("state_batch_size=1 is the supported first GPU mode.")
    return config


def memory_preflight(config, n_vertices, representation, dtype="complex64", layers=1, buffer_count=4):
    """Check the dominant allocation before a GPU state is created.

    The estimate intentionally counts only this run's large arrays.  It does
    not treat another process's CuPy pool as exclusive free memory.
    """
    _cupy()
    if representation == "full":
        size = 1 << int(n_vertices)
    elif representation == "spin_flip_pairs":
        size = 1 << max(0, int(n_vertices) - 1)
    else:
        raise ValueError("Unknown state representation.")
    complex_bytes = 16 if dtype == "complex128" else 8
    diagonal_bytes = 8 if (config or {}).get("diagonal_dtype", "float64") == "float64" else 4
    # State, diagonal, probabilities, and one temporary are counted.  A
    # more exact plan is reported by the solver; this guard is conservative.
    estimate = int(size) * (complex_bytes * buffer_count + diagonal_bytes + 8 * 2)
    gpu = config if isinstance(config, dict) else {}
    info = gpu_device_info(gpu.get("device_id", 0))
    fraction = float(gpu.get("memory", {}).get("fraction_of_available", 0.8))
    reserve = int(gpu.get("memory", {}).get("reserve_bytes", 268435456))
    limit = min(int(info["memory_free"] * fraction), max(0, int(info["memory_free"]) - reserve))
    if estimate > limit:
        raise MemoryError(
            f"GPU memory preflight rejected N={n_vertices} ({representation}): "
            f"estimated {estimate} bytes, workspace {limit} bytes."
        )
    return {
        "estimated_bytes": int(estimate),
        "workspace_limit": int(limit),
        "device_free": int(info["memory_free"]),
        "device_total": int(info["memory_total"]),
        "fraction_of_available": fraction,
        "reserve_bytes": reserve,
    }


def _dtype(cp, name):
    return cp.complex64 if name == "complex64" else cp.complex128


def _diag_dtype(cp, name):
    return cp.float32 if name == "float32" else cp.float64


def cut_diagonal_gpu(graph, representation, *, diagonal_dtype="float64"):
    """Compute C(s) on GPU for full or representative pair indices."""
    cp = _cupy()
    if representation not in ("full", "spin_flip_pairs"):
        raise ValueError("Unknown state representation.")
    n = len(graph)
    size = (1 << n) if representation == "full" else (1 << max(0, n - 1))
    if size <= 0:
        raise ValueError("An empty graph has no QAOA basis states.")
    indices = cp.arange(size, dtype=cp.uint64)
    cuts = cp.zeros(size, dtype=_diag_dtype(cp, diagonal_dtype))
    mapping = {node: index for index, node in enumerate(graph.nodes)}
    for u, v, weight in graph.edges:
        i = mapping[u]
        j = mapping[v]
        if representation == "full":
            bit_i = (indices >> cp.uint64(i)) & cp.uint64(1)
            bit_j = (indices >> cp.uint64(j)) & cp.uint64(1)
        else:
            bit_i = (
                cp.zeros_like(indices)
                if i == n - 1
                else (indices >> cp.uint64(i)) & cp.uint64(1)
            )
            bit_j = (
                cp.zeros_like(indices)
                if j == n - 1
                else (indices >> cp.uint64(j)) & cp.uint64(1)
            )
        cuts += (bit_i ^ bit_j).astype(_diag_dtype(cp, diagonal_dtype)) * _diag_dtype(
            cp, diagonal_dtype
        )(float(weight))
    return cuts


def state_at_angles_gpu(
    cuts,
    angles,
    layers,
    *,
    n_vertices,
    representation,
    dtype="complex64",
):
    """Apply the alternating cost/mixer circuit to a CuPy state array."""
    states, costs = forward_states_gpu(
        cuts,
        angles,
        layers,
        n_vertices=n_vertices,
        representation=representation,
        dtype=dtype,
    )
    return states[-1], cp_abs_squared(states[-1])


def cp_abs_squared(state):
    cp = _cupy()
    return cp.abs(state) ** 2


def _mixer_apply_gpu(state, beta, n_vertices, representation):
    cp = _cupy()
    result = state.copy()
    cosine, sine = cp.cos(beta), cp.sin(beta)
    if representation == "full":
        qubits = range(int(n_vertices))
    elif representation == "spin_flip_pairs":
        qubits = range(max(0, int(n_vertices) - 1))
    else:
        raise ValueError("Unknown state representation.")
    for qubit in qubits:
        blocks = result.reshape(-1, 2, 1 << qubit)
        first = blocks[:, 0, :].copy()
        second = blocks[:, 1, :].copy()
        blocks[:, 0, :] = cosine * first - 1j * sine * second
        blocks[:, 1, :] = cosine * second - 1j * sine * first
    if representation == "spin_flip_pairs":
        result = cosine * result - 1j * sine * result[::-1]
    return result


def _x_sum_apply_gpu(state, n_vertices, representation):
    """Apply the sum of single-qubit X operators."""
    cp = _cupy()
    result = cp.zeros_like(state)
    if representation == "full":
        qubits = range(int(n_vertices))
    elif representation == "spin_flip_pairs":
        qubits = range(max(0, int(n_vertices) - 1))
    else:
        raise ValueError("Unknown state representation.")
    for qubit in qubits:
        blocks = state.reshape(-1, 2, 1 << qubit)
        first = blocks[:, 0, :]
        second = blocks[:, 1, :]
        result.reshape(-1, 2, 1 << qubit)[:, 0, :] += second
        result.reshape(-1, 2, 1 << qubit)[:, 1, :] += first
    if representation == "spin_flip_pairs":
        result += state[::-1]
    return result


def forward_states_gpu(
    cuts,
    angles,
    layers,
    *,
    n_vertices,
    representation,
    dtype="complex64",
):
    """Return states after each mixer and after each cost phase."""
    cp = _cupy()
    angles = np.asarray(angles, dtype=float)
    if angles.shape != (2 * int(layers),) or not np.isfinite(angles).all():
        raise ValueError("angles must contain 2*layers finite values.")
    size = (
        1 << int(n_vertices)
        if representation == "full"
        else 1 << max(0, int(n_vertices) - 1)
    )
    if int(cuts.size) != size:
        raise ValueError("Cut diagonal has the wrong size.")
    state = cp.full(cuts.shape, 1 / math.sqrt(size), dtype=_dtype(cp, dtype))
    states = []
    costs = []
    for layer in range(int(layers)):
        cost_state = state * cp.exp(1j * angles[layer] * cuts)
        mixed = _mixer_apply_gpu(
            cost_state, float(angles[layers + layer]), n_vertices, representation
        )
        costs.append(cost_state)
        states.append(mixed)
        state = mixed
    return states, costs


def adjoint_mean_cut_gradient_gpu(
    cuts,
    angles,
    layers,
    *,
    n_vertices,
    representation,
    dtype="complex64",
):
    """Reverse-mode gradient of physical mean C for exact QAOA.

    Returns the gradient of ``-mean(C)``, ready for the ADAM loss path.
    Only the exact Mean Cut objective is supported; no measurements are made.
    """
    cp = _cupy()
    states, costs = forward_states_gpu(
        cuts,
        angles,
        layers,
        n_vertices=n_vertices,
        representation=representation,
        dtype=dtype,
    )
    gradient = np.zeros(2 * int(layers), dtype=float)
    adjoint = cuts * states[-1]
    for layer in range(int(layers) - 1, -1, -1):
        beta = float(angles[int(layers) + layer])
        dgamma = 1j * _mixer_apply_gpu(
            costs[layer] * cuts, beta, n_vertices, representation
        )
        dbeta = -1j * _x_sum_apply_gpu(states[layer], n_vertices, representation)
        gradient[layer] = -2.0 * float(
            cp.asnumpy(cp.real(cp.sum(cp.conj(adjoint) * dgamma)))
        )
        gradient[int(layers) + layer] = -2.0 * float(
            cp.asnumpy(cp.real(cp.sum(cp.conj(adjoint) * dbeta)))
        )
        adjoint = cp.exp(-1j * float(angles[layer]) * cuts) * _mixer_apply_gpu(
            adjoint, -beta, n_vertices, representation
        )
    return gradient



def restore_full_gpu(chi, n_vertices):
    """Expand normalized pair amplitudes chi to the physical full psi."""
    cp = _cupy()
    return cp.concatenate([chi, chi[::-1]]) / math.sqrt(2.0)


def probabilities_gpu(state, representation, n_vertices):
    """Return full P or pair masses q with an explicit representation."""
    cp = _cupy()
    if representation == "full":
        return cp.abs(state) ** 2
    if representation == "spin_flip_pairs":
        return cp.abs(state) ** 2
    raise ValueError("Unknown state representation.")


def _bits_to_spins(indices, n_vertices, representation):
    cp = _cupy()
    result = []
    for qubit in range(int(n_vertices)):
        if representation == "spin_flip_pairs" and qubit == n_vertices - 1:
            result.append(cp.ones_like(indices, dtype=cp.float64))
        else:
            result.append(1.0 - 2.0 * ((indices >> cp.uint64(qubit)) & cp.uint64(1)))
    return result


def correlations_gpu(masses, n_vertices, representation):
    """Compute the full physical ZZ matrix from P or q."""
    cp = _cupy()
    size = 1 << max(0, int(n_vertices) - 1) if representation == "spin_flip_pairs" else 1 << int(n_vertices)
    if masses.shape != (size,):
        raise ValueError("Probability mass vector has the wrong size.")
    indices = cp.arange(size, dtype=cp.uint64)
    spins = _bits_to_spins(indices, n_vertices, representation)
    matrix = cp.eye(n_vertices, dtype=cp.float64)
    for i in range(int(n_vertices)):
        for j in range(i + 1, int(n_vertices)):
            value = cp.sum(masses * spins[i] * spins[j])
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def objective_value_gpu(cuts, masses, config):
    """Evaluate one of the five criteria on a GPU distribution."""
    cp = _cupy()
    cuts = cp.asarray(cuts)
    masses = cp.asarray(masses)
    kind = config["type"]
    pars = config["parameters"]
    if kind == "mean_cut":
        return float(cp.asnumpy(cp.sum(cuts * masses)))
    if kind == "threshold_probability":
        return float(cp.asnumpy(cp.sum(masses[cuts >= pars["threshold"]])))
    if kind == "gibbs":
        keep = masses > 0
        logs = cp.log(masses[keep]) + pars["eta"] * cuts[keep]
        maximum = cp.max(logs)
        value = maximum + cp.log(cp.sum(cp.exp(logs - maximum)))
        return float(cp.asnumpy(value / pars["eta"]))
    order = cp.argsort(cuts)
    if kind == "cvar":
        order = order[::-1]
        values = cuts[order]
        weights = masses[order]
        before = cp.cumsum(weights) - weights
        selected = cp.minimum(weights, cp.maximum(pars["alpha"] - before, 0))
        return float(cp.asnumpy(cp.sum(values * selected) / pars["alpha"]))
    if kind == "best_of_k":
        cdf = cp.clip(cp.cumsum(masses[order]), 0, 1)
        cdf[-1] = 1.0
        maximum_mass = cp.diff(cp.concatenate([cp.array([0.0], dtype=cp.float64), cdf ** pars["k"]]))
        return float(cp.asnumpy(cp.sum(cuts[order] * maximum_mass)))
    raise ValueError(f"Unknown objective: {kind}")


def objective_from_samples_gpu(sample_cuts, config):
    """Evaluate a criterion from M actual cut samples on GPU."""
    cp = _cupy()
    kind = config["type"]
    pars = config["parameters"]
    count = int(sample_cuts.size)
    if count == 0:
        raise ValueError("At least one sample is required.")
    if kind == "mean_cut":
        return float(cp.asnumpy(cp.mean(sample_cuts)))
    if kind == "threshold_probability":
        return float(cp.asnumpy(cp.sum(sample_cuts >= pars["threshold"]) / count))
    if kind == "gibbs":
        scaled = pars["eta"] * sample_cuts
        maximum = cp.max(scaled)
        return float(cp.asnumpy(maximum + cp.log(cp.mean(cp.exp(scaled - maximum))) / pars["eta"]))
    values, counts = cp.unique(sample_cuts, return_counts=True)
    masses = counts.astype(cp.float64) / count
    return objective_value_gpu(values, masses, config)


def sample_full_indices_gpu(masses, count, *, n_vertices, representation, rng):
    """Draw M physical full indices on GPU from P or pair masses q."""
    cp = _cupy()
    if count < 0:
        raise ValueError("count must be nonnegative.")
    if count == 0:
        return cp.empty(0, dtype=cp.uint64)
    cdf = cp.cumsum(masses)
    if not bool(cp.isfinite(cdf[-1])):
        raise ArithmeticError("Probability CDF is not finite.")
    cdf /= cdf[-1]
    cdf[-1] = 1.0
    uniforms = rng.random_sample(size=int(count)).astype(cp.float64)
    indices = cp.searchsorted(cdf, uniforms, side="right")
    indices = cp.minimum(indices, masses.size - 1).astype(cp.uint64)
    if representation == "spin_flip_pairs":
        orientations = rng.randint(0, 2, size=int(count)).astype(cp.uint8)
        mask = cp.uint64((1 << int(n_vertices)) - 1)
        return cp.where(orientations.astype(bool), mask - indices, indices)
    return indices


def train_statevector_gpu(
    cuts,
    layers,
    *,
    objective,
    estimation_mode,
    shots_per_evaluation,
    optimizer,
    initial_angles,
    dtype,
    n_vertices,
    representation,
    rng,
    budget,
    reserved_shots=0,
    collect_optimizer_history=False,
    collect_objective_history=False,
):
    """Run the ADAM/central-difference loop using CuPy arrays."""
    cp = _cupy()
    theta = cp.asarray(np.asarray(initial_angles, dtype=float), dtype=cp.float64).copy()
    if theta.shape != (2 * int(layers),) or not bool(cp.isfinite(theta).all()):
        raise ValueError("initial_angles must contain 2*layers finite values.")
    op = optimizer["parameters"]
    step_size = float(optimizer["gradient"]["parameters"].get("step", 0.0))
    shots_each = int(shots_per_evaluation) if estimation_mode == "shots" else 0
    best_theta = theta.copy()
    best_loss = None
    evaluations = 0
    gradient_evaluations = 0
    state_evaluations = 0
    shots = 0
    history = []
    evaluation_history = []
    moment = cp.zeros_like(theta)
    variance = cp.zeros_like(theta)
    reason = "max_steps"
    final_state = None
    final_masses = None
    generator = rng

    def evaluate(angles, kind, iteration):
        nonlocal evaluations, shots, best_theta, best_loss, state_evaluations
        state, masses = state_at_angles_gpu(
            cuts,
            cp.asnumpy(angles),
            int(layers),
            n_vertices=n_vertices,
            representation=representation,
            dtype=dtype,
        )
        state_evaluations += 1
        if shots_each:
            sampled = sample_full_indices_gpu(
                masses,
                shots_each,
                n_vertices=n_vertices,
                representation=representation,
                rng=generator,
            )
            if representation == "spin_flip_pairs":
                sampled_cuts = cuts[sampled]
                # Full index is not needed for C; the representative index is
                # recovered from the full index's lower bits.
                rep = cp.where(
                    sampled < (1 << int(n_vertices) - 1),
                    sampled,
                    ((1 << int(n_vertices)) - 1) - sampled,
                )
                sampled_cuts = cuts[rep]
            else:
                sampled_cuts = cuts[sampled]
            budget.spend(shots_each, reserved=reserved_shots)
            shots += shots_each
            value = objective_from_samples_gpu(sampled_cuts, objective)
        else:
            value = objective_value_gpu(cuts, masses, objective)
        loss = -value
        evaluations += 1
        if not math.isfinite(loss):
            raise ArithmeticError("The objective is not finite.")
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_theta = angles.copy()
        if collect_objective_history:
            evaluation_history.append(
                {
                    "evaluation_id": evaluations,
                    "optimizer_step": iteration,
                    "kind": kind,
                    "objective_value": float(value),
                    "optimizer_loss": float(loss),
                    "angles": [float(x) for x in cp.asnumpy(angles)],
                    "shots_used": int(shots),
                }
            )
        return loss

    for iteration in range(1, int(op["max_steps"]) + 1):
        needed = (2 * len(theta) + 1) * shots_each
        if not budget.can_spend(needed, reserved=reserved_shots):
            reason = "shot_budget"
            break
        gradient = cp.zeros_like(theta)
        if optimizer["gradient"]["type"] == "adjoint":
            loss = evaluate(theta, "central", iteration)
            gradient = cp.asarray(
                adjoint_mean_cut_gradient_gpu(
                    cuts,
                    cp.asnumpy(theta),
                    int(layers),
                    n_vertices=n_vertices,
                    representation=representation,
                    dtype=dtype,
                )
            )
            gradient_evaluations += 1
            state_evaluations += int(layers)
        else:
            for index in range(len(theta)):
                delta = cp.zeros_like(theta)
                delta[index] = step_size
                plus = evaluate(theta + delta, "gradient_plus", iteration)
                minus = evaluate(theta - delta, "gradient_minus", iteration)
                gradient[index] = (plus - minus) / (2 * step_size)
                gradient_evaluations += 2
            loss = evaluate(theta, "central", iteration)
        row = {
            "optimizer_step": iteration,
            "objective_evaluations": evaluations,
            "gradient_evaluations": gradient_evaluations,
            "objective_value": float(-loss),
            "optimizer_loss": float(loss),
            "gradient_norm": float(cp.asnumpy(cp.linalg.norm(gradient))),
            "angles": [float(x) for x in cp.asnumpy(theta)],
            "shots_used": int(shots),
        }
        history.append(row)
        if op["gradient_tolerance"] and row["gradient_norm"] <= op["gradient_tolerance"]:
            reason = "gradient_tolerance"
            break
        moment = op["beta1"] * moment + (1 - op["beta1"]) * gradient
        variance = op["beta2"] * variance + (1 - op["beta2"]) * gradient**2
        corrected_m = moment / (1 - op["beta1"] ** iteration)
        corrected_v = variance / (1 - op["beta2"] ** iteration)
        theta -= op["learning_rate"] * corrected_m / (cp.sqrt(corrected_v) + op["epsilon"])

    final_state, final_masses = state_at_angles_gpu(
        cuts,
        cp.asnumpy(best_theta),
        int(layers),
        n_vertices=n_vertices,
        representation=representation,
        dtype=dtype,
    )
    return {
        "angles": cp.asnumpy(best_theta),
        "state": final_state,
        "masses": final_masses,
        "best_loss": best_loss,
        "objective_value": None if best_loss is None else -best_loss,
        "optimizer_history": history if collect_optimizer_history else None,
        "objective_history": evaluation_history if collect_objective_history else None,
        "evaluations": evaluations,
        "gradient_evaluations": gradient_evaluations,
        "state_evaluations": state_evaluations,
        "shots": shots,
        "optimizer_steps": len(history),
        "reason": reason,
        "initial_angles": np.asarray(initial_angles, dtype=float).copy(),
    }
