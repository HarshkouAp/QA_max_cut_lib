"""NVIDIA CUDA QAOA solver and correlation source implemented with CuPy."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
import math
from numbers import Real
import numpy as np

from ..core.budget import ShotBudget, ShotBudgetExceeded
from ..core.reduction import ReductionState
from .base import BaseSolver, positive_int
from .components import (
    CorrelationResult,
    _initial_angles,
    _validate_initialization,
    validate_adam_config,
)
from .gpu_state import (
    correlations_gpu,
    cut_diagonal_gpu,
    gpu_device_info,
    memory_preflight,
    objective_value_gpu,
    restore_full_gpu,
    sample_full_indices_gpu,
    state_at_angles_gpu,
    train_statevector_gpu,
    validate_gpu_config,
)
from .objectives import validate_objective

__all__ = [
    "QAOAGPU",
    "QAOAGPUCorrelationSource",
    "validate_gpu_source_config",
]

SOURCE_TYPE = "qaoa_statevector_gpu"


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _validate_source_fields(config):
    allowed = {
        "type",
        "parameters",
        "gpu",
        "objective",
        "objective_reference",
        "estimation",
        "optimizer",
        "initialization",
        "step_initialization",
        "correlations",
    }
    if set(config) - allowed:
        raise ValueError(f"Unknown GPU correlation-source fields: {set(config) - allowed}")
    if config.get("type") != SOURCE_TYPE:
        raise ValueError(f"source type must be {SOURCE_TYPE!r}.")
    parameters = config.setdefault("parameters", {})
    if set(parameters) - {"layers", "max_vertices", "dtype", "representation"}:
        raise ValueError("Unknown GPU source parameter.")
    parameters.setdefault("layers", 1)
    parameters.setdefault("max_vertices", 20)
    parameters.setdefault("dtype", "complex64")
    parameters.setdefault("representation", "spin_flip_pairs")
    positive_int(parameters["layers"], "layers")
    positive_int(parameters["max_vertices"], "max_vertices")
    if parameters["max_vertices"] > 30:
        raise ValueError("GPU source max_vertices cannot exceed 30.")
    if parameters["dtype"] not in ("complex64", "complex128"):
        raise ValueError("GPU source dtype must be complex64 or complex128.")
    if parameters["representation"] not in ("full", "spin_flip_pairs"):
        raise ValueError("GPU representation must be full or spin_flip_pairs.")
    config["gpu"] = validate_gpu_config(config.get("gpu"))
    config["objective"] = validate_objective(config.get("objective", {}))
    config.setdefault("objective_reference", "working_graph")
    if config["objective_reference"] not in ("working_graph", "original_graph"):
        raise ValueError("objective_reference must be working_graph or original_graph.")
    estimation = config.setdefault("estimation", {})
    if set(estimation) - {"mode", "shots_per_evaluation"}:
        raise ValueError("Unknown GPU estimation field.")
    estimation.setdefault("mode", "exact")
    estimation.setdefault("shots_per_evaluation", 256)
    if estimation["mode"] not in ("exact", "shots"):
        raise ValueError("estimation.mode must be exact or shots.")
    positive_int(estimation["shots_per_evaluation"], "shots_per_evaluation")
    config["optimizer"] = validate_adam_config(config.get("optimizer", {}), allow_adjoint=True)
    if config["optimizer"]["gradient"]["type"] == "adjoint":
        if config["objective"]["type"] != "mean_cut":
            raise ValueError("The GPU adjoint supports exact Mean Cut only.")
        if estimation["mode"] != "exact":
            raise ValueError("The GPU adjoint requires estimation.mode='exact'.")
    _validate_initialization(config, parameters["layers"])
    config.setdefault("step_initialization", "restart")
    if config["step_initialization"] != "restart":
        raise ValueError("GPU source supports step_initialization='restart' only.")
    correlations = config.setdefault("correlations", {})
    if set(correlations) - {"mode", "shots"}:
        raise ValueError("Unknown GPU correlations field.")
    correlations.setdefault("mode", "exact")
    if correlations["mode"] not in ("exact", "shots"):
        raise ValueError("correlations.mode must be exact or shots.")
    if correlations["mode"] == "shots":
        if "shots" not in correlations:
            raise ValueError("correlations.shots is required in shots mode.")
        positive_int(correlations["shots"], "correlations.shots")
    elif "shots" in correlations:
        raise ValueError("exact correlations do not take a shot count.")
    return config


def validate_gpu_source_config(config):
    """Resolve and validate a ``qaoa_statevector_gpu`` source section."""
    return _validate_source_fields(deepcopy(config or {}))


def _rep_index(full_indices, n_vertices):
    cp = __import__("cupy")
    mask = cp.uint64((1 << int(n_vertices)) - 1)
    return cp.where(
        full_indices < (1 << (int(n_vertices) - 1)),
        full_indices,
        mask - full_indices,
    ).astype(cp.uint64)


class QAOAGPUCorrelationSource:
    """Train GPU QAOA and return physical ZZ correlations.

    Parameters
    ----------
    config : dict
        The nested GPU correlation-source configuration.
    seed : int or None, optional
        Root seed for fixed/random angles and shot streams.
    """

    type = SOURCE_TYPE

    def __init__(self, config, *, seed=None):
        self._config = validate_gpu_source_config(config)
        self._seed = None if seed is None else int(seed)
        self._rng = np.random.default_rng(seed)
        self._cp_rng = None
        self._previous_angles = None

    @property
    def config(self):
        return deepcopy(self._config)

    @property
    def max_vertices(self):
        return self._config["parameters"]["max_vertices"]

    @property
    def layers(self):
        return self._config["parameters"]["layers"]

    def _angles_for_step(self):
        values, name = _initial_angles(self._config, self.layers)
        if values is None:
            values = np.r_[
                self._rng.uniform(-np.pi, np.pi, self.layers),
                self._rng.uniform(-np.pi / 2, np.pi / 2, self.layers),
            ]
        return values, name

    def _objective_for_state(self, state):
        objective = deepcopy(self._config["objective"])
        if self._config["objective_reference"] == "original_graph":
            if state.cut_offset is None:
                raise ValueError("original_graph objective needs a Max-Cut offset.")
            shift = float(state.cut_offset)
        else:
            shift = 0.0
        if objective["type"] == "threshold_probability":
            objective["parameters"]["threshold"] -= shift
        return objective, shift

    def estimate(
        self,
        state,
        *,
        budget,
        pair_domain="all_pairs",
        collect_optimizer_history=False,
        collect_objective_history=False,
    ):
        if not isinstance(state, ReductionState):
            raise TypeError("state must be a ReductionState.")
        if not isinstance(budget, ShotBudget):
            raise TypeError("budget must be a ShotBudget.")
        if pair_domain not in ("edges", "all_pairs"):
            raise ValueError("pair_domain must be edges or all_pairs.")
        n = len(state.current_nodes)
        if n < 1:
            raise ValueError("A GPU correlation table needs at least one node.")
        if n > self.max_vertices:
            raise ValueError(
                f"GPU source supports at most {self.max_vertices} local nodes."
            )
        cp = __import__("cupy")
        params = self._config["parameters"]
        representation = params["representation"]
        gpu = self._config["gpu"]
        plan = memory_preflight(
            gpu,
            n,
            representation,
            dtype=params["dtype"],
            layers=self.layers,
        )
        graph = state.current_graph()
        cuts = cut_diagonal_gpu(graph, representation, diagonal_dtype=gpu["diagonal_dtype"])
        objective, shift = self._objective_for_state(state)
        angles, initialization = self._angles_for_step()
        corr_cfg = self._config["correlations"]
        corr_shots = int(corr_cfg["shots"]) if corr_cfg["mode"] == "shots" else 0
        available = budget.available()
        if available is not None and available < corr_shots:
            raise ShotBudgetExceeded(corr_shots, available)
        if self._cp_rng is None:
            self._cp_rng = cp.random.RandomState(
                None if self._seed is None else int(self._seed)
            )
        training = train_statevector_gpu(
            cuts,
            self.layers,
            objective=objective,
            estimation_mode=self._config["estimation"]["mode"],
            shots_per_evaluation=self._config["estimation"]["shots_per_evaluation"],
            optimizer=self._config["optimizer"],
            initial_angles=angles,
            dtype=params["dtype"],
            n_vertices=n,
            representation=representation,
            rng=self._cp_rng,
            budget=budget,
            reserved_shots=corr_shots,
            collect_optimizer_history=collect_optimizer_history,
            collect_objective_history=collect_objective_history,
        )
        masses = training["masses"]
        if corr_cfg["mode"] == "shots":
            samples = sample_full_indices_gpu(
                masses,
                corr_shots,
                n_vertices=n,
                representation=representation,
                rng=self._cp_rng,
            )
            budget.spend(corr_shots)
            rep = self._rep_index_for(samples, n) if representation == "spin_flip_pairs" else samples
            values = self._sampled_correlations(
                rep, samples, n, representation, tuple(state.current_nodes)
            )
        else:
            matrix = correlations_gpu(masses, n, representation)
            node_order = tuple(state.current_nodes)
            values = {
                (int(u), int(v)): float(cp.asnumpy(matrix[i, j]))
                for i, u in enumerate(node_order)
                for j, v in enumerate(node_order)
                if i < j
            }
        all_pairs = [
            (min(int(u), int(v)), max(int(u), int(v)))
            for u, v in combinations(state.current_nodes, 2)
        ]
        values = {pair: values[pair] for pair in all_pairs if pair in values}
        corr_mode = corr_cfg["mode"]
        training_meta = {
            "initialization": initialization,
            "objective_value": training["objective_value"],
            "objective_reference": self._config["objective_reference"],
            "objective_shift": shift,
            "profile_type": None,
            "optimizer_history": training["optimizer_history"],
            "objective_history": training["objective_history"],
            "gpu_plan": plan,
            "representation": representation,
            "dtype": params["dtype"],
        }
        resources = {
            "shots_used": int(training["shots"]) + int(corr_shots),
            "training_shots": int(training["shots"]),
            "correlation_shots": int(corr_shots),
            "readout_shots": 0,
            "objective_evaluations": int(training["evaluations"]),
            "gradient_evaluations": int(training["gradient_evaluations"]),
            "state_evaluations": int(training["state_evaluations"]),
            "optimizer_steps": int(training["optimizer_steps"]),
            "source_calls": 1,
            "gpu_plan": plan,
        }
        return CorrelationResult(
            node_order=tuple(state.current_nodes),
            task_version=int(state.version),
            values=values,
            pairs=tuple(all_pairs),
            mode=corr_mode,
            angles=np.asarray(training["angles"], dtype=float),
            layers=int(self.layers),
            training=training_meta,
            resources=resources,
            objective_reference=self._config["objective_reference"],
        )

    @property
    def _seed_value(self):
        return self._seed

    def _rep_index_for(self, full_indices, n):
        cp = __import__("cupy")
        mask = cp.uint64((1 << int(n)) - 1)
        return cp.where(
            full_indices < (1 << (int(n) - 1)),
            full_indices,
            mask - full_indices,
        ).astype(cp.uint64)

    def _sampled_correlations(self, rep, full, n, representation, node_order):
        cp = __import__("cupy")
        bits = []
        for qubit in range(n):
            bits.append(1.0 - 2.0 * ((full >> cp.uint64(qubit)) & cp.uint64(1)))
        result = {}
        for i, u in enumerate(node_order):
            for j in range(i + 1, n):
                result[(int(u), int(node_order[j]))] = float(
                    cp.asnumpy(cp.mean(bits[i] * bits[j]))
                )
        return result


class QAOAGPU(BaseSolver):
    """Train a CuPy GPU QAOA and produce a complete spin solution."""

    algorithm_type = "qaoa_gpu"
    extra_config_fields = ("gpu",)
    optional_fields = BaseSolver.optional_fields + (
        "state_representation",
        "compressed_statevector",
        "pair_probabilities",
        "gpu_report",
    )

    def _validate(self, config, graph):
        p = config.setdefault("parameters", {})
        if set(p) - {"layers", "max_vertices", "dtype", "representation"}:
            raise ValueError("Unknown GPU QAOA parameter.")
        p.setdefault("layers", 1)
        p.setdefault("max_vertices", 30)
        p.setdefault("dtype", "complex64")
        p.setdefault("representation", "spin_flip_pairs")
        positive_int(p["layers"], "layers")
        positive_int(p["max_vertices"], "max_vertices")
        if p["max_vertices"] > 30:
            raise ValueError("GPU max_vertices cannot exceed 30.")
        if p["dtype"] not in ("complex64", "complex128"):
            raise ValueError("dtype must be complex64 or complex128.")
        if p["representation"] not in ("full", "spin_flip_pairs"):
            raise ValueError("representation must be full or spin_flip_pairs.")
        if not 1 <= len(graph) <= p["max_vertices"]:
            raise ValueError("GPU QAOA requires 1..max_vertices nodes.")
        config["gpu"] = validate_gpu_config(config.get("gpu"))
        config["objective"] = validate_objective(config.get("objective", {}))
        estimation = config.setdefault("estimation", {})
        if set(estimation) - {"mode", "shots_per_evaluation"}:
            raise ValueError("Unknown estimation field.")
        estimation.setdefault("mode", "exact")
        estimation.setdefault("shots_per_evaluation", 256)
        if estimation["mode"] not in ("exact", "shots"):
            raise ValueError("estimation.mode must be exact or shots.")
        positive_int(estimation["shots_per_evaluation"], "shots_per_evaluation")
        config["optimizer"] = validate_adam_config(config.get("optimizer", {}), allow_adjoint=True)
        gradient = config["optimizer"]["gradient"]["type"]
        if gradient == "adjoint":
            if config["objective"]["type"] != "mean_cut":
                raise ValueError("The GPU adjoint supports exact Mean Cut only.")
            if estimation["mode"] != "exact":
                raise ValueError("The GPU adjoint requires estimation.mode='exact'.")
        _validate_initialization(config, p["layers"])
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
            raise ValueError("Unknown GPU limit.")
        limits.setdefault("shots_total", None)
        if limits["shots_total"] is not None:
            positive_int(limits["shots_total"], "shots_total", minimum=0)
            if readout.get("shots", 0) > limits["shots_total"]:
                raise ValueError("The total shot budget cannot cover final readout.")
        collect = config.setdefault("collect", {})
        allowed = {
            "optimizer_history",
            "objective_history",
            "statevector",
            "probabilities",
            "correlations",
            "compressed_statevector",
            "pair_probabilities",
        }
        if set(collect) - allowed:
            raise ValueError("Unknown GPU collect field.")
        for key in allowed:
            collect.setdefault(key, False)
            if type(collect[key]) is not bool:
                raise ValueError(f"collect.{key} must be boolean.")
        if p["representation"] == "spin_flip_pairs" and collect["statevector"]:
            raise ValueError(
                "collect.statevector is a full physical state request; use "
                "collect.compressed_statevector for pair mode."
            )
        if p["representation"] == "spin_flip_pairs" and collect["probabilities"]:
            raise ValueError(
                "collect.probabilities is a full physical probability request; "
                "use collect.pair_probabilities for pair mode."
            )
        memory_preflight(
            config["gpu"],
            len(graph),
            p["representation"],
            dtype=p["dtype"],
            layers=p["layers"],
        )

    def _run(self, graph, config, rng):
        cp = __import__("cupy")
        p = config["parameters"]
        n = len(graph)
        representation = p["representation"]
        gpu = config["gpu"]
        plan = memory_preflight(
            gpu, n, representation, dtype=p["dtype"], layers=p["layers"]
        )
        cuts = cut_diagonal_gpu(graph, representation, diagonal_dtype=gpu["diagonal_dtype"])
        if config["initialization"]["type"] == "fixed_angles":
            theta = np.asarray(
                config["initialization"]["parameters"]["values"], dtype=float
            )
        else:
            theta = np.r_[
                rng.uniform(-np.pi, np.pi, p["layers"]),
                rng.uniform(-np.pi / 2, np.pi / 2, p["layers"]),
            ]
        budget = ShotBudget(config["limits"]["shots_total"])
        reserved = int(config["readout"].get("shots", 0))
        cp_rng = cp.random.RandomState(
            None if config.get("seed") is None else int(config["seed"])
        )
        training = train_statevector_gpu(
            cuts,
            p["layers"],
            objective=config["objective"],
            estimation_mode=config["estimation"]["mode"],
            shots_per_evaluation=config["estimation"]["shots_per_evaluation"],
            optimizer=config["optimizer"],
            initial_angles=theta,
            dtype=p["dtype"],
            n_vertices=n,
            representation=representation,
            rng=cp_rng,
            budget=budget,
            reserved_shots=reserved,
            collect_optimizer_history=config["collect"]["optimizer_history"],
            collect_objective_history=config["collect"]["objective_history"],
        )
        masses = training["masses"]
        if reserved:
            samples = sample_full_indices_gpu(
                masses,
                reserved,
                n_vertices=n,
                representation=representation,
                rng=cp_rng,
            )
            if representation == "spin_flip_pairs":
                rep = _rep_index(samples, n)
                sample_cuts = cuts[rep]
            else:
                sample_cuts = cuts[samples]
            chosen = int(cp.asnumpy(samples[int(cp.argmax(sample_cuts))]))
            budget.spend(reserved)
        else:
            chosen = int(cp.asnumpy(cp.argmax(masses)))
        solution = {
            int(node): int(1 - 2 * ((chosen >> i) & 1))
            for i, node in enumerate(graph.nodes)
        }
        expected_cut = float(cp.asnumpy(cp.sum(cuts * masses)))
        correlations = None
        if config["collect"]["correlations"]:
            correlations = cp.asnumpy(correlations_gpu(masses, n, representation))
        state_representation = {
            "type": representation,
            "n_vertices": n,
            "node_order": [int(node) for node in graph.nodes],
            "dtype": p["dtype"],
            "amplitude_kind": (
                "physical_psi"
                if representation == "full"
                else "normalized_pair_chi"
            ),
            "probability_kind": (
                "physical_P"
                if representation == "full"
                else "representative_pair_q"
            ),
            "representative_rule": (
                None
                if representation == "full"
                else "last_node_order_bit_zero"
            ),
            "normalization": (
                "sum_abs_psi_squared=1"
                if representation == "full"
                else "sum_abs_chi_squared=1; P(r)=P(bar_r)=q_r/2"
            ),
            "device": gpu_device_info(gpu["device_id"]),
        }
        statevector = None
        probabilities = None
        compressed = None
        pair_probabilities = None
        if config["collect"]["statevector"]:
            statevector = cp.asnumpy(training["state"])
        if config["collect"]["probabilities"]:
            probabilities = (
                cp.asnumpy(training["masses"])
                if representation == "full"
                else None
            )
        if config["collect"]["compressed_statevector"]:
            compressed = (
                cp.asnumpy(training["state"])
                if representation == "spin_flip_pairs"
                else None
            )
        if config["collect"]["pair_probabilities"]:
            pair_probabilities = (
                cp.asnumpy(training["masses"])
                if representation == "spin_flip_pairs"
                else None
            )
        return {
            "solution": solution,
            "status": "stopped" if training["reason"] == "shot_budget" else "completed",
            "termination_reason": training["reason"],
            "expected_cut": expected_cut,
            "expected_energy": -expected_cut,
            "expected_cut_source": f"gpu_{representation}_final_masses",
            "objective_value": training["objective_value"],
            "optimizer_loss": training["best_loss"],
            "objective_source": (
                "best_training_evaluation"
                if training["best_loss"] is not None
                else "not_evaluated"
            ),
            "optimal_angles": np.asarray(training["angles"], dtype=float),
            "optimizer_history": training["optimizer_history"],
            "objective_history": training["objective_history"],
            "statevector": statevector,
            "probabilities": probabilities,
            "compressed_statevector": compressed,
            "pair_probabilities": pair_probabilities,
            "correlations": correlations,
            "state_representation": state_representation,
            "gpu_report": {
                "device": gpu_device_info(gpu["device_id"]),
                "memory_plan": plan,
                "kernels": deepcopy(gpu["kernels"]),
                "representation": representation,
                "shots": budget.used,
            },
            "resources": {
                "shots_used": int(budget.used),
                "training_shots": int(training["shots"]),
                "readout_shots": int(reserved),
                "objective_evaluations": int(training["evaluations"]),
                "gradient_evaluations": int(training["gradient_evaluations"]),
                "state_evaluations": int(training["state_evaluations"]),
                "optimizer_steps": int(training["optimizer_steps"]),
                "num_stages": int(training["optimizer_steps"]) + 1,
                "num_treads": 1,
                "backend": "cupy",
                "gpu_plan": plan,
            },
            "data_status": {
                key: ("collected" if value else "not_collected")
                for key, value in config["collect"].items()
            },
        }


def main(argv=None):
    """Run a small GPU QAOA example through the common demo interface."""
    from ..tools.demo import run_solver_demo

    return run_solver_demo(QAOAGPU, argv)


if __name__ == "__main__":
    raise SystemExit(main())
