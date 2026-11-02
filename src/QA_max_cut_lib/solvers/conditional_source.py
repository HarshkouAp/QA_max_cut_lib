"""Correlation source for one fixed-spin conditional Ising block."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
import math

import numpy as np

from ..core.budget import ShotBudget, ShotBudgetExceeded
from .components import _validate_initialization, validate_adam_config
from .conditional import ConditionalProblemError
from .objectives import validate_objective
from .qaoa_training import train_statevector

__all__ = ["ConditionalCorrelationSource"]


class ConditionalCorrelationSource:
    """Train one conditional block and return physical ZZ correlations."""

    type = "qaoa_ising_statevector_cpu"

    def __init__(self, config, *, seed=None):
        config = deepcopy(config)
        if set(config) - {
            "type",
            "parameters",
            "objective",
            "objective_reference",
            "estimation",
            "optimizer",
            "initialization",
            "step_initialization",
            "correlations",
        }:
            raise ConditionalProblemError("Unknown conditional source field.")
        if config.get("type") != self.type:
            raise ConditionalProblemError("Wrong conditional source type.")
        parameters = config.setdefault("parameters", {})
        parameters.setdefault("layers", 1)
        parameters.setdefault("max_vertices", 18)
        parameters.setdefault("dtype", "complex128")
        parameters["layers"] = int(parameters["layers"])
        parameters["max_vertices"] = int(parameters["max_vertices"])
        if parameters["dtype"] not in ("complex64", "complex128"):
            raise ConditionalProblemError("dtype must be complex64 or complex128.")
        if parameters["max_vertices"] > 24:
            raise ConditionalProblemError("Full-state cap is 24 active vertices.")
        config["objective"] = validate_objective(config.get("objective", {}))
        config.setdefault("objective_reference", "conditional_affected_cut")
        if config["objective_reference"] not in (
            "conditional_affected_cut",
            "conditional_working_graph",
            "original_graph",
        ):
            raise ConditionalProblemError("Unknown conditional objective_reference.")
        estimation = config.setdefault("estimation", {})
        estimation.setdefault("mode", "exact")
        estimation.setdefault("shots_per_evaluation", 32)
        if estimation["mode"] not in ("exact", "shots"):
            raise ConditionalProblemError("Unknown estimation mode.")
        config["optimizer"] = validate_adam_config(config.get("optimizer", {}))
        _validate_initialization(config, parameters["layers"])
        if config.get("step_initialization", "restart") != "restart":
            raise ConditionalProblemError("Conditional source requires restart.")
        correlations = config.setdefault("correlations", {})
        correlations.setdefault("mode", "exact")
        if correlations["mode"] not in ("exact", "shots"):
            raise ConditionalProblemError("Unknown correlation mode.")
        if correlations["mode"] == "shots":
            correlations.setdefault("shots", 16)
            if int(correlations["shots"]) < 1:
                raise ConditionalProblemError("correlation shots must be positive.")
        elif "shots" in correlations:
            raise ConditionalProblemError("Exact correlations do not take shots.")
        self._config = config
        self._rng = np.random.default_rng(seed)

    @property
    def config(self):
        return deepcopy(self._config)

    @property
    def layers(self):
        return int(self._config["parameters"]["layers"])

    def _initial_angles(self):
        init = self._config["initialization"]
        if init["type"] == "fixed_angles":
            return np.asarray(init["parameters"]["values"], dtype=float)
        return np.r_[
            self._rng.uniform(-np.pi, np.pi, self.layers),
            self._rng.uniform(-np.pi / 2, np.pi / 2, self.layers),
        ]

    def estimate_problem(self, problem, *, budget, pair_domain="edges"):
        """Train and estimate correlations for one ``ConditionalCutProblem``."""
        if len(problem.nodes) > int(self._config["parameters"]["max_vertices"]):
            raise ConditionalProblemError("Conditional block exceeds max_vertices.")
        ref = self._config["objective_reference"]
        score = {
            "conditional_affected_cut": problem.affected_cuts,
            "conditional_working_graph": problem.working_cuts,
            "original_graph": problem.original_cuts,
        }[ref]
        corr_mode = self._config["correlations"]["mode"]
        corr_shots = (
            int(self._config["correlations"]["shots"]) if corr_mode == "shots" else 0
        )
        if budget.available() is not None and budget.available() < corr_shots:
            raise ShotBudgetExceeded(corr_shots, budget.available())
        training = train_statevector(
            problem.affected_cuts,
            self.layers,
            objective=self._config["objective"],
            estimation_mode=self._config["estimation"]["mode"],
            shots_per_evaluation=int(self._config["estimation"]["shots_per_evaluation"]),
            optimizer=self._config["optimizer"],
            initial_angles=self._initial_angles(),
            dtype=self._config["parameters"]["dtype"],
            rng=self._rng,
            budget=budget,
            reserved_shots=corr_shots,
            score_cuts=score,
        )
        probabilities = training["probabilities"]
        nodes = tuple(problem.nodes)
        if corr_mode == "shots":
            indices = self._rng.choice(
                len(probabilities), size=corr_shots, p=probabilities
            ).astype(np.uint64)
            shifts = np.arange(len(nodes), dtype=np.uint64)
            bits = ((indices[:, None] >> shifts[None, :]) & np.uint64(1)).astype(float)
            spins = 1.0 - 2.0 * bits
            budget.spend(corr_shots)
            matrix = (spins.T @ spins) / corr_shots
        else:
            from ..tools.numerics import correlation_matrix

            matrix = correlation_matrix(probabilities)
        pairs = (
            tuple(
                (u, v)
                for u, v, _ in problem.graph.edges
                if u in nodes and v in nodes
            )
            if pair_domain == "edges"
            else tuple(combinations(nodes, 2))
        )
        positions = {node: i for i, node in enumerate(nodes)}
        values = {}
        for u, v in pairs:
            value = float(matrix[positions[u], positions[v]])
            if not math.isfinite(value) or abs(value) > 1.0 + 1e-9:
                raise ConditionalProblemError("Conditional correlation is invalid.")
            values[(int(u), int(v))] = min(1.0, max(-1.0, value))
        return {
            "values": values,
            "pairs": pairs,
            "mode": corr_mode,
            "angles": np.asarray(training["angles"], dtype=float),
            "layers": self.layers,
            "training": {
                "objective_value": training["objective_value"],
                "optimizer_history": training["optimizer_history"],
                "objective_history": training["objective_history"],
                "initialization": self._config["initialization"],
            },
            "resources": {
                "training_shots": int(training["shots"]),
                "correlation_shots": corr_shots,
                "objective_evaluations": int(training["evaluations"]),
                "optimizer_steps": int(training["optimizer_steps"]),
                "num_treads": 1,
            },
            "objective_reference": ref,
        }
