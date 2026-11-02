"""Exact fixed-spin conditioning of a weighted Max-Cut problem."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from typing import Mapping
import numpy as np

from ..core.graph import WeightedGraph

__all__ = [
    "ConditionalProblemError",
    "ConditionalCutProblem",
    "build_conditional_problem",
    "validate_reference",
]


class ConditionalProblemError(ValueError):
    """Raised when a conditional block or reference is invalid."""


def _spin(value, name):
    if isinstance(value, bool) or value not in (-1, 1):
        raise ConditionalProblemError(f"{name} must be integer +1 or -1.")
    return int(value)


def validate_reference(graph, reference):
    """Validate a complete integer spin assignment on the current graph."""
    if not isinstance(reference, Mapping):
        raise ConditionalProblemError("reference must be a mapping.")
    if set(reference) != set(graph.nodes):
        missing = sorted(set(graph.nodes) - set(reference))
        extra = sorted(set(reference) - set(graph.nodes))
        raise ConditionalProblemError(
            f"reference does not cover the graph; missing={missing}, extra={extra}."
        )
    out = {}
    for node in graph.nodes:
        out[int(node)] = _spin(reference[node], f"reference[{node}]")
    return out


@dataclass(frozen=True)
class ConditionalCutProblem:
    """One local Ising problem with fixed external spins.

    Attributes
    ----------
    nodes : tuple of int
        Active node order.  Bit ``q`` in an active index follows this order.
    reference : dict
        Full assignment on the current graph, including the active values.
    couplings : dict
        ``(u, v) -> J_uv`` for internal active edges.
    fields : dict
        ``i -> h_i`` from crossing edges.
    constant : float
        Local Ising constant ``c_aff``.
    outside_cut : float
        Cut of the fixed external subgraph, ``C_TT``.
    cut_offset : float
        Global reduction offset ``b_t``.
    affected_cuts : ndarray
        ``A_S(x; r_T)`` for active basis indices.
    working_cuts : ndarray
        ``C_TT + A_S``.
    original_cuts : ndarray
        ``b_t + C_TT + A_S``.
    """

    graph: WeightedGraph
    nodes: tuple
    reference: dict
    couplings: dict
    fields: dict
    constant: float
    outside_cut: float
    cut_offset: float
    affected_cuts: np.ndarray = field(init=False, repr=False)
    working_cuts: np.ndarray = field(init=False, repr=False)
    original_cuts: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        n = len(self.nodes)
        if len(set(self.nodes)) != n or not set(self.nodes) <= set(self.graph.nodes):
            raise ConditionalProblemError("Active nodes must be unique graph labels.")
        if tuple(self.nodes) != tuple(node for node in self.graph.nodes if node in set(self.nodes)):
            raise ConditionalProblemError("Active order must follow graph node order.")
        if set(self.reference) != set(self.graph.nodes):
            raise ConditionalProblemError("The reference must cover the full graph.")
        if not math.isfinite(self.constant) or not math.isfinite(self.outside_cut):
            raise ConditionalProblemError("Conditional constants must be finite.")
        object.__setattr__(self, "affected_cuts", self._build_cuts())
        object.__setattr__(self, "working_cuts", self.outside_cut + self.affected_cuts)
        object.__setattr__(
            self, "original_cuts", self.cut_offset + self.outside_cut + self.affected_cuts
        )

    @property
    def active_nodes(self):
        """Return active external labels in state-vector order."""
        return self.nodes

    @property
    def outside_nodes(self):
        """Return fixed external labels in graph order."""
        active = set(self.nodes)
        return tuple(node for node in self.graph.nodes if node not in active)

    @property
    def full_reference(self):
        """Return the full reference assignment."""
        return dict(self.reference)

    def _build_cuts(self):
        n = len(self.nodes)
        size = 1 << n
        indices = np.arange(size, dtype=np.uint64)
        spins = np.empty((size, n), dtype=float)
        for i in range(n):
            spins[:, i] = 1.0 - 2.0 * ((indices >> np.uint64(i)) & np.uint64(1))
        # A_S = -c - sum h_i x_i - sum J_ik x_i x_k.
        cuts = np.full(size, -float(self.constant), dtype=float)
        for node, value in self.fields.items():
            i = self.nodes.index(node)
            cuts -= float(value) * spins[:, i]
        for (u, v), value in self.couplings.items():
            i = self.nodes.index(u)
            j = self.nodes.index(v)
            cuts -= float(value) * spins[:, i] * spins[:, j]
        if not np.isfinite(cuts).all():
            raise ConditionalProblemError("Conditional cut diagonal is not finite.")
        return cuts

    def completion(self, index):
        """Return the full assignment for one active little-endian index."""
        index = int(index)
        if index < 0 or index >= len(self.affected_cuts):
            raise IndexError("Active index is outside the conditional support.")
        result = dict(self.reference)
        for i, node in enumerate(self.nodes):
            result[node] = 1 - 2 * ((index >> i) & 1)
        return result

    def active_spins(self, index):
        """Return active spins by node order for one index."""
        result = {}
        for i, node in enumerate(self.nodes):
            result[node] = 1 - 2 * ((int(index) >> i) & 1)
        return result

    def to_dict(self):
        """Return a compact JSON-compatible problem record."""
        return {
            "schema_version": 1,
            "nodes": list(self.nodes),
            "outside_nodes": list(self.outside_nodes),
            "reference": {str(node): int(spin) for node, spin in self.reference.items()},
            "couplings": [
                [int(u), int(v), float(value)] for (u, v), value in sorted(self.couplings.items())
            ],
            "fields": [[int(node), float(value)] for node, value in sorted(self.fields.items())],
            "constant": float(self.constant),
            "outside_cut": float(self.outside_cut),
            "cut_offset": float(self.cut_offset),
            "graph_fingerprint": self.graph.snapshot()["fingerprint"],
        }


def build_conditional_problem(graph, active_vertices, reference, *, cut_offset=0.0):
    """Build ``A_S``, ``J``, ``h``, ``c_aff`` and ``C_TT``.

    The caller supplies the full reference on ``graph``.  Active labels are
    normalized to graph node order; the external assignment remains fixed.
    """
    if not isinstance(graph, WeightedGraph):
        raise TypeError("graph must be a WeightedGraph.")
    ref = validate_reference(graph, reference)
    if isinstance(active_vertices, (str, bytes)):
        raise ConditionalProblemError("active_vertices must be a sequence of labels.")
    active = []
    for value in active_vertices:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConditionalProblemError("Active labels must be integers.")
        if value not in graph.nodes:
            raise ConditionalProblemError(f"Active label is not in the graph: {value}")
        if value in active:
            raise ConditionalProblemError(f"Duplicate active label: {value}")
        active.append(value)
    active_set = set(active)
    active = [node for node in graph.nodes if node in active_set]
    internal = {}
    fields = {node: 0.0 for node in active}
    internal_sum = 0.0
    crossing_sum = 0.0
    outside_cut = 0.0
    for u, v, weight in graph.edges:
        weight = float(weight)
        in_u = u in active_set
        in_v = v in active_set
        if in_u and in_v:
            internal[(min(u, v), max(u, v))] = weight / 2.0
            internal_sum += weight
        elif in_u:
            fields[u] += 0.5 * weight * ref[v]
            crossing_sum += weight
        elif in_v:
            fields[v] += 0.5 * weight * ref[u]
            crossing_sum += weight
        else:
            if ref[u] != ref[v]:
                outside_cut += weight
    constant = -0.5 * (internal_sum + crossing_sum)
    if not all(math.isfinite(value) for value in (constant, outside_cut, cut_offset)):
        raise ConditionalProblemError("Conditional constants must be finite.")
    return ConditionalCutProblem(
        graph=graph.copy(),
        nodes=tuple(active),
        reference=ref,
        couplings=internal,
        fields=fields,
        constant=float(constant),
        outside_cut=float(outside_cut),
        cut_offset=float(cut_offset),
    )


class ConditionalCorrelationSource:
    """Train one conditional block and return physical ZZ correlations."""

    type = "qaoa_ising_statevector_cpu"

    def __init__(self, config, *, seed=None):
        from .components import _validate_initialization, validate_adam_config
        from .objectives import validate_objective
        from copy import deepcopy

        config = deepcopy(config)
        if set(config) - {
            "type", "parameters", "objective", "objective_reference",
            "estimation", "optimizer", "initialization", "step_initialization",
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
            "conditional_affected_cut", "conditional_working_graph", "original_graph"
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
        from copy import deepcopy
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
        from .qaoa_training import train_statevector
        from .qaoa_state import state_at_angles
        from ..tools.numerics import correlation_matrix
        from itertools import combinations
        import math

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
            int(self._config["correlations"]["shots"])
            if corr_mode == "shots"
            else 0
        )
        reserved = corr_shots
        if budget.available() is not None and budget.available() < reserved:
            from ..core.budget import ShotBudgetExceeded
            raise ShotBudgetExceeded(reserved, budget.available())
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
            reserved_shots=reserved,
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
            matrix = correlation_matrix(probabilities)
        pairs = (
            tuple((u, v) for u, v, _ in problem.graph.edges if u in nodes and v in nodes)
            if pair_domain == "edges"
            else tuple(combinations(nodes, 2))
        )
        pos = {node: i for i, node in enumerate(nodes)}
        values = {}
        for u, v in pairs:
            value = float(matrix[pos[u], pos[v]])
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
