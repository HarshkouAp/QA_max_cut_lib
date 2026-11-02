"""Analytic single-layer QAOA moments for weighted Max-Cut.

The module evaluates physical ZZ correlations and mean cut without a
``2**N`` state vector.  It supports the standard ``|+>`` start, the Ising
cost phase, the X mixer, and exactly one layer.  It intentionally does
not provide a full distribution or a discrete readout.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
import math
import time

import numpy as np

from ..core.budget import ShotBudget
from ..core.reduction import ReductionState
from .components import CorrelationResult, _positive_int, _mapping, _finite

__all__ = [
    "AnalyticP1Evaluator",
    "AnalyticP1CorrelationSource",
    "validate_analytic_p1_source_config",
]

_ANALYTIC_TYPE = "qaoa_analytic_p1_cpu"


def _as_gamma(value, name):
    value = _finite(value, name)
    return value


def _as_beta(value, name):
    value = _finite(value, name)
    return value


def _validate_gamma_bounds(bounds):
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError("gamma_bounds must be a two-element sequence.")
    low = _finite(bounds[0], "gamma_bounds[0]")
    high = _finite(bounds[1], "gamma_bounds[1]")
    if not low < high:
        raise ValueError("gamma_bounds must be strictly increasing.")
    return [low, high]


def validate_analytic_p1_source_config(config):
    """Resolve and validate one analytic p=1 source configuration.

    Parameters
    ----------
    config : dict
        The nested ``correlation_source`` section.

    Returns
    -------
    dict
        A new resolved section.  The source supports only ``mean_cut``,
        exact moments, exact correlations, and a finite gamma grid with
        exact beta minimization.
    """
    config = deepcopy(_mapping(config, "correlation_source"))
    allowed = {
        "type",
        "parameters",
        "objective",
        "objective_reference",
        "estimation",
        "optimizer",
        "correlations",
    }
    if set(config) - allowed:
        raise ValueError(f"Unknown analytic correlation-source fields: {set(config) - allowed}")
    if config.get("type") != _ANALYTIC_TYPE:
        raise ValueError(f"type must be {_ANALYTIC_TYPE}.")
    parameters = config.setdefault("parameters", {})
    if set(parameters) - {"layers", "dtype", "pair_batch_size"}:
        raise ValueError("Unknown analytic source parameter.")
    parameters.setdefault("layers", 1)
    parameters.setdefault("dtype", "float64")
    parameters.setdefault("pair_batch_size", 2048)
    if _positive_int(parameters["layers"], "layers") != 1:
        raise ValueError("The analytic source supports exactly one layer.")
    if parameters["dtype"] != "float64":
        raise ValueError("The analytic source supports dtype='float64' only.")
    _positive_int(parameters["pair_batch_size"], "pair_batch_size")

    objective = deepcopy(config.get("objective", {}))
    if objective.get("type", "mean_cut") != "mean_cut":
        raise ValueError("The analytic source supports only objective mean_cut.")
    if set(objective) - {"type", "parameters"}:
        raise ValueError("Unknown objective fields.")
    if objective.get("parameters", {}):
        raise ValueError("mean_cut has no parameters.")
    config["objective"] = {"type": "mean_cut", "parameters": {}}
    config.setdefault("objective_reference", "working_graph")
    if config["objective_reference"] not in ("working_graph", "original_graph"):
        raise ValueError("objective_reference must be working_graph or original_graph.")

    estimation = config.setdefault("estimation", {})
    if set(estimation) - {"mode"} or estimation.get("mode", "exact") != "exact":
        raise ValueError("The analytic source supports exact estimation only.")
    estimation["mode"] = "exact"

    optimizer = config.setdefault("optimizer", {})
    if set(optimizer) - {"type", "parameters"}:
        raise ValueError("Unknown analytic optimizer fields.")
    if optimizer.get("type") != "gamma_grid_exact_beta":
        raise ValueError("optimizer type must be gamma_grid_exact_beta.")
    op = optimizer.setdefault("parameters", {})
    if set(op) - {"gamma_bounds", "points", "gamma_batch_size"}:
        raise ValueError("Unknown gamma-grid parameter.")
    if "gamma_bounds" not in op or "points" not in op:
        raise ValueError("gamma_bounds and points are required.")
    op["gamma_bounds"] = _validate_gamma_bounds(op["gamma_bounds"])
    _positive_int(op["points"], "points", minimum=2)
    op.setdefault("gamma_batch_size", 4)
    _positive_int(op["gamma_batch_size"], "gamma_batch_size")

    correlations = config.setdefault("correlations", {})
    if set(correlations) - {"mode"} or correlations.get("mode", "exact") != "exact":
        raise ValueError("The analytic source supports exact correlations only.")
    correlations["mode"] = "exact"
    return config


class AnalyticP1Evaluator:
    """Evaluate weighted p=1 QAOA moments without a state vector.

    Parameters
    ----------
    graph : WeightedGraph
        The working graph.  Node order defines matrix indices.
    pair_batch_size : int, default 2048
        Maximum number of pairs processed together.  A batch allocates
        ``O(pair_batch_size * N)`` temporary floating-point values.
    dtype : {'float64'}, default 'float64'
        Only float64 is implemented.

    Attributes
    ----------
    node_order : tuple of int
        External labels in matrix order.
    weight_matrix : ndarray, shape (N, N)
        Symmetric dense weights with a zero diagonal.
    edge_count : int
        Number of nonzero edges.
    e0 : float
        ``-0.5 * sum(w)`` over existing nonzero edges.

    Notes
    -----
    The formulation uses ``E=-C``, ``J=w/2``, and the phase
    ``exp(+1j*gamma*C)``.  The formulas are those of the single-layer
    QAOA moment calculation in Ozaeta, van Dam, and McMahon (2022).
    The cost of a full correlation pass is O(K*N), where K is the number
    of requested pairs.  No object of size ``2**N`` is allocated.
    """

    def __init__(self, graph, *, pair_batch_size=2048, dtype="float64"):
        from ..core.graph import WeightedGraph

        if not isinstance(graph, WeightedGraph):
            raise TypeError("graph must be a WeightedGraph.")
        _positive_int(pair_batch_size, "pair_batch_size")
        if dtype != "float64":
            raise ValueError("Only dtype='float64' is implemented.")
        self.node_order = tuple(graph.nodes)
        self._index = {node: index for index, node in enumerate(self.node_order)}
        n = len(self.node_order)
        matrix = np.zeros((n, n), dtype=float)
        for u, v, weight in graph.edges:
            i, j = self._index[u], self._index[v]
            matrix[i, j] = matrix[j, i] = float(weight)
        self.weight_matrix = matrix
        self.pair_batch_size = int(pair_batch_size)
        self.dtype = dtype
        self.edge_indices = tuple(
            (i, j) for i in range(n) for j in range(i + 1, n) if matrix[i, j] != 0.0
        )
        self.edge_weights = np.asarray(
            [matrix[i, j] for i, j in self.edge_indices], dtype=float
        )
        self.edge_count = len(self.edge_indices)
        self.e0 = -0.5 * float(math.fsum(self.edge_weights.tolist()))
        if not math.isfinite(self.e0):
            raise ArithmeticError("The analytic E0 constant is not finite.")

    @property
    def n_nodes(self):
        """Return the number of active nodes."""
        return len(self.node_order)

    def _indices_for_pairs(self, pairs):
        pairs = tuple(pairs)
        i = np.asarray([self._index[u] for u, _ in pairs], dtype=np.int64)
        j = np.asarray([self._index[v] for _, v in pairs], dtype=np.int64)
        return i, j

    def _terms_for_indices(self, gamma, i_array, j_array):
        """Return a and d arrays for internal index arrays."""
        i_array = np.asarray(i_array, dtype=np.int64)
        j_array = np.asarray(j_array, dtype=np.int64)
        if i_array.shape != j_array.shape:
            raise ValueError("Pair index arrays must have the same shape.")
        gamma = _as_gamma(gamma, "gamma")
        n = self.n_nodes
        a_values = np.empty(len(i_array), dtype=float)
        d_values = np.empty(len(i_array), dtype=float)
        for start in range(0, len(i_array), self.pair_batch_size):
            stop = min(start + self.pair_batch_size, len(i_array))
            ii = i_array[start:stop]
            jj = j_array[start:stop]
            if np.any(ii == jj):
                raise ValueError("A correlation pair needs two different nodes.")
            rows = np.arange(len(ii), dtype=np.int64)
            u_weights = self.weight_matrix[ii, :]
            v_weights = self.weight_matrix[jj, :]
            mask = np.ones_like(u_weights, dtype=bool)
            mask[rows, ii] = False
            mask[rows, jj] = False
            with np.errstate(over="ignore", invalid="ignore"):
                cos_u = np.cos(gamma * u_weights)
                cos_v = np.cos(gamma * v_weights)
                cos_minus = np.cos(gamma * (u_weights - v_weights))
                cos_plus = np.cos(gamma * (u_weights + v_weights))
            for array in (cos_u, cos_v, cos_minus, cos_plus):
                if not np.isfinite(array).all():
                    raise ArithmeticError("A cosine product is not finite.")
            cos_u[~mask] = 1.0
            cos_v[~mask] = 1.0
            cos_minus[~mask] = 1.0
            cos_plus[~mask] = 1.0
            p_u = np.prod(cos_u, axis=1)
            p_v = np.prod(cos_v, axis=1)
            p_minus = np.prod(cos_minus, axis=1)
            p_plus = np.prod(cos_plus, axis=1)
            w_uv = self.weight_matrix[ii, jj]
            a = 0.5 * np.sin(gamma * w_uv) * (p_u + p_v)
            d = 0.5 * (p_minus - p_plus)
            if not np.isfinite(a).all() or not np.isfinite(d).all():
                raise ArithmeticError("An analytic pair term is not finite.")
            a_values[start:stop] = a
            d_values[start:stop] = d
        return a_values, d_values

    def pair_terms(self, gamma, pairs):
        """Return ``(a_uv, d_uv)`` for external node pairs."""
        pairs = tuple(pairs)
        i_array, j_array = self._indices_for_pairs(pairs)
        return self._terms_for_indices(gamma, i_array, j_array)

    def edge_coefficients(self, gamma):
        """Return ``A``, ``B``, and ``E0`` for all nonzero edges.

        The scan uses only existing edges.  It never prepares the full
        all-pairs table.
        """
        gamma = _as_gamma(gamma, "gamma")
        i_array = np.asarray([i for i, _ in self.edge_indices], dtype=np.int64)
        j_array = np.asarray([j for _, j in self.edge_indices], dtype=np.int64)
        if len(i_array):
            a_values, d_values = self._terms_for_indices(gamma, i_array, j_array)
            weights = self.edge_weights
            coefficient_a = 0.5 * float(math.fsum((weights * a_values).tolist()))
            coefficient_b = 0.5 * float(math.fsum((weights * d_values).tolist()))
        else:
            coefficient_a = 0.0
            coefficient_b = 0.0
        if not math.isfinite(coefficient_a) or not math.isfinite(coefficient_b):
            raise ArithmeticError("The analytic energy coefficients are not finite.")
        return {
            "gamma": gamma,
            "E0": self.e0,
            "A": coefficient_a,
            "B": coefficient_b,
        }

    def energy_min_beta(self, gamma):
        """Return coefficients and the exact minimum over beta.

        Parameters
        ----------
        gamma : float
            Fixed gamma in radians.

        Returns
        -------
        dict
            Keys ``gamma``, ``E0``, ``A``, ``B``, ``F0``, ``Fs``,
            ``Fc``, ``R``, ``energy_min``, and ``beta_star``.
        """
        coefficients = self.edge_coefficients(gamma)
        e0 = coefficients["E0"]
        a = coefficients["A"]
        b = coefficients["B"]
        f0 = e0 + 0.5 * b
        f_s = a
        f_c = -0.5 * b
        radius = float(math.hypot(f_s, f_c))
        beta = 0.0 if radius == 0.0 else 0.25 * math.atan2(-f_s, -f_c)
        beta %= math.pi / 2.0
        energy_min = f0 - radius
        if not math.isfinite(energy_min) or not math.isfinite(beta):
            raise ArithmeticError("The analytic beta minimum is not finite.")
        return {
            **coefficients,
            "F0": f0,
            "Fs": f_s,
            "Fc": f_c,
            "R": radius,
            "energy_min": energy_min,
            "beta_star": beta,
        }

    def expected_energy(self, gamma, beta):
        """Return the mean signed energy at fixed angles."""
        gamma = _as_gamma(gamma, "gamma")
        beta = _as_beta(beta, "beta")
        coefficients = self.edge_coefficients(gamma)
        return (
            coefficients["E0"]
            + coefficients["A"] * math.sin(4.0 * beta)
            + coefficients["B"] * math.sin(2.0 * beta) ** 2
        )

    def expected_cut(self, gamma, beta):
        """Return the mean cut value at fixed angles."""
        return -self.expected_energy(gamma, beta)

    def correlations(self, gamma, beta, *, pair_domain="all_pairs"):
        """Return all requested physical ZZ correlations.

        Parameters
        ----------
        gamma, beta : float
            Angles in radians.
        pair_domain : {'edges', 'all_pairs'}
            ``edges`` covers nonzero edges.  ``all_pairs`` covers every
            unordered pair of active nodes, including nonedges.

        Returns
        -------
        dict of (int, int) to float
            Values use the smaller external label first.  A zero value
            is a computed correlation, not a missing entry.
        """
        gamma = _as_gamma(gamma, "gamma")
        beta = _as_beta(beta, "beta")
        if pair_domain not in ("edges", "all_pairs"):
            raise ValueError("pair_domain must be edges or all_pairs.")
        if pair_domain == "edges":
            pairs = [
                (self.node_order[i], self.node_order[j]) for i, j in self.edge_indices
            ]
        else:
            pairs = list(combinations(sorted(self.node_order), 2))
        if not pairs:
            return {}
        ordered = [(min(u, v), max(u, v)) for u, v in pairs]
        i_array, j_array = self._indices_for_pairs(ordered)
        a_values, d_values = self._terms_for_indices(gamma, i_array, j_array)
        sin4 = math.sin(4.0 * beta)
        sin2_sq = math.sin(2.0 * beta) ** 2
        values = {}
        for pair, a_value, d_value in zip(ordered, a_values, d_values):
            value = float(a_value * sin4 + d_value * sin2_sq)
            if not math.isfinite(value):
                raise ArithmeticError("A physical correlation is not finite.")
            if abs(value) > 1.0 + 1e-9:
                raise ArithmeticError("A physical correlation is outside [-1, 1].")
            values[pair] = min(1.0, max(-1.0, value))
        return values

    def correlation_matrix(self, gamma, beta):
        """Return the full physical matrix including the unit diagonal."""
        n = self.n_nodes
        matrix = np.eye(n, dtype=float)
        values = self.correlations(gamma, beta, pair_domain="all_pairs")
        for (u, v), value in values.items():
            i, j = self._index[u], self._index[v]
            matrix[i, j] = matrix[j, i] = value
        return matrix

    def scan_gamma(self, *, gamma_bounds, points, gamma_batch_size=4):
        """Evaluate the exact-beta energy profile on a finite gamma grid.

        Parameters
        ----------
        gamma_bounds : sequence of two floats
            Increasing inclusive grid bounds.
        points : int
            Number of grid points, at least two.
        gamma_batch_size : int, default 4
            Number of gamma points in one planning batch.  The public
            arrays are still returned for the complete grid.

        Returns
        -------
        dict
            ``gamma_values``, ``beta_values``, ``profile_energy``,
            ``A_values``, ``B_values``, ``E0``, ``best_index``, and
            ``best_gamma``.
        """
        bounds = _validate_gamma_bounds(gamma_bounds)
        _positive_int(points, "points", minimum=2)
        _positive_int(gamma_batch_size, "gamma_batch_size")
        gamma_values = np.linspace(bounds[0], bounds[1], points, dtype=float)
        profile_energy = np.empty(points, dtype=float)
        beta_values = np.empty(points, dtype=float)
        a_values = np.empty(points, dtype=float)
        b_values = np.empty(points, dtype=float)
        for start in range(0, points, gamma_batch_size):
            stop = min(start + gamma_batch_size, points)
            for index in range(start, stop):
                result = self.energy_min_beta(float(gamma_values[index]))
                profile_energy[index] = result["energy_min"]
                beta_values[index] = result["beta_star"]
                a_values[index] = result["A"]
                b_values[index] = result["B"]
        if not (
            np.isfinite(profile_energy).all()
            and np.isfinite(beta_values).all()
            and np.isfinite(a_values).all()
            and np.isfinite(b_values).all()
        ):
            raise ArithmeticError("The analytic gamma profile is not finite.")
        best_index = int(np.argmin(profile_energy))
        return {
            "gamma_values": gamma_values,
            "beta_values": beta_values,
            "profile_energy": profile_energy,
            "A_values": a_values,
            "B_values": b_values,
            "E0": self.e0,
            "best_index": best_index,
            "best_gamma": float(gamma_values[best_index]),
        }


class AnalyticP1CorrelationSource:
    """Provide analytic p=1 correlations to the existing RQAOA.

    Parameters
    ----------
    config : dict
        The ``correlation_source`` section of an RQAOA configuration.
    seed : int or None, optional
        Unused.  The analytic grid is deterministic.  The argument is
        accepted for interchangeability with the state-vector source.

    Notes
    -----
    The source supports only weighted Max-Cut without fields, exactly
    one layer, exact Mean Cut, exact correlations, and a finite gamma
    grid.  It does not create a state vector and consumes no shots.
    """

    type = _ANALYTIC_TYPE

    def __init__(self, config, seed=None):
        self._config = validate_analytic_p1_source_config(config)
        self._seed = seed
        self._cache = None

    @property
    def config(self):
        """Return a copy of the resolved source configuration."""
        return deepcopy(self._config)

    @property
    def layers(self):
        """Return the fixed source depth, one."""
        return 1

    def _evaluator(self, state):
        graph = state.current_graph()
        key = (
            graph.snapshot()["fingerprint"],
            state.version,
            tuple(state.current_nodes),
            self._config["parameters"]["pair_batch_size"],
        )
        if self._cache is None or self._cache[0] != key:
            evaluator = AnalyticP1Evaluator(
                graph,
                pair_batch_size=self._config["parameters"]["pair_batch_size"],
                dtype=self._config["parameters"]["dtype"],
            )
            self._cache = (key, evaluator)
        return self._cache[1]

    def estimate(
        self,
        state,
        *,
        budget,
        pair_domain="all_pairs",
        collect_optimizer_history=False,
        collect_objective_history=False,
    ):
        """Search the gamma grid and return correlations at the best point.

        Parameters
        ----------
        state : ReductionState
            Current task and version.
        budget : ShotBudget
            Root budget.  It is validated but not spent.
        pair_domain : {'edges', 'all_pairs'}
            Coverage of the returned table.
        collect_optimizer_history, collect_objective_history : bool
            Accepted for interface compatibility.  The gamma profile is
            always returned as compact search metadata.

        Returns
        -------
        CorrelationResult
            Physical correlations, angles, coverage, profile, and
            classical resource counters.
        """
        if not isinstance(state, ReductionState):
            raise TypeError("state must be a ReductionState.")
        if not isinstance(budget, ShotBudget):
            raise TypeError("budget must be a ShotBudget.")
        if pair_domain not in ("edges", "all_pairs"):
            raise ValueError("pair_domain must be edges or all_pairs.")
        started = time.perf_counter()
        evaluator = self._evaluator(state)
        optimizer = self._config["optimizer"]["parameters"]
        scan = evaluator.scan_gamma(
            gamma_bounds=optimizer["gamma_bounds"],
            points=optimizer["points"],
            gamma_batch_size=optimizer["gamma_batch_size"],
        )
        best_index = scan["best_index"]
        gamma = float(scan["gamma_values"][best_index])
        beta = float(scan["beta_values"][best_index])
        profile_energy = float(scan["profile_energy"][best_index])
        checked = evaluator.expected_energy(gamma, beta)
        tolerance = 1e-9 * max(1.0, abs(profile_energy), abs(checked))
        if not math.isclose(checked, profile_energy, rel_tol=0.0, abs_tol=tolerance):
            raise ArithmeticError("The analytic profile and final angles disagree.")
        values = evaluator.correlations(gamma, beta, pair_domain=pair_domain)
        if pair_domain == "edges":
            pairs = tuple(
                (min(evaluator.node_order[i], evaluator.node_order[j]),
                 max(evaluator.node_order[i], evaluator.node_order[j]))
                for i, j in evaluator.edge_indices
            )
        else:
            pairs = tuple(combinations(sorted(evaluator.node_order), 2))
        shift = 0.0
        if self._config["objective_reference"] == "original_graph":
            if state.cut_offset is None:
                raise ValueError("original_graph objective needs a Max-Cut offset.")
            shift = float(state.cut_offset)
        cut_working = -profile_energy
        cut_original = cut_working + shift
        elapsed = time.perf_counter() - started
        gamma_values = [float(value) for value in scan["gamma_values"]]
        profile = {
            "type": "analytic_gamma_grid",
            "strategy": "gamma_grid_exact_beta",
            "gamma_bounds": list(optimizer["gamma_bounds"]),
            "points": int(optimizer["points"]),
            "gamma_values": gamma_values,
            "beta_star": [float(value) for value in scan["beta_values"]],
            "profile_energy": [float(value) for value in scan["profile_energy"]],
            "profile_energy_original": [
                float(value) - shift for value in scan["profile_energy"]
            ],
            "A": [float(value) for value in scan["A_values"]],
            "B": [float(value) for value in scan["B_values"]],
            "E0": float(evaluator.e0),
            "best_index": int(best_index),
            "best_gamma": gamma,
            "best_beta": beta,
            "task_version": int(state.version),
            "objective_reference": self._config["objective_reference"],
        }
        return CorrelationResult(
            node_order=tuple(evaluator.node_order),
            task_version=state.version,
            values=values,
            pairs=pairs,
            mode="exact",
            angles=np.asarray([gamma, beta], dtype=float),
            layers=1,
            training={
                "initialization": "gamma_grid_exact_beta",
                "reason": "best_grid",
                "objective_value": cut_working,
                "objective_value_original": cut_original,
                "objective_reference": self._config["objective_reference"],
                "objective_shift": shift,
                "objective_threshold_requested": None,
                "objective_threshold_effective": None,
                "expected_cut_working": cut_working,
                "expected_cut_original": cut_original,
                "optimizer_steps": 0,
                "objective_evaluations": 0,
                "training_shots": 0,
                "best_loss": profile_energy,
                "optimizer_history": None,
                "objective_history": None,
                "profile_type": "analytic_gamma_grid",
                "gamma_profile": profile,
                "classical_counters": {
                    "gamma_points": int(optimizer["points"]),
                    "coefficient_passes": int(optimizer["points"]),
                    "pair_correlators": int(len(values)),
                    "classical_seconds": float(elapsed),
                },
            },
            resources={
                "training_shots": 0,
                "correlation_shots": 0,
                "readout_shots": 0,
                "objective_evaluations": 0,
                "optimizer_steps": 0,
                "source_calls": 1,
                "gamma_points": int(optimizer["points"]),
                "coefficient_passes": int(optimizer["points"]),
                "pair_correlators": int(len(values)),
                "classical_seconds": float(elapsed),
            },
            objective_reference=self._config["objective_reference"],
        )
