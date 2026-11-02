"""Replaceable components for the first CPU RQAOA solver.

The module contains:

* ``QAOAStatevectorCorrelationSource``: trains the existing full-state
  QAOA simulator and returns all physical ZZ correlations.
* ``MaxAbsCorrelationSelector``: selects one allowed pair by ``abs(c)``.
* ``ExactCPUFinisher``: solves a small residual graph with ``ExactCPU``.

The components do not contain reduction formulas.  ``RQAOACPU`` applies
the verified core from ``core.reduction``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
import math
from numbers import Integral, Real
import numpy as np

from ..core.budget import ShotBudget, ShotBudgetExceeded
from ..core.reduction import ReductionState
from .exact_cpu import ExactCPU
from .objectives import validate_objective
from .qaoa_state import cut_diagonal
from .qaoa_training import train_statevector

__all__ = [
    "CorrelationResult",
    "ExactCPUFinisher",
    "MaxAbsCorrelationSelector",
    "QAOAStatevectorCorrelationSource",
    "SelectionResult",
]

_CORRELATION_TOLERANCE = 1e-9


def _positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _mapping(value, name):
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping.")
    return value


def validate_adam_config(config, *, allow_adjoint=False):
    """Resolve and validate one ADAM optimizer section.

    Parameters
    ----------
    config : dict
        The ``optimizer`` section of a QAOA source configuration.
    allow_adjoint : bool, optional
        Permit the exact Mean-Cut GPU adjoint gradient.  CPU paths leave it
        false.

    Returns
    -------
    dict
        A new resolved section with all defaults and no unknown fields.
    """
    config = deepcopy(_mapping(config, "optimizer"))
    if (
        set(config) - {"type", "parameters", "gradient"}
        or config.get("type", "adam") != "adam"
    ):
        raise ValueError("Only the documented ADAM optimizer is implemented.")
    config["type"] = "adam"
    parameters = config.setdefault("parameters", {})
    if set(parameters) - {
        "max_steps",
        "learning_rate",
        "beta1",
        "beta2",
        "epsilon",
        "gradient_tolerance",
    }:
        raise ValueError("Unknown ADAM parameter.")
    defaults = {
        "max_steps": 20,
        "learning_rate": 0.05,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "gradient_tolerance": 0.0,
    }
    for key, value in defaults.items():
        parameters.setdefault(key, value)
    _positive_int(parameters["max_steps"], "max_steps", minimum=0)
    for key in ("learning_rate", "epsilon"):
        value = _finite(parameters[key], key)
        if value <= 0:
            raise ValueError(f"{key} must be positive.")
    for key in ("beta1", "beta2"):
        value = _finite(parameters[key], key)
        if not 0 <= value < 1:
            raise ValueError(f"{key} must be in [0, 1).")
    tolerance = _finite(parameters["gradient_tolerance"], "gradient_tolerance")
    if tolerance < 0:
        raise ValueError("gradient_tolerance must be nonnegative.")
    gradient = config.setdefault(
        "gradient", {"type": "central_difference", "parameters": {"step": 0.01}}
    )
    if set(gradient) - {"type", "parameters"} or gradient.get("type") not in (
        ("central_difference", "adjoint") if allow_adjoint else ("central_difference",)
    ):
        raise ValueError(
            "Only central_difference is implemented"
            + (" or the exact GPU adjoint" if allow_adjoint else "")
            + "."
        )
    gp = gradient.setdefault("parameters", {})
    if gradient["type"] == "adjoint":
        if set(gp):
            raise ValueError("The adjoint gradient has no parameters.")
    elif set(gp) != {"step"}:
        raise ValueError("The gradient section needs only a step.")
    if gradient["type"] == "central_difference":
        step = _finite(gp["step"], "gradient step")
        if step <= 0:
            raise ValueError("The gradient step must be positive.")
    return config


def _initial_angles(config, layers):
    """Return configured initial angles and a human-readable source name."""
    initialization = config["initialization"]
    if initialization["type"] == "fixed_angles":
        values = np.asarray(initialization["parameters"]["values"], dtype=float)
        return values.copy(), "fixed_angles"
    if initialization["type"] == "random_angles":
        return None, "random_angles"
    raise ValueError("Unknown initialization type.")


def _validate_initialization(config, layers):
    initialization = config.setdefault("initialization", {})
    if set(initialization) - {"type", "parameters"}:
        raise ValueError("Unknown initialization field.")
    initialization.setdefault("type", "fixed_angles")
    parameters = initialization.setdefault("parameters", {})
    if initialization["type"] == "fixed_angles":
        parameters.setdefault("values", [0.5] * layers + [0.3] * layers)
        if (
            set(parameters) != {"values"}
            or np.shape(parameters["values"]) != (2 * layers,)
            or not np.isfinite(parameters["values"]).all()
        ):
            raise ValueError("Supply exactly 2*layers finite initial angles.")
    elif initialization["type"] == "random_angles":
        if parameters:
            raise ValueError("random_angles has no parameters.")
    else:
        raise ValueError("Unknown initialization type.")


def validate_correlation_source_config(config):
    """Resolve and validate one QAOA correlation-source configuration.

    Parameters
    ----------
    config : dict
        The nested ``correlation_source`` section of RQAOA.

    Returns
    -------
    dict
        A new configuration with defaults.  The public source has no
        readout, save, or independent shot limit.
    """
    config = deepcopy(_mapping(config, "correlation_source"))
    if config.get("type") == "qaoa_analytic_p1_cpu":
        from .analytic_p1 import validate_analytic_p1_source_config

        return validate_analytic_p1_source_config(config)
    if config.get("type") == "qaoa_statevector_gpu":
        from .qaoa_gpu import validate_gpu_source_config

        return validate_gpu_source_config(config)
    if config.get("type") == "warm_start_ensemble_cpu":
        from .warm_start_ensemble import validate_warm_ensemble_source_config

        return validate_warm_ensemble_source_config(config)
    if config.get("type") == "qaoa_ising_statevector_cpu":
        from .conditional_source import ConditionalCorrelationSource

        return ConditionalCorrelationSource(config).config
    allowed = {
        "type",
        "parameters",
        "objective",
        "objective_reference",
        "estimation",
        "optimizer",
        "initialization",
        "step_initialization",
        "correlations",
    }
    if set(config) - allowed:
        raise ValueError(f"Unknown correlation-source fields: {set(config) - allowed}")
    if config.get("type") != "qaoa_statevector_cpu":
        raise ValueError(
            "Only qaoa_statevector_cpu and qaoa_analytic_p1_cpu are implemented."
        )
    parameters = config.setdefault("parameters", {})
    if set(parameters) - {"layers", "max_vertices", "dtype"}:
        raise ValueError("Unknown QAOA source parameter.")
    parameters.setdefault("layers", 1)
    parameters.setdefault("max_vertices", 20)
    parameters.setdefault("dtype", "complex128")
    layers = _positive_int(parameters["layers"], "layers")
    max_vertices = _positive_int(parameters["max_vertices"], "max_vertices")
    if max_vertices > 24:
        raise ValueError("The direct state-vector source has a hard cap of 24 vertices.")
    if parameters["dtype"] not in ("complex64", "complex128"):
        raise ValueError("dtype must be complex64 or complex128.")
    config["objective"] = validate_objective(config.get("objective", {}))
    config.setdefault("objective_reference", "working_graph")
    if config["objective_reference"] not in ("working_graph", "original_graph"):
        raise ValueError("objective_reference must be working_graph or original_graph.")
    estimation = config.setdefault("estimation", {})
    if set(estimation) - {"mode", "shots_per_evaluation"}:
        raise ValueError("Unknown estimation field.")
    estimation.setdefault("mode", "exact")
    estimation.setdefault("shots_per_evaluation", 256)
    if estimation["mode"] not in ("exact", "shots"):
        raise ValueError("estimation.mode must be exact or shots.")
    _positive_int(estimation["shots_per_evaluation"], "shots_per_evaluation")
    config["optimizer"] = validate_adam_config(config.get("optimizer", {}))
    _validate_initialization(config, layers)
    config.setdefault("step_initialization", "restart")
    if config["step_initialization"] not in ("restart", "previous_step"):
        raise ValueError("step_initialization must be restart or previous_step.")
    correlations = config.setdefault("correlations", {})
    if set(correlations) - {"mode", "shots"}:
        raise ValueError("Unknown correlations field.")
    correlations.setdefault("mode", "exact")
    if correlations["mode"] not in ("exact", "shots"):
        raise ValueError("correlations.mode must be exact or shots.")
    if correlations["mode"] == "shots":
        if "shots" not in correlations:
            raise ValueError("correlations.shots is required in shots mode.")
        _positive_int(correlations["shots"], "correlations.shots")
    elif "shots" in correlations:
        raise ValueError("exact correlations do not take a shot count.")
    return config


@dataclass(frozen=True)
class CorrelationResult:
    """Store one correlation estimate and its provenance.

    Attributes
    ----------
    node_order : tuple of int
        Active node labels.  Bit ``k`` belongs to ``node_order[k]``.
    task_version : int
        Reduction-state version that was evaluated.
    values : dict of (int, int) to float
        Physical ``<Z_u Z_v>`` for every unordered pair.
    pairs : tuple of (int, int)
        Covered unordered pairs.
    mode : {'exact', 'shots'}
        Correlation estimation mode.
    angles : ndarray
        Best evaluated angles in gamma-then-beta order.
    layers : int
        QAOA depth used for this table.
    training : dict
        Compact training metadata.
    resources : dict
        Training and correlation counters.
    objective_reference : str
        ``working_graph`` or ``original_graph``.
    """

    node_order: tuple
    task_version: int
    values: dict
    pairs: tuple
    mode: str
    angles: np.ndarray
    layers: int
    training: dict
    resources: dict
    objective_reference: str


class QAOAStatevectorCorrelationSource:
    """Train full-state QAOA and return physical ZZ correlations.

    Parameters
    ----------
    config : dict
        The ``correlation_source`` section.  The source has no ``save``,
        no public readout, and no independent root shot limit.
    seed : int or None, optional
        Root seed.  A new source instance is created for every solve().

    Notes
    -----
    One instance keeps the previous-step angles for
    ``step_initialization="previous_step"``.  It does not keep a
    correlation table after a graph change.  Every estimate call computes
    a table for the current working graph and task version.
    """

    type = "qaoa_statevector_cpu"

    def __init__(self, config, *, seed=None):
        self._config = validate_correlation_source_config(config)
        self._rng = np.random.default_rng(seed)
        self._previous_angles = None
        self._step_index = 0

    @property
    def config(self):
        """Return a copy of the resolved source configuration."""
        return deepcopy(self._config)

    @property
    def max_vertices(self):
        """Return the configured state-vector size limit."""
        return self._config["parameters"]["max_vertices"]

    @property
    def layers(self):
        """Return the configured QAOA depth."""
        return self._config["parameters"]["layers"]

    def _angles_for_step(self):
        config = self._config
        if (
            config["step_initialization"] == "previous_step"
            and self._previous_angles is not None
        ):
            return self._previous_angles.copy(), "previous_step"
        values, name = _initial_angles(config, self.layers)
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
        requested = None
        effective = None
        if objective["type"] == "threshold_probability":
            requested = float(objective["parameters"]["threshold"])
            effective = requested - shift
            objective["parameters"]["threshold"] = effective
        return objective, shift, requested, effective

    def estimate(
        self,
        state,
        *,
        budget,
        pair_domain="all_pairs",
        collect_optimizer_history=False,
        collect_objective_history=False,
    ):
        """Estimate all active-pair correlations on the current task.

        Parameters
        ----------
        state : ReductionState
            The current reduction state.  Its working graph and version
            define the evaluated task.
        budget : ShotBudget
            The root budget.  Correlation shots are reserved before
            training starts.
        pair_domain : {'edges', 'all_pairs'}, default 'all_pairs'
            Reported coverage domain.  This source covers all pairs.
        collect_optimizer_history, collect_objective_history : bool
            Store local training histories in the returned metadata.

        Returns
        -------
        CorrelationResult
            Values, coverage, angles, and resource counters.

        Raises
        ------
        ValueError
            The task is too large, empty, or has an invalid bit order.
        ShotBudgetExceeded
            The mandatory correlation sample does not fit.
        """
        if not isinstance(state, ReductionState):
            raise TypeError("state must be a ReductionState.")
        if not isinstance(budget, ShotBudget):
            raise TypeError("budget must be a ShotBudget.")
        if pair_domain not in ("edges", "all_pairs"):
            raise ValueError("pair_domain must be edges or all_pairs.")
        node_order = tuple(state.current_nodes)
        n = len(node_order)
        if n < 2:
            raise ValueError("A correlation table needs at least two active nodes.")
        if n > self.max_vertices:
            raise ValueError(
                f"State-vector source supports at most {self.max_vertices} active nodes."
            )
        graph = state.current_graph()
        cuts = cut_diagonal(graph)
        objective, shift, threshold_requested, threshold_effective = (
            self._objective_for_state(state)
        )
        angles, initialization_name = self._angles_for_step()
        correlations_config = self._config["correlations"]
        corr_shots = (
            int(correlations_config["shots"])
            if correlations_config["mode"] == "shots"
            else 0
        )
        available = budget.available()
        if available is not None and available < corr_shots:
            raise ShotBudgetExceeded(corr_shots, available)

        training = train_statevector(
            cuts,
            self.layers,
            objective=objective,
            estimation_mode=self._config["estimation"]["mode"],
            shots_per_evaluation=self._config["estimation"]["shots_per_evaluation"],
            optimizer=self._config["optimizer"],
            initial_angles=angles,
            dtype=self._config["parameters"]["dtype"],
            rng=self._rng,
            budget=budget,
            reserved_shots=corr_shots,
            collect_optimizer_history=collect_optimizer_history,
            collect_objective_history=collect_objective_history,
        )
        probabilities = training["probabilities"]
        if corr_shots:
            indices = self._rng.choice(len(cuts), size=corr_shots, p=probabilities).astype(
                np.uint64
            )
            # Each sampled index is one bit string.  Build an (M, n) spin
            # table; do not allocate an (M, 2**n) table.
            shifts = np.arange(n, dtype=np.uint64)
            bits = ((indices[:, None] >> shifts[None, :]) & np.uint64(1)).astype(float)
            spins = 1.0 - 2.0 * bits
            matrix = (spins.T @ spins) / corr_shots
            budget.spend(corr_shots)
        else:
            from ..tools.numerics import correlation_matrix

            matrix = correlation_matrix(probabilities)

        pairs = tuple(combinations(sorted(node_order), 2))
        values = {}
        for i, u in enumerate(node_order):
            for j, v in enumerate(node_order[:i]):
                value = float(matrix[i, j])
                if not math.isfinite(value):
                    raise ArithmeticError("A physical correlation is not finite.")
                if abs(value) > 1.0 + _CORRELATION_TOLERANCE:
                    raise ValueError("A physical correlation is outside [-1, 1].")
                value = min(1.0, max(-1.0, value))
                values[(min(u, v), max(u, v))] = value
        expected_working = float(cuts @ probabilities)
        objective_value = training["objective_value"]
        if objective_value is None:
            objective_original = None
        elif objective["type"] == "threshold_probability":
            objective_original = objective_value
        else:
            objective_original = objective_value + shift
        self._previous_angles = np.asarray(training["angles"], dtype=float).copy()
        self._step_index += 1
        return CorrelationResult(
            node_order=node_order,
            task_version=state.version,
            values=values,
            pairs=pairs,
            mode=correlations_config["mode"],
            angles=np.asarray(training["angles"], dtype=float),
            layers=self.layers,
            training={
                "initialization": initialization_name,
                "reason": training["reason"],
                "objective_value": objective_value,
                "objective_value_original": objective_original,
                "objective_reference": self._config["objective_reference"],
                "objective_shift": shift,
                "objective_threshold_requested": threshold_requested,
                "objective_threshold_effective": threshold_effective,
                "expected_cut_working": expected_working,
                "expected_cut_original": expected_working + shift,
                "optimizer_steps": training["optimizer_steps"],
                "objective_evaluations": training["evaluations"],
                "training_shots": training["shots"],
                "best_loss": training["best_loss"],
                "optimizer_history": training["optimizer_history"],
                "objective_history": training["objective_history"],
            },
            resources={
                "training_shots": training["shots"],
                "correlation_shots": corr_shots,
                "objective_evaluations": training["evaluations"],
                "optimizer_steps": training["optimizer_steps"],
                "source_calls": 1,
            },
            objective_reference=self._config["objective_reference"],
        )


@dataclass(frozen=True)
class SelectionResult:
    """Store one selected directed pair and its sign."""

    removed: int
    retained: int
    sign: int
    score: float
    correlation: float


class MaxAbsCorrelationSelector:
    """Select the allowed pair with the largest absolute correlation.

    Parameters
    ----------
    config : dict
        ``{"type": "max_abs_correlation", "parameters": {"pairs": ...,
        "zero_sign": ...}}``.  ``pairs`` is ``edges`` or ``all_pairs``.
        ``zero_sign`` is the constraint sign for exact zero correlation.

    Notes
    -----
    Ties use the lexicographically first unordered pair of external
    labels.  The default direction removes the larger label and retains
    the smaller label.
    """

    type = "max_abs_correlation"

    def __init__(self, config=None):
        config = deepcopy(config or {})
        if set(config) - {"type", "parameters"} or config.get(
            "type", self.type
        ) != self.type:
            raise ValueError("selection type must be max_abs_correlation.")
        parameters = config.setdefault("parameters", {})
        if set(parameters) - {"pairs", "zero_sign"}:
            raise ValueError("Unknown selection parameter.")
        parameters.setdefault("pairs", "edges")
        parameters.setdefault("zero_sign", 1)
        if parameters["pairs"] not in ("edges", "all_pairs"):
            raise ValueError("selection.pairs must be edges or all_pairs.")
        if isinstance(parameters["zero_sign"], bool) or parameters["zero_sign"] not in (-1, 1):
            raise ValueError("selection.zero_sign must be integer +1 or -1.")
        self._pairs = parameters["pairs"]
        self._zero_sign = int(parameters["zero_sign"])

    @property
    def config(self):
        """Return a resolved selection configuration."""
        return {
            "type": self.type,
            "parameters": {
                "pairs": self._pairs,
                "zero_sign": self._zero_sign,
            },
        }

    def select(self, state, correlations):
        """Select one pair from a validated correlation table.

        Parameters
        ----------
        state : ReductionState
            Current task and version.
        correlations : CorrelationResult
            A table produced for exactly this state.

        Returns
        -------
        SelectionResult
            Directed pair, sign, separate score, and signed correlation.

        Raises
        ------
        ValueError
            The table is stale, incomplete, or outside the physical range.
        """
        if not isinstance(state, ReductionState):
            raise TypeError("state must be a ReductionState.")
        if not isinstance(correlations, CorrelationResult):
            raise TypeError("correlations must be a CorrelationResult.")
        if correlations.task_version != state.version:
            raise ValueError("The correlation table belongs to an older task.")
        if tuple(correlations.node_order) != tuple(state.current_nodes):
            raise ValueError("The correlation node order does not match the task.")
        if self._pairs == "edges":
            allowed = [
                (min(u, v), max(u, v))
                for u, v, _ in state.current_graph().edges
            ]
        else:
            allowed = list(combinations(sorted(state.current_nodes), 2))
        if not allowed:
            raise ValueError("No allowed pair exists for selection.")
        candidates = []
        for pair in allowed:
            if pair not in correlations.values:
                raise ValueError(f"Correlation is missing for pair {pair}.")
            value = float(correlations.values[pair])
            if not math.isfinite(value):
                raise ValueError("A selected correlation is not finite.")
            if abs(value) > 1.0 + _CORRELATION_TOLERANCE:
                raise ValueError("A selected correlation is outside [-1, 1].")
            value = min(1.0, max(-1.0, value))
            candidates.append((abs(value), pair, value))
        best = min(candidates, key=lambda item: (-item[0], item[1][0], item[1][1]))
        _, pair, correlation = best
        sign = 1 if correlation > 0 else -1 if correlation < 0 else self._zero_sign
        return SelectionResult(
            removed=max(pair),
            retained=min(pair),
            sign=sign,
            score=abs(correlation),
            correlation=correlation,
        )


class ExactCPUFinisher:
    """Solve a small residual Max-Cut graph with the exact CPU solver."""

    type = "exact_cpu"

    def __init__(self, config=None):
        config = deepcopy(config or {})
        if set(config) - {"type", "parameters"} or config.get(
            "type", self.type
        ) != self.type:
            raise ValueError("finisher type must be exact_cpu.")
        parameters = config.setdefault("parameters", {})
        if set(parameters) - {"max_vertices", "batch_size", "tie_tolerance"}:
            raise ValueError("Unknown exact finisher parameter.")
        parameters.setdefault("max_vertices", 16)
        _positive_int(parameters["max_vertices"], "finisher max_vertices")
        solver_config = {"parameters": deepcopy(parameters)}
        self._solver = ExactCPU(solver_config)

    @property
    def max_vertices(self):
        """Return the finisher vertex limit."""
        return self._solver.config["parameters"]["max_vertices"]

    @property
    def config(self):
        """Return a resolved finisher configuration."""
        return {
            "type": self.type,
            "parameters": deepcopy(self._solver.config["parameters"]),
        }

    def solve(self, graph):
        """Return the exact residual spin dictionary and its result record."""
        solution = self._solver.solve(graph, report=0, logs=0)
        return solution, self._solver.last_run
