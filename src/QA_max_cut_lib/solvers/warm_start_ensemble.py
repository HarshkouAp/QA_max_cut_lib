"""Multi-start warm-start correlation source for CPU RQAOA.

One SDP/GW preprocessing supplies bounded rounded assignments.  A finite
number of globally unique assignments become independent product-state QAOA
tasks on the current working graph.  Their physical ZZ correlations are
averaged with equal component weights and returned through the ordinary
RQAOA ``CorrelationResult`` contract.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
import math
from numbers import Integral
import numpy as np

from ..core.budget import ShotBudget, ShotBudgetExceeded
from ..core.graph import WeightedGraph
from ..core.reduction import ReductionState
from .components import (
    CorrelationResult,
    _positive_int,
    _validate_initialization,
    validate_adam_config,
)
from .objectives import validate_objective
from .qaoa_state import cut_diagonal
from .qaoa_training import train_statevector
from .warm_start import (
    WarmStartError,
    mixer_unitary,
    product_state,
    source_spins_to_probabilities,
    state_at_warm_angles,
)

__all__ = [
    "WarmStartEnsembleError",
    "WarmStartEnsembleCorrelationSource",
    "validate_warm_ensemble_source_config",
]

_CORRELATION_TOLERANCE = 1e-9
_MIXERS = ("warm_aligned", "rounded_modified", "x")


class WarmStartEnsembleError(ValueError):
    """Raised when an ensemble recipe or candidate batch is invalid."""


def _mapping(value, name):
    if not isinstance(value, dict):
        raise WarmStartEnsembleError(f"{name} must be a mapping.")
    return value


def _positive(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        raise WarmStartEnsembleError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WarmStartEnsembleError(f"{name} must be finite.")
    value = float(value)
    if not math.isfinite(value):
        raise WarmStartEnsembleError(f"{name} must be finite.")
    return value


def validate_warm_ensemble_source_config(config):
    """Resolve and validate ``correlation_source`` for the ensemble recipe."""
    config = deepcopy(_mapping(config, "correlation_source"))
    allowed = {
        "type",
        "parameters",
        "candidate_source",
        "starts",
        "warm_start",
        "objective",
        "objective_reference",
        "estimation",
        "optimizer",
        "initialization",
        "step_initialization",
        "correlations",
    }
    if set(config) - allowed:
        raise WarmStartEnsembleError(
            f"Unknown ensemble source fields: {set(config) - allowed}"
        )
    if config.get("type") != "warm_start_ensemble_cpu":
        raise WarmStartEnsembleError(
            "The ensemble source type must be warm_start_ensemble_cpu."
        )
    parameters = config.setdefault("parameters", {})
    if set(parameters) - {"layers", "max_vertices", "dtype"}:
        raise WarmStartEnsembleError("Unknown ensemble parameter.")
    parameters.setdefault("layers", 1)
    parameters.setdefault("max_vertices", 18)
    parameters.setdefault("dtype", "complex128")
    _positive(parameters["layers"], "parameters.layers")
    _positive(parameters["max_vertices"], "parameters.max_vertices")
    if int(parameters["max_vertices"]) > 24:
        raise WarmStartEnsembleError("The full-state CPU cap is 24 vertices.")
    if parameters["dtype"] not in ("complex64", "complex128"):
        raise WarmStartEnsembleError("dtype must be complex64 or complex128.")

    candidate = _mapping(config.get("candidate_source"), "candidate_source")
    candidate_type = candidate.get("type")
    if candidate_type not in ("gw_rounds_cpu", "provided_by_graph"):
        raise WarmStartEnsembleError(
            "candidate_source.type must be gw_rounds_cpu or provided_by_graph."
        )
    if set(candidate) - {"type", "parameters"}:
        raise WarmStartEnsembleError("Unknown candidate_source fields.")
    cparams = candidate.setdefault("parameters", {})
    if candidate_type == "gw_rounds_cpu":
        if set(cparams) - {
            "num_hyperplanes",
            "solver",
            "allow_inaccurate",
            "eps_abs",
            "eps_rel",
            "max_iters",
            "psd_tolerance",
            "diag_tolerance",
            "repair_tolerance",
        }:
            raise WarmStartEnsembleError("Unknown gw_rounds_cpu parameter.")
        cparams.setdefault("num_hyperplanes", 16)
        cparams.setdefault("solver", "SCS")
        cparams.setdefault("allow_inaccurate", False)
        cparams.setdefault("eps_abs", 1e-6)
        cparams.setdefault("eps_rel", 1e-6)
        cparams.setdefault("max_iters", 10000)
        cparams.setdefault("psd_tolerance", 1e-6)
        cparams.setdefault("diag_tolerance", 1e-6)
        cparams.setdefault("repair_tolerance", 1e-6)
        _positive(cparams["num_hyperplanes"], "num_hyperplanes")
        if cparams["solver"] != "SCS":
            raise WarmStartEnsembleError("Only SCS is supported for gw_rounds_cpu.")
        if type(cparams["allow_inaccurate"]) is not bool:
            raise WarmStartEnsembleError("allow_inaccurate must be boolean.")
        _positive(cparams["max_iters"], "max_iters")
        for key in ("eps_abs", "eps_rel", "psd_tolerance", "diag_tolerance", "repair_tolerance"):
            if _finite(cparams[key], key) <= 0:
                raise WarmStartEnsembleError(f"{key} must be positive.")
    else:
        if set(cparams) - {"records"}:
            raise WarmStartEnsembleError(
                "provided_by_graph requires exactly parameters.records."
            )
        if not isinstance(cparams.get("records"), list):
            raise WarmStartEnsembleError("parameters.records must be a list.")

    starts = _mapping(config.get("starts"), "starts")
    if set(starts) - {"max_starts", "deduplication", "on_shortage", "ranking"}:
        raise WarmStartEnsembleError("Unknown starts field.")
    starts.setdefault("max_starts", 2)
    starts.setdefault("deduplication", "global_flip")
    starts.setdefault("on_shortage", "use_available")
    starts.setdefault("ranking", "current_cut_then_canonical_spins")
    _positive(starts["max_starts"], "starts.max_starts")
    if starts["deduplication"] not in ("global_flip", "none"):
        raise WarmStartEnsembleError("deduplication must be global_flip or none.")
    if starts["on_shortage"] not in ("use_available", "error"):
        raise WarmStartEnsembleError("on_shortage must be use_available or error.")
    if starts["ranking"] != "current_cut_then_canonical_spins":
        raise WarmStartEnsembleError("Unsupported starts.ranking.")

    warm = _mapping(config.get("warm_start"), "warm_start")
    if set(warm) - {"epsilon", "mixer"}:
        raise WarmStartEnsembleError("Unknown warm_start field.")
    warm.setdefault("epsilon", 0.25)
    warm.setdefault("mixer", {"type": "rounded_modified"})
    eps = _finite(warm["epsilon"], "warm_start.epsilon")
    if not 0 <= eps <= 0.5:
        raise WarmStartEnsembleError("warm_start.epsilon must be in [0, 0.5].")
    if not isinstance(warm["mixer"], dict) or set(warm["mixer"]) - {"type"}:
        raise WarmStartEnsembleError("warm_start.mixer must contain only type.")
    if warm["mixer"].get("type") not in _MIXERS:
        raise WarmStartEnsembleError("Unknown warm_start mixer.")

    config["objective"] = validate_objective(config.get("objective", {}))
    config.setdefault("objective_reference", "working_graph")
    if config["objective_reference"] not in ("working_graph", "original_graph"):
        raise WarmStartEnsembleError("objective_reference is invalid.")
    estimation = config.setdefault("estimation", {})
    if set(estimation) - {"mode", "shots_per_evaluation"}:
        raise WarmStartEnsembleError("Unknown estimation field.")
    estimation.setdefault("mode", "exact")
    estimation.setdefault("shots_per_evaluation", 32)
    if estimation["mode"] not in ("exact", "shots"):
        raise WarmStartEnsembleError("estimation.mode must be exact or shots.")
    _positive(estimation["shots_per_evaluation"], "shots_per_evaluation")
    config["optimizer"] = validate_adam_config(config.get("optimizer", {}))
    _validate_initialization(config, int(parameters["layers"]))
    if config.get("step_initialization", "restart") != "restart":
        raise WarmStartEnsembleError(
            "Ensemble sources require step_initialization='restart'."
        )
    correlations = config.setdefault("correlations", {})
    if set(correlations) - {"mode", "shots", "aggregation"}:
        raise WarmStartEnsembleError("Unknown correlations field.")
    correlations.setdefault("mode", "exact")
    correlations.setdefault("aggregation", "uniform_components")
    if correlations["mode"] not in ("exact", "shots"):
        raise WarmStartEnsembleError("correlations.mode must be exact or shots.")
    if correlations["mode"] == "shots":
        if "shots" not in correlations:
            raise WarmStartEnsembleError("correlations.shots is required in shots mode.")
        _positive(correlations["shots"], "correlations.shots")
    elif "shots" in correlations:
        raise WarmStartEnsembleError("exact correlations do not take a shot count.")
    if correlations["aggregation"] != "uniform_components":
        raise WarmStartEnsembleError("Only uniform_components aggregation is supported.")
    return config


def _graph_record(graph):
    return {
        "nodes": list(graph.nodes),
        "edges": [[int(u), int(v), float(w)] for u, v, w in graph.edges],
        "fingerprint": graph.snapshot()["fingerprint"],
    }


def _record_matches(graph, record):
    raw = record.get("graph")
    if not isinstance(raw, dict):
        raise WarmStartEnsembleError("A provided record needs a graph snapshot.")
    try:
        saved = WeightedGraph.from_snapshot(raw)
    except Exception as error:
        raise WarmStartEnsembleError(f"Invalid provided graph snapshot: {error}") from error
    if tuple(saved.nodes) != tuple(graph.nodes):
        return False
    return saved.snapshot()["fingerprint"] == graph.snapshot()["fingerprint"]


def _candidate_assignments(record):
    raw = record.get("assignments", record.get("candidates"))
    if not isinstance(raw, list) or not raw:
        raise WarmStartEnsembleError("A provided graph record needs assignments.")
    out = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise WarmStartEnsembleError("Each assignment must be an object.")
        spins = entry.get("spins", entry)
        if not isinstance(spins, dict):
            raise WarmStartEnsembleError("Assignment needs a spins mapping.")
        out.append(
            {
                "solution": {int(node): int(spin) for node, spin in spins.items()},
                "source_id": str(entry.get("candidate_id", entry.get("id", f"provided:{len(out)}"))),
            }
        )
    return out


def _provided_batch(graph, records):
    matches = []
    for record in records:
        if not isinstance(record, dict):
            raise WarmStartEnsembleError("Each provided record must be an object.")
        if _record_matches(graph, record):
            matches.append(record)
    if len(matches) != 1:
        raise WarmStartEnsembleError(
            f"provided_by_graph needs exactly one matching record; found {len(matches)}."
        )
    return _candidate_assignments(matches[0]), {
        "type": "provided_by_graph",
        "cost": None,
        "cost_status": "external_unaccounted",
        "records": len(records),
    }


def _gw_batch(graph, source, *, seed):
    from .goemans_williamson_cpu import GoemansWilliamsonCPU

    params = source["parameters"]
    config = {
        "schema_version": 1,
        "type": "gw_cpu",
        "seed": seed,
        "parameters": {
            "num_hyperplanes": int(params["num_hyperplanes"]),
            "solver": params["solver"],
            "allow_inaccurate": bool(params["allow_inaccurate"]),
            "eps_abs": float(params["eps_abs"]),
            "eps_rel": float(params["eps_rel"]),
            "max_iters": int(params["max_iters"]),
            "psd_tolerance": float(params["psd_tolerance"]),
            "diag_tolerance": float(params["diag_tolerance"]),
            "repair_tolerance": float(params["repair_tolerance"]),
        },
        "collect": {
            "candidate_history": True,
            "sdp_factor": True,
            "roundings": True,
        },
        "output": {"report": 0, "logs": 0, "color": "never"},
    }
    child = GoemansWilliamsonCPU(config)
    solution = child.solve(graph)
    result = child.result
    roundings = result.get("candidate_roundings") if result is not None else None
    if not isinstance(roundings, list):
        raise WarmStartEnsembleError("GW did not return its rounding batch.")
    candidates = []
    for item in roundings:
        candidates.append(
            {
                "solution": {int(node): int(spin) for node, spin in item["solution"].items()},
                "source_id": str(item.get("source_id", f"gw_round:{len(candidates)}")),
                "round_id": str(item.get("round_id", "")),
            }
        )
    return candidates, {
        "type": "gw_rounds_cpu",
        "rounds_requested": int(params["num_hyperplanes"]),
        "rounds_completed": len(candidates),
        "source_cost": deepcopy(result.get("resources") if result is not None else {}),
        "source_cost_status": "measured",
        "sdp_factor_collected": result.get("sdp_factor") is not None if result is not None else False,
        "sdp_status": result.get("sdp_status") if result is not None else None,
        "solution": solution,
    }


class WarmStartEnsembleCorrelationSource:
    """Train K current-graph warm-start components and average their ZZ data."""

    type = "warm_start_ensemble_cpu"

    def __init__(self, config, *, seed=None):
        self._config = validate_warm_ensemble_source_config(config)
        self._seed = seed
        self._step_history = []

    @property
    def config(self):
        return deepcopy(self._config)

    @property
    def max_vertices(self):
        return int(self._config["parameters"]["max_vertices"])

    @property
    def layers(self):
        return int(self._config["parameters"]["layers"])

    def _angles_for_child(self, rng):
        init = self._config["initialization"]
        if init["type"] == "fixed_angles":
            return np.asarray(init["parameters"]["values"], dtype=float)
        return np.r_[
            rng.uniform(-np.pi, np.pi, self.layers),
            rng.uniform(-np.pi / 2, np.pi / 2, self.layers),
        ]

    def _canonical_seed(self, graph, solution, source_id):
        if set(solution) != set(graph.nodes):
            raise WarmStartEnsembleError(
                "A candidate must cover exactly the current graph labels."
            )
        for value in solution.values():
            if isinstance(value, bool) or value not in (-1, 1):
                raise WarmStartEnsembleError("Candidate spins must be integer +/-1.")
        order = tuple(sorted(graph.nodes))
        raw = tuple(int(solution[node]) for node in order)
        anchor = order[0]
        if int(solution[anchor]) == -1:
            raw = tuple(-value for value in raw)
        return raw, {node: raw[order.index(node)] for node in graph.nodes}

    def _select_candidates(self, state):
        graph = state.current_graph()
        source = self._config["candidate_source"]
        if source["type"] == "provided_by_graph":
            raw, provenance = _provided_batch(graph, source["parameters"]["records"])
        else:
            raw, provenance = _gw_batch(graph, source, seed=self._seed)
        starts = self._config["starts"]
        unique = {}
        for index, item in enumerate(raw):
            source_id = str(item.get("source_id", f"candidate:{index}"))
            key, solution = self._canonical_seed(graph, item["solution"], source_id)
            if key not in unique:
                unique[key] = {
                    "key": key,
                    "solution": solution,
                    "source_ids": [],
                    "source_id": source_id,
                    "cuts": [],
                }
            unique[key]["source_ids"].append(source_id)
            unique[key]["cuts"].append(float(graph.cut_value(solution)))
        ranked = []
        for item in unique.values():
            current_cut = float(graph.cut_value(item["solution"]))
            ranked.append(
                {
                    **item,
                    "current_cut": current_cut,
                    "multiplicity": len(item["source_ids"]),
                }
            )
        ranked.sort(
            key=lambda item: (
                -item["current_cut"],
                item["key"],
                item["source_id"],
            )
        )
        requested = int(starts["max_starts"])
        if not ranked:
            raise WarmStartEnsembleError("The candidate batch is empty.")
        if len(ranked) < requested and starts["on_shortage"] == "error":
            raise WarmStartEnsembleError(
                f"Only {len(ranked)} unique starts are available; {requested} requested."
            )
        selected = ranked[:requested]
        provenance.update(
            {
                "graph": _graph_record(graph),
                "task_version": int(state.version),
                "node_order": list(graph.nodes),
                "rounds": len(raw),
                "unique": len(ranked),
                "requested_starts": requested,
                "selected_starts": len(selected),
                "shortage": len(ranked) < requested,
                "ranking": starts["ranking"],
                "deduplication": starts["deduplication"],
            }
        )
        return graph, selected, provenance

    def _quotas(self, budget, k, config):
        p = self.layers
        train_shots = int(config["estimation"]["shots_per_evaluation"]) if config["estimation"]["mode"] == "shots" else 0
        corr_shots = int(config["correlations"].get("shots", 0)) if config["correlations"]["mode"] == "shots" else 0
        reserve = k * corr_shots
        if budget.available() is not None and budget.available() < reserve:
            raise ShotBudgetExceeded(reserve, budget.available())
        max_steps = int(config["optimizer"]["parameters"]["max_steps"])
        if train_shots == 0:
            return [max_steps] * k, reserve
        step = (4 * p + 1) * train_shots
        if budget.available() is None:
            return [max_steps] * k, reserve
        available = budget.available() - reserve
        full = min(max_steps, available // (k * step)) if k else 0
        iterations = [full] * k
        extra = available - full * k * step
        for index in range(k):
            if extra >= step and iterations[index] < max_steps:
                iterations[index] += 1
                extra -= step
        return iterations, reserve

    def _objective_for_state(self, state):
        objective = deepcopy(self._config["objective"])
        shift = 0.0
        if self._config["objective_reference"] == "original_graph":
            if state.cut_offset is None:
                raise WarmStartEnsembleError(
                    "original_graph requires a Max-Cut cut offset."
                )
            shift = float(state.cut_offset)
        if objective["type"] == "threshold_probability":
            objective["parameters"]["threshold"] = (
                float(objective["parameters"]["threshold"]) - shift
            )
        return objective

    def _child(self, state, seed, quota, budget, reserve, config, run_seed, rank):
        graph = state.current_graph()
        cuts = cut_diagonal(graph)
        warm = self._config["warm_start"]
        q = np.clip(
            source_spins_to_probabilities([seed["solution"][node] for node in graph.nodes]),
            warm["epsilon"],
            1.0 - warm["epsilon"],
        )
        state_function = lambda angles: state_at_warm_angles(
            cuts,
            angles,
            self.layers,
            q,
            warm["mixer"]["type"],
            config["parameters"]["dtype"],
        )
        child_rng = np.random.default_rng([int(run_seed), int(state.version), int(rank), int(len(self._step_history))])
        initial = self._angles_for_child(child_rng)
        child_optimizer = deepcopy(config["optimizer"])
        child_optimizer["parameters"]["max_steps"] = int(quota)
        training = train_statevector(
            cuts,
            self.layers,
            objective=self._objective_for_state(state),
            estimation_mode=config["estimation"]["mode"],
            shots_per_evaluation=int(config["estimation"]["shots_per_evaluation"]),
            optimizer=child_optimizer,
            initial_angles=initial,
            dtype=config["parameters"]["dtype"],
            rng=child_rng,
            budget=budget,
            reserved_shots=reserve,
            state_function=state_function,
        )
        return {
            "solution": seed["solution"],
            "q": [float(value) for value in q],
            "alpha": [float(2.0 * np.arcsin(np.sqrt(value))) for value in q],
            "training": training,
            "initial_angles": initial.tolist(),
            "state": training["state"],
            "probabilities": training["probabilities"],
            "cuts": cuts,
            "rng": child_rng,
        }

    def _correlation_table(self, children, graph, pair_domain, mode, shots, rng):
        node_order = tuple(graph.nodes)
        if pair_domain not in ("edges", "all_pairs"):
            raise WarmStartEnsembleError("pair_domain must be edges or all_pairs.")
        pairs = (
            tuple((u, v) for u, v, _ in graph.edges)
            if pair_domain == "edges"
            else tuple(combinations(sorted(node_order), 2))
        )
        sums = {pair: 0.0 for pair in pairs}
        for child in children:
            probabilities = np.asarray(child["probabilities"], dtype=float)
            if mode == "shots":
                indices = child["rng"].choice(
                    len(probabilities), size=int(shots), p=probabilities
                ).astype(np.uint64)
                bits = (
                    (indices[:, None] >> np.arange(len(node_order), dtype=np.uint64)[None, :])
                    & np.uint64(1)
                )
                spins = 1.0 - 2.0 * bits.astype(float)
                for u, v in pairs:
                    i = node_order.index(u)
                    j = node_order.index(v)
                    sums[(u, v)] += float(np.mean(spins[:, i] * spins[:, j]))
            else:
                from ..tools.numerics import correlation_matrix

                matrix = correlation_matrix(probabilities)
                for u, v in pairs:
                    i = node_order.index(u)
                    j = node_order.index(v)
                    sums[(u, v)] += float(matrix[i, j])
        k = len(children)
        values = {pair: sums[pair] / k for pair in pairs}
        for pair, value in values.items():
            if not math.isfinite(value) or abs(value) > 1.0 + _CORRELATION_TOLERANCE:
                raise WarmStartEnsembleError(
                    f"Aggregated correlation is outside [-1, 1]: {pair}={value}."
                )
            values[pair] = min(1.0, max(-1.0, value))
        return values, pairs

    def estimate(
        self,
        state,
        *,
        budget,
        pair_domain="all_pairs",
        collect_optimizer_history=False,
        collect_objective_history=False,
    ):
        """Return one equal-weight physical correlation table for G_t."""
        if not isinstance(state, ReductionState):
            raise TypeError("state must be a ReductionState.")
        if not isinstance(budget, ShotBudget):
            raise TypeError("budget must be a ShotBudget.")
        if len(state.current_nodes) > self.max_vertices:
            raise WarmStartEnsembleError(
                f"State-vector ensemble supports at most {self.max_vertices} active nodes."
            )
        graph = state.current_graph()
        graph, seeds, provenance = self._select_candidates(state)
        quotas, reserve = self._quotas(budget, len(seeds), self._config)
        mode = self._config["correlations"]["mode"]
        corr_shots = int(self._config["correlations"].get("shots", 0)) if mode == "shots" else 0
        children = []
        try:
            for index, (seed, quota) in enumerate(zip(seeds, quotas)):
                children.append(
                    self._child(
                        state,
                        seed,
                        quota,
                        budget,
                        reserve,
                        self._config,
                        self._seed if self._seed is not None else 0,
                        index,
                    )
                )
            if mode == "shots":
                budget.spend(corr_shots * len(children))
            values, pairs = self._correlation_table(
                children, graph, pair_domain, mode, corr_shots, None
            )
        except ShotBudgetExceeded:
            raise
        training_shots = sum(int(child["training"]["shots"]) for child in children)
        evaluations = sum(int(child["training"]["evaluations"]) for child in children)
        optimizer_steps = sum(int(child["training"]["optimizer_steps"]) for child in children)
        details = {
            "source": provenance,
            "children": [
                {
                    "rank": index,
                    "source_ids": seed["source_ids"],
                    "multiplicity": seed["multiplicity"],
                    "current_cut": seed["current_cut"],
                    "solution": seed["solution"],
                    "q": child["q"],
                    "alpha": child["alpha"],
                    "mixer": self._config["warm_start"]["mixer"]["type"],
                    "epsilon": self._config["warm_start"]["epsilon"],
                    "requested_iterations": quotas[index],
                    "completed_iterations": child["training"]["optimizer_steps"],
                    "angles": child["training"]["angles"].tolist(),
                    "status": child["training"]["reason"],
                }
                for index, (seed, child) in enumerate(zip(seeds, children))
            ],
            "pair_domain": pair_domain,
            "correlation_mode": mode,
            "correlation_shots_per_child": corr_shots,
            "aggregation": "uniform_components",
        }
        resources = {
            "shots_used": int(training_shots + (corr_shots * len(children) if mode == "shots" else 0)),
            "training_shots": training_shots,
            "correlation_shots": corr_shots * len(children) if mode == "shots" else 0,
            "readout_shots": 0,
            "objective_evaluations": evaluations,
            "optimizer_steps": optimizer_steps,
            "num_children": len(children),
            "num_treads": 1,
            "source_cost": deepcopy(provenance.get("source_cost")),
            "source_cost_status": provenance.get("cost_status", provenance.get("source_cost_status")),
            "children_completed": len(children),
        }
        self._step_history.append(deepcopy(details))
        angles = children[0]["training"]["angles"] if children else np.empty(0)
        return CorrelationResult(
            node_order=tuple(graph.nodes),
            task_version=int(state.version),
            values=values,
            pairs=pairs,
            mode=mode,
            angles=np.asarray(angles, dtype=float),
            layers=self.layers,
            training=details,
            resources=resources,
            objective_reference=self._config["objective_reference"],
        )
