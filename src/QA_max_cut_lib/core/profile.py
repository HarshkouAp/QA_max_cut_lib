"""Parallelism profiling, Amdahl/Karp-Flatt analysis, and recommendations."""

from __future__ import annotations

from copy import deepcopy
import math
import statistics
from time import perf_counter

import numpy as np

from .experiment_ids import measurement_protocol_id, stable_hash
from .experiment_spec import BenchmarkSpec, PlanError, PlannedTrial, load_benchmark_spec
from .measurement import CampaignBudget, estimate_trial_shots, execute_trial, measurement_environment
from .records import to_json_data
from .references import ReferencePolicy
from .series_analysis import analyze_series
from .series_store import ExperimentStore

__all__ = [
    "ProfileSpec",
    "ProfilePlan",
    "ParallelismProfiler",
    "build_profile_plan",
    "profile_parallelism",
    "fit_amdahl",
    "amdahl_model",
    "karp_flatt",
    "recommend_total_curve",
]


_PROFILE_FIELDS = {
    "schema_version",
    "num_treads_values",
    "parameter_coverage",
    "seed_index",
    "seed",
    "warmup_runs",
    "repetitions",
    "execution_mode",
    "timing_scope",
    "measurement_order",
    "measurement_order_seed",
    "models",
    "uncertainty",
    "recommendation",
    "series_limits",
    "save",
    "case_weights",
}


def _positive_ints(values, name):
    if not isinstance(values, list) or not values:
        raise PlanError(f"{name} must be a nonempty list.")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise PlanError(f"{name} values must be positive integers.")
        if value in result:
            raise PlanError(f"{name} values must be unique.")
        result.append(int(value))
    return result


class ProfileSpec:
    """Validated parallelism-sweep configuration."""

    def __init__(self, data):
        if not isinstance(data, dict):
            raise PlanError("A parallelism configuration must be a mapping.")
        extra = set(data) - _PROFILE_FIELDS
        if extra:
            raise PlanError(f"Unknown parallelism fields: {sorted(extra)}.")
        if data.get("schema_version", 1) != 1:
            raise PlanError("Unsupported parallelism schema_version.")
        self.data = deepcopy(data)
        self.values = _positive_ints(data.get("num_treads_values"), "num_treads_values")
        if 1 not in self.values:
            raise PlanError("num_treads_values must include the base value 1.")
        coverage = data.get("parameter_coverage", "full_grid")
        if coverage != "full_grid":
            raise PlanError("a9 supports parameter_coverage='full_grid' only.")
        self.coverage = coverage
        if "seed_index" in data and "seed" in data:
            raise PlanError("Use seed_index or seed, not both.")
        if "seed" in data:
            seed = data["seed"]
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise PlanError("seed must be an integer.")
            self.seed_index = None
            self.seed = int(seed)
        else:
            index = data.get("seed_index", 0)
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise PlanError("seed_index must be a nonnegative integer.")
            self.seed_index = index
            self.seed = None
        self.warmup_runs = data.get("warmup_runs", 0)
        self.repetitions = data.get("repetitions", 1)
        for name, value in (("warmup_runs", self.warmup_runs), ("repetitions", self.repetitions)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PlanError(f"{name} must be a nonnegative integer.")
        if self.repetitions < 1:
            raise PlanError("repetitions must be at least one.")
        self.execution_mode = data.get("execution_mode", "isolated")
        if self.execution_mode not in ("isolated", "concurrent_batch"):
            raise PlanError("execution_mode must be isolated or concurrent_batch.")
        self.timing_scope = data.get("timing_scope", "benchmark")
        if self.timing_scope not in ("benchmark", "end_to_end", "solve_only"):
            raise PlanError("timing_scope must be benchmark, end_to_end, or solve_only.")
        self.order = data.get("measurement_order", "shuffle_within_case_repetition")
        if self.order not in ("shuffle_within_case_repetition", "fixed"):
            raise PlanError("Unknown measurement_order.")
        seed = data.get("measurement_order_seed", 2027)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise PlanError("measurement_order_seed must be an integer.")
        self.order_seed = int(seed)
        models = data.get("models", ["amdahl", "karp_flatt"])
        if not isinstance(models, list) or set(models) - {"amdahl", "karp_flatt"}:
            raise PlanError("models may contain amdahl and karp_flatt.")
        self.models = list(models)
        uncertainty = data.get("uncertainty", {})
        if not isinstance(uncertainty, dict):
            raise PlanError("uncertainty must be a mapping.")
        if set(uncertainty) - {"method", "resamples", "confidence", "seed"}:
            raise PlanError("Unknown uncertainty field.")
        self.uncertainty = {
            "method": uncertainty.get("method", "paired_block_bootstrap"),
            "resamples": int(uncertainty.get("resamples", 0)),
            "confidence": float(uncertainty.get("confidence", 0.95)),
            "seed": int(uncertainty.get("seed", 0)),
        }
        if self.uncertainty["method"] not in ("paired_block_bootstrap", "none"):
            raise PlanError("Unsupported uncertainty method.")
        if self.uncertainty["resamples"] < 0:
            raise PlanError("uncertainty.resamples must be nonnegative.")
        recommendation = data.get("recommendation", {})
        if not isinstance(recommendation, dict):
            raise PlanError("recommendation must be a mapping.")
        if set(recommendation) - {"criterion", "tolerance_rel", "prefer_lower_resource"}:
            raise PlanError("Unknown recommendation field.")
        self.recommendation = {
            "criterion": recommendation.get("criterion", "total_time"),
            "tolerance_rel": float(recommendation.get("tolerance_rel", 0.0)),
            "prefer_lower_resource": bool(recommendation.get("prefer_lower_resource", True)),
        }
        if self.recommendation["criterion"] != "total_time":
            raise PlanError("a9 supports recommendation.criterion='total_time' only.")
        if self.recommendation["tolerance_rel"] < 0:
            raise PlanError("tolerance_rel must be nonnegative.")
        limits = deepcopy(data.get("series_limits", {}))
        if not isinstance(limits, dict) or set(limits) - {"max_runs_total", "shots_total"}:
            raise PlanError("Unknown profile series_limits field.")
        self.series_limits = {
            "max_runs_total": limits.get("max_runs_total"),
            "shots_total": limits.get("shots_total"),
        }
        for key, value in self.series_limits.items():
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise PlanError(f"profile series_limits.{key} must be nonnegative.")
        weights = data.get("case_weights", {})
        if not isinstance(weights, dict):
            raise PlanError("case_weights must be a mapping.")
        if any(not isinstance(value, (int, float)) or value <= 0 for value in weights.values()):
            raise PlanError("Every case weight must be positive.")
        self.case_weights = {str(key): float(value) for key, value in weights.items()}
        self.save = deepcopy(data.get("save")) if "save" in data else None


class ProfilePlan:
    """Measured profile points and their warmup calls."""

    def __init__(self, *, base_plan, seed, values, warmups, measured, protocol, config):
        self.base_plan = base_plan
        self.seed = int(seed)
        self.values = list(values)
        self.warmups = list(warmups)
        self.measured = list(measured)
        self.protocol = deepcopy(protocol)
        self.config = deepcopy(config)
        self.case_count = len(base_plan.cases)
        self.points = self.case_count * len(self.values)
        self.calls = len(self.warmups) + len(self.measured)
        self.warmup_records = []
        self.repetitions = int(self.config.get("repetitions", 1))
        self.case_weights = deepcopy(self.config.get("case_weights", {}))
        self.recommendation_config = deepcopy(self.config.get("recommendation", {}))
        self.uncertainty_config = deepcopy(self.config.get("uncertainty", {}))

    @property
    def summary(self):
        return (
            f"seed={self.seed} cases={self.case_count} candidates={len(self.values)} "
            f"points={self.points} warmups={len(self.warmups)} "
            f"measured={len(self.measured)} calls={self.calls}"
        )

    @property
    def result(self):
        return to_json_data(
            {
                "schema_version": 1,
                "record_type": "parallelism_plan",
                "summary": self.summary,
                "seed": self.seed,
                "num_treads_values": list(self.values),
                "case_count": self.case_count,
                "points": self.points,
                "warmup_calls": len(self.warmups),
                "measured_calls": len(self.measured),
                "total_calls": self.calls,
                "measurement_protocol": self.protocol,
                "cases": [case.to_dict() for case in self.base_plan.cases],
                "warmups": [trial.to_dict() for trial in self.warmups],
                "measured": [trial.to_dict() for trial in self.measured],
            }
        )


def build_profile_plan(spec, config):
    """Build a full-grid profile from the same benchmark spec.

    The selected seed replaces the algorithm-seed repetition axis.  Graph
    recipe seeds and all graph/solver branches remain unchanged.
    """
    if not isinstance(spec, BenchmarkSpec):
        spec = BenchmarkSpec(spec)
    if not isinstance(config, ProfileSpec):
        config = ProfileSpec(config)
    if config.seed is not None:
        seed = config.seed
        if seed not in spec.trial_seeds:
            raise PlanError("Explicit profile seed is not in benchmark trial_seeds.")
    else:
        if config.seed_index >= len(spec.trial_seeds):
            raise PlanError("seed_index is outside trial_seeds.")
        seed = spec.trial_seeds[config.seed_index]
    base = spec.plan(trial_seeds=[seed], profile=True)
    protocol = deepcopy(spec.measurement_protocol)
    if config.timing_scope != "benchmark":
        protocol["timing_scope"] = config.timing_scope
    protocol["declared_timing_scope"] = config.timing_scope
    protocol["profile_execution_mode"] = config.execution_mode
    warmups, measured = [], []
    for case in base.cases:
        for candidate in config.values:
            for _ in range(config.warmup_runs):
                warmups.append(
                    PlannedTrial(
                        case,
                        seed=seed,
                        num_treads=candidate,
                        protocol=protocol,
                        repetition_id=-1,
                        role="warmup",
                    )
                )
            for repetition in range(config.repetitions):
                measured.append(
                    PlannedTrial(
                        case,
                        seed=seed,
                        num_treads=candidate,
                        protocol=protocol,
                        repetition_id=repetition,
                        role="measurement",
                    )
                )
    return ProfilePlan(
        base_plan=base,
        seed=seed,
        values=config.values,
        warmups=warmups,
        measured=measured,
        protocol=protocol,
        config=config.data,
    )


def amdahl_model(t1, fraction_serial, n):
    """Return the fixed-work Amdahl time for one resource value."""
    if t1 <= 0 or not 0 <= fraction_serial <= 1 or n < 1:
        raise ValueError("Amdahl inputs are outside their domain.")
    return float(t1) * (float(fraction_serial) + (1.0 - fraction_serial) / float(n))


def karp_flatt(t1, tn, n):
    """Return the experimental serial fraction, or None at n=1."""
    if n == 1:
        return None
    if t1 <= 0 or tn <= 0:
        raise ValueError("Karp-Flatt times must be positive.")
    speedup = float(t1) / float(tn)
    return float((1.0 / speedup - 1.0 / n) / (1.0 - 1.0 / n))


def fit_amdahl(points):
    """Fit ``T(n) = T1*(f + (1-f)/n)`` to finite positive points.

    Parameters
    ----------
    points : list of (n, T)
        At least two distinct resource values; three including 1 are needed
        for the documented minimum-coverage label.

    Returns
    -------
    dict
        Coefficients, percentages, residual, coverage, and applicability.
        A boundary coefficient is reported, not treated as proof.
    """
    rows = []
    for n, value in points:
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
            continue
        rows.append((int(n), float(value)))
    result = {
        "points": len(rows),
        "n_values": [row[0] for row in rows],
        "a": None,
        "b": None,
        "t1": None,
        "serial_fraction": None,
        "parallel_fraction": None,
        "serial_percent": None,
        "parallel_percent": None,
        "rmse": None,
        "relative_rmse": None,
        "minimum_coverage": False,
        "applicability": "insufficient_points",
        "boundary": False,
    }
    if len(rows) < 2 or len({row[0] for row in rows}) < 2:
        return result
    ns = np.asarray([row[0] for row in rows], dtype=float)
    ts = np.asarray([row[1] for row in rows], dtype=float)
    design = np.column_stack([np.ones_like(ns), 1.0 / ns])
    coefficients, *_ = np.linalg.lstsq(design, ts, rcond=None)
    a, b = float(coefficients[0]), float(coefficients[1])
    t1 = a + b
    if not math.isfinite(t1) or t1 <= 0:
        result["applicability"] = "invalid_total"
        return result
    serial = a / t1
    fitted = design @ coefficients
    residual = ts - fitted
    rmse = float(np.sqrt(np.mean(residual**2)))
    relative = rmse / max(float(np.mean(ts)), 1e-300)
    result.update(
        a=a,
        b=b,
        t1=t1,
        serial_fraction=serial,
        parallel_fraction=1.0 - serial,
        serial_percent=100.0 * serial,
        parallel_percent=100.0 * (1.0 - serial),
        rmse=rmse,
        relative_rmse=relative,
        minimum_coverage=len(rows) >= 3 and 1 in {row[0] for row in rows},
        boundary=bool(serial <= 0.0 or serial >= 1.0),
    )
    if not result["minimum_coverage"]:
        result["applicability"] = "insufficient_coverage"
    elif result["boundary"]:
        result["applicability"] = "boundary_or_unphysical"
    elif relative > 0.15:
        result["applicability"] = "poor_fit"
    else:
        result["applicability"] = "tentative"
    return result


def recommend_total_curve(points, *, tolerance_rel=0.0, prefer_lower_resource=True):
    """Choose a global resource value from complete case totals.

    Parameters
    ----------
    points : list of dict
        Each item must contain ``n``, ``total_seconds``, and ``complete``.
    tolerance_rel : float, default 0
        Explicit allowed loss relative to the best complete total.
    prefer_lower_resource : bool, default True
        Choose the smallest n among totals within the tolerance band.

    Returns
    -------
    dict
        Raw argmin, practical recommendation, and completeness flags.
    """
    candidates = [
        item
        for item in points
        if item.get("complete") is True
        and isinstance(item.get("n"), int)
        and item["n"] >= 1
        and isinstance(item.get("total_seconds"), (int, float))
        and math.isfinite(float(item["total_seconds"]))
        and item["total_seconds"] > 0
    ]
    result = {
        "raw_argmin": None,
        "recommended_num_treads": None,
        "tolerance_rel": float(tolerance_rel),
        "search_complete": False,
        "best_observed_complete": None,
        "reason": None,
    }
    if not candidates:
        result["reason"] = "no_complete_candidate"
        return result
    raw = min(candidates, key=lambda item: (item["total_seconds"], item["n"]))
    best = float(raw["total_seconds"])
    allowed = best * (1.0 + float(tolerance_rel))
    band = [item for item in candidates if item["total_seconds"] <= allowed]
    practical = min(band, key=lambda item: (item["n"] if prefer_lower_resource else item["total_seconds"]))
    result.update(
        raw_argmin=raw["n"],
        recommended_num_treads=practical["n"],
        search_complete=all(item.get("search_complete", False) for item in candidates),
        best_observed_complete=raw["n"],
    )
    return result


def _median(values):
    values = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value)) and value > 0]
    return statistics.median(values) if values else None


def _bootstrap_totals(by_case, *, weights, values, repetitions, config):
    method = (config or {}).get("method", "none")
    resamples = int((config or {}).get("resamples", 0))
    confidence = float((config or {}).get("confidence", 0.95))
    seed = int((config or {}).get("seed", 0))
    result = {
        "method": method,
        "resamples": resamples,
        "confidence": confidence,
        "intervals": {},
        "recommended_stability": None,
        "reason": None,
    }
    if method != "paired_block_bootstrap" or resamples <= 0:
        result["reason"] = "disabled"
        return result
    if repetitions < 2:
        result["reason"] = "needs_at_least_two_repetitions"
        return result
    rng = np.random.default_rng(seed)
    totals = {n: [] for n in values}
    recommendations = []
    for _ in range(resamples):
        block = rng.integers(0, repetitions, size=repetitions)
        total_by_n = {n: 0.0 for n in values}
        possible = True
        for case_id, groups in by_case.items():
            weight = weights.get(case_id, 1.0)
            per_n = {}
            for n in values:
                records = [
                    record
                    for record in groups.get(n, [])
                    if record.get("status") in ("completed", "stopped")
                    and record.get("cut_value") is not None
                ]
                if len(records) < repetitions:
                    possible = False
                    break
                per_n[n] = records
            if not possible:
                break
            for n in values:
                records = per_n[n]
                sampled = [
                    records[int(index)]["timing"]["wall_seconds"]
                    for index in block
                    if int(index) < len(records)
                ]
                median = _median(sampled)
                if median is None:
                    possible = False
                    break
                total_by_n[n] += weight * median
            if not possible:
                break
        if possible:
            for n in values:
                totals[n].append(total_by_n[n])
            complete = [
                {"n": n, "total_seconds": total_by_n[n], "complete": True}
                for n in values
            ]
            choice = recommend_total_curve(
                complete,
                tolerance_rel=float(config.get("tolerance_rel", 0.0)),
                prefer_lower_resource=bool(config.get("prefer_lower_resource", True)),
            )
            if choice["recommended_num_treads"] is not None:
                recommendations.append(choice["recommended_num_treads"])
    alpha = (1.0 - confidence) / 2.0
    for n, samples in totals.items():
        if len(samples) >= 2:
            result["intervals"][n] = {
                "count": len(samples),
                "low": float(np.quantile(samples, alpha)),
                "high": float(np.quantile(samples, 1.0 - alpha)),
            }
    if recommendations:
        result["recommended_stability"] = {
            n: recommendations.count(n) / len(recommendations)
            for n in sorted(set(recommendations))
        }
    elif not totals[values[0]]:
        result["reason"] = "no_complete_bootstrap_sample"
    return result


def _profile_summary(plan, records, references, errors):
    cases = {case.case_id: case for case in plan.base_plan.cases}
    by_case = {}
    for record in records:
        if record.get("role") != "measurement":
            continue
        case_id = record.get("case_id")
        candidate = record.get("requested_num_treads")
        if case_id not in cases or not isinstance(candidate, int):
            continue
        status = record.get("status")
        valid = status in ("completed", "stopped") and record.get("cut_value") is not None
        by_case.setdefault(case_id, {}).setdefault(candidate, []).append(record)
    case_curves = {}
    work_mismatch = []
    for case_id, candidates in by_case.items():
        case = cases[case_id]
        signatures = {}
        for n, group in candidates.items():
            signatures[n] = sorted({record.get("work") for record in group if record.get("status") in ("completed", "stopped")})
        baseline = signatures.get(1)
        curve = {}
        for n, group in sorted(candidates.items()):
            times = [record["timing"]["wall_seconds"] for record in group if record.get("status") in ("completed", "stopped") and record.get("cut_value") is not None]
            cut_values = [record["cut_value"] for record in group if record.get("status") in ("completed", "stopped") and record.get("cut_value") is not None]
            complete = len(group) >= plan.repetitions and len(times) == len(group)
            comparable = baseline is None or signatures[n] == baseline
            if not comparable:
                work_mismatch.append({"case_id": case_id, "num_treads": n, "reason": "work_signature_mismatch"})
            if not comparable:
                complete = False
            curve[n] = {
                "num_treads": n,
                "times": [float(value) for value in times],
                "median_seconds": _median(times),
                "cut_median": statistics.median(cut_values) if cut_values else None,
                "records": len(group),
                "valid": len(times),
                "complete": bool(complete),
                "work_comparable": bool(comparable),
                "work_mismatch": not comparable,
                "capability_limited": any(record.get("capability_limited") for record in group),
            }
        baseline_time = curve.get(1, {}).get("median_seconds")
        for n, item in curve.items():
            if baseline_time and item["median_seconds"]:
                item["speedup"] = float(baseline_time / item["median_seconds"])
                item["efficiency"] = float(item["speedup"] / n)
                item["rate"] = float(1.0 / item["median_seconds"])
                item["f_eff"] = None if n == 1 else float((1.0 / item["speedup"] - 1.0 / n) / (1.0 - 1.0 / n))
                item["f_eff_reason"] = "undefined_at_one" if n == 1 else None
            else:
                item["speedup"] = None
                item["efficiency"] = None
                item["rate"] = None
                item["f_eff"] = None
                item["f_eff_reason"] = "missing_baseline" if n > 1 else "undefined_at_one"
        case_curves[case_id] = {
            "case_id": case_id,
            "display_name": case.display_name,
            "solver_case": case.solver_case,
            "node_count": len(case.graph),
            "candidates": curve,
        }
    # Weighted totals over the full case set.
    weights = {}
    for case in plan.base_plan.cases:
        weights[case.case_id] = float(plan.case_weights.get(case.case_id, 1.0))
    aggregate = {}
    for n in plan.values:
        total = 0.0
        complete = True
        cover = 0
        for case in plan.base_plan.cases:
            item = case_curves.get(case.case_id, {}).get("candidates", {}).get(n)
            if item is None or not item.get("complete") or item.get("median_seconds") is None:
                complete = False
                continue
            cover += 1
            total += weights[case.case_id] * float(item["median_seconds"])
        aggregate[n] = {
            "n": n,
            "total_seconds": total if cover else None,
            "covered_cases": cover,
            "required_cases": len(plan.base_plan.cases),
            "complete": bool(complete and cover == len(plan.base_plan.cases)),
            "search_complete": bool(complete and cover == len(plan.base_plan.cases)),
        }
    baseline = aggregate.get(1, {}).get("total_seconds")
    for n, item in aggregate.items():
        if baseline and item["total_seconds"]:
            item["speedup"] = float(baseline / item["total_seconds"])
            item["efficiency"] = float(item["speedup"] / n)
            item["f_eff"] = None if n == 1 else float((1.0 / item["speedup"] - 1.0 / n) / (1.0 - 1.0 / n))
        else:
            item["speedup"] = None
            item["efficiency"] = None
            item["f_eff"] = None
    model_points = [(n, item["total_seconds"]) for n, item in aggregate.items() if item["complete"]]
    fit = fit_amdahl(model_points)
    recommendation = recommend_total_curve(
        list(aggregate.values()),
        tolerance_rel=plan.recommendation_config.get("tolerance_rel", 0.0),
        prefer_lower_resource=plan.recommendation_config.get("prefer_lower_resource", True),
    )
    per_solver = {}
    by_solver = {}
    for case_id, case_curve in case_curves.items():
        by_solver.setdefault(case_curve["solver_case"], []).append(case_curve)
    uncertainty = _bootstrap_totals(
        by_case,
        weights=weights,
        values=plan.values,
        repetitions=plan.repetitions,
        config={**plan.uncertainty_config, **plan.recommendation_config},
    )
    for name, curves in by_solver.items():
        points = []
        for n in plan.values:
            total = 0.0
            complete = True
            for curve in curves:
                item = curve["candidates"].get(n)
                if not item or not item.get("complete") or item.get("median_seconds") is None:
                    complete = False
                    break
                total += item["median_seconds"]
            points.append({"n": n, "total_seconds": total, "complete": complete, "search_complete": complete})
        per_solver[name] = {
            "cases": len(curves),
            "points": points,
            "recommendation": recommend_total_curve(
                points,
                tolerance_rel=plan.recommendation_config.get("tolerance_rel", 0.0),
                prefer_lower_resource=plan.recommendation_config.get("prefer_lower_resource", True),
            ),
            "fit": fit_amdahl([(item["n"], item["total_seconds"]) for item in points if item["complete"]]),
        }
    return {
        "schema_version": 1,
        "record_type": "parallelism_profile",
        "seed": plan.seed,
        "num_treads_values": list(plan.values),
        "case_curves": case_curves,
        "aggregate": aggregate,
        "fit_amdahl": fit,
        "uncertainty": uncertainty,
        "recommendation": recommendation,
        "per_solver": per_solver,
        "work_mismatch": work_mismatch,
        "warmups": len([record for record in records if record.get("role") == "warmup"]),
        "measurements": len([record for record in records if record.get("role") == "measurement"]),
        "references": {"required": len(references), "completed": sum(r.get("status") == "completed" for r in references)},
        "errors": deepcopy(errors),
    }


class ParallelismProfile:
    """Independent profile result with direct analysis attributes."""

    def __init__(self, *, plan, records, references, analysis, errors, environment, budget, wall_seconds, save_status):
        self.plan = plan
        self.records = list(records)
        self.measurements = list(records)
        self.references = list(references)
        self.analysis = analysis
        self.errors = list(errors)
        self.environment = environment
        self.budget = budget
        self.wall_seconds = wall_seconds
        self.save_status = save_status
        self.aggregate = analysis["aggregate"]
        self.per_solver = analysis["per_solver"]
        self.recommended_num_treads = analysis["recommendation"]["recommended_num_treads"]
        self.summary = (
            f"seed={plan.seed} points={plan.points} calls={plan.calls} "
            f"recommended={self.recommended_num_treads} wall={wall_seconds:.3f}s"
        )

    @property
    def measured_records(self):
        return [record for record in self.records if record.get("role") == "measurement"]

    @property
    def warmup_records(self):
        return [record for record in self.records if record.get("role") == "warmup"]

    @property
    def result(self):
        return to_json_data(
            {
                "schema_version": 1,
                "record_type": "parallelism_profile",
                "summary": self.summary,
                "plan": self.plan.result,
                "analysis": self.analysis,
                "records": self.records,
                "references": self.references,
                "errors": self.errors,
                "environment": self.environment,
                "budget": self.budget,
                "timing": {"wall_seconds": self.wall_seconds},
                "save": self.save_status,
            }
        )


class ParallelismProfiler:
    """Run a full-grid CPU sweep at one selected algorithm seed."""

    def __init__(self, spec, config=None):
        self.spec = spec if isinstance(spec, BenchmarkSpec) else BenchmarkSpec(spec)
        self.config = config if isinstance(config, ProfileSpec) else ProfileSpec(config or {})
        self.plan = build_profile_plan(self.spec, self.config)
        self.environment = measurement_environment()
        save = self.config.save
        self.store = ExperimentStore(save, default_filename="parallelism")
        self.records = []
        self.references = []
        self.errors = []
        self._existing = {}
        if self.store.enabled:
            self._existing = self.store.read_trial_records()

    def run(self):
        """Execute all warmups and measured points, then analyze the full plan."""
        limits = self.config.series_limits
        total = self.plan.calls + len(self.plan.base_plan.reference_jobs)
        if limits.get("max_runs_total") is not None and total > limits["max_runs_total"]:
            raise PlanError(
                f"Profile needs {total} top-level calls but max_runs_total={limits['max_runs_total']}."
            )
        budget = CampaignBudget(
            max_runs=limits.get("max_runs_total"),
            shots_total=limits.get("shots_total"),
        )
        started = perf_counter()
        ref_policy = ReferencePolicy(self.spec.data.get("references", {}))
        for job in self.plan.base_plan.reference_jobs:
            if not budget.reserve_run(1):
                self.errors.append({"reference_id": job.reference_id, "reason": "max_runs_total"})
                continue
            record = ref_policy.run_job(job)
            record["environment_id"] = self.environment["environment_id"]
            budget.runs_completed += 1
            self.references.append(record)
            if self.store.enabled:
                self.store.write_record(record)
        # Warmups are real calls but never timing samples.
        for trial in self.plan.warmups:
            self._execute(trial, budget, role="warmup")
        cases = self.plan.base_plan.cases
        for case_index, case in enumerate(cases):
            for repetition in range(self.config.repetitions):
                candidates = list(self.config.values)
                if self.config.order == "shuffle_within_case_repetition":
                    rng = np.random.default_rng([self.config.order_seed, case_index, repetition])
                    rng.shuffle(candidates)
                for order, candidate in enumerate(candidates):
                    trial = next(
                        item
                        for item in self.plan.measured
                        if item.case.case_id == case.case_id
                        and item.repetition_id == repetition
                        and item.num_treads == candidate
                    )
                    record = self._execute(trial, budget, role="measurement")
                    record["measurement_order"] = order
                    record["profile_case_index"] = case_index
                    record["profile_candidate"] = candidate
        wall = perf_counter() - started
        all_records = list(self.plan.warmup_records) + list(self.records)
        analysis = _profile_summary(self.plan, all_records, self.references, self.errors)
        self.records = all_records
        profile = ParallelismProfile(
            plan=self.plan,
            records=self.records,
            references=self.references,
            analysis=analysis,
            errors=self.errors,
            environment=self.environment,
            budget=budget.snapshot(),
            wall_seconds=wall,
            save_status=self.store.snapshot(),
        )
        if self.store.enabled:
            self.store.write_manifest(profile.result)
        return profile

    def _execute(self, trial, budget, *, role):
        if not budget.reserve_run(1):
            record = self._failed_trial(trial, "max_runs_total")
            self.records.append(record)
            return record
        estimate = estimate_trial_shots(trial.case.config)
        if not budget.reserve_shots(estimate):
            budget.release_run(1)
            record = self._failed_trial(trial, "shots_total_reservation")
            self.records.append(record)
            return record
        if trial.trial_id in self._existing and self._existing[trial.trial_id].get("environment_id") == self.environment["environment_id"]:
            record = deepcopy(self._existing[trial.trial_id])
            record["reuse"] = True
        else:
            record = execute_trial(trial)
        used = record.get("resources", {}).get("shots_used", 0)
        budget.settle_shots(estimate, used)
        budget.runs_completed += 1
        record["environment_id"] = self.environment["environment_id"]
        record["profile_role"] = role
        if self.store.enabled:
            self.store.write_record(record)
        if role == "warmup":
            self.plan.warmup_records.append(record)
        else:
            self.records.append(record)
        if record.get("status") == "failed":
            self.errors.append({"trial_id": trial.trial_id, "reason": record.get("termination_reason")})
        return record

    @staticmethod
    def _failed_trial(trial, reason):
        from .runner import _skipped

        return _skipped(trial, reason)


def profile_parallelism(spec, config=None):
    """Convenience wrapper for a complete parallelism profile."""
    return ParallelismProfiler(spec, config).run()
