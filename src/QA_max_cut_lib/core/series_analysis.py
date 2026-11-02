"""Quality metrics and series summaries computed from saved records."""

from __future__ import annotations

from copy import deepcopy
import math
import statistics

__all__ = ["annotate_trial_quality", "analyze_series", "reference_index"]


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def reference_index(references):
    """Index reference records by graph content and meaning."""
    result = {}
    for record in references:
        if record.get("status") != "completed" or not _finite(record.get("cut_value")):
            continue
        gid = record.get("graph_content_id")
        kind = record.get("kind")
        if not gid:
            continue
        entry = result.setdefault(gid, {})
        if kind == "exact_max_cut":
            entry["exact_max"] = record
        elif kind == "exact_min_cut":
            entry["exact_min"] = record
        elif kind in ("gw_rounded_cut", "best_known_cut"):
            entry.setdefault("heuristic", record)
        elif record.get("proof") == "certified":
            entry["certified_upper_bound"] = record
    return result


def annotate_trial_quality(record, references, metrics_config=None):
    """Return a copy with C/E and reference-relative metrics.

    Exact gaps use ``C_star - C``.  A heuristic reference uses
    ``C - C_ref`` and never labels that difference as an optimality gap.
    Range normalisation requires both proven extrema.
    """
    result = deepcopy(record)
    cut = record.get("cut_value")
    result["metrics"] = {
        "cut_value": cut,
        "solution_energy": record.get("solution_energy"),
        "reference_kind": None,
        "reference_cut": None,
        "optimality_gap": None,
        "cut_minus_reference": None,
        "ratio_to_exact_max": None,
        "ratio_reason": None,
        "q_range": None,
        "q_range_reason": None,
        "success": None,
        "success_reason": None,
    }
    if not _finite(cut):
        result["metrics"]["ratio_reason"] = "no_valid_solution"
        result["metrics"]["success_reason"] = "no_valid_solution"
        return result
    found = references.get(record.get("graph_content_id"), {})
    exact_max = found.get("exact_max")
    exact_min = found.get("exact_min")
    heuristic = found.get("heuristic")
    if exact_max is not None:
        star = float(exact_max["cut_value"])
        result["metrics"].update(
            reference_kind="exact_max_cut",
            reference_cut=star,
            optimality_gap=float(star - cut),
        )
        if star > 0:
            result["metrics"]["ratio_to_exact_max"] = float(cut / star)
        else:
            result["metrics"]["ratio_reason"] = "zero_exact_max"
    elif heuristic is not None:
        ref = float(heuristic["cut_value"])
        result["metrics"].update(
            reference_kind=heuristic.get("kind"),
            reference_cut=ref,
            cut_minus_reference=float(cut - ref),
        )
    else:
        result["metrics"]["ratio_reason"] = "no_reference"
    if exact_max is not None and exact_min is not None:
        low = float(exact_min["cut_value"])
        high = float(exact_max["cut_value"])
        if high != low:
            result["metrics"]["q_range"] = float((cut - low) / (high - low))
        else:
            result["metrics"]["q_range_reason"] = "constant_objective"
    else:
        result["metrics"]["q_range_reason"] = "requires_both_exact_extrema"

    success = (metrics_config or {}).get("success")
    if success is not None:
        kind = success.get("type")
        if kind == "fraction_of_exact_max":
            fraction = success.get("fraction")
            if exact_max is None or not _finite(fraction):
                result["metrics"]["success_reason"] = "requires_exact_max"
            else:
                star = float(exact_max["cut_value"])
                if star <= 0:
                    result["metrics"]["success_reason"] = "degenerate_exact_max"
                else:
                    result["metrics"]["success"] = bool(cut >= float(fraction) * star)
        elif kind == "absolute_cut":
            threshold = success.get("threshold")
            if not _finite(threshold):
                result["metrics"]["success_reason"] = "missing_absolute_threshold"
            else:
                result["metrics"]["success"] = bool(cut >= float(threshold))
        else:
            result["metrics"]["success_reason"] = "unknown_success_type"
    return result


def _summary(values):
    values = [float(value) for value in values if _finite(value)]
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "stdev": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _group(records, key):
    groups = {}
    for record in records:
        name = record.get(key) or "unknown"
        groups.setdefault(name, []).append(record)
    return groups


def analyze_series(trials, references, *, metrics_config=None, plan=None, errors=None):
    """Analyze independent records without running a solver.

    Parameters
    ----------
    trials : list of dict
        Trial records, including failed and skipped attempts.
    references : list of dict
        Reference records.
    metrics_config : dict, optional
        The benchmark ``metrics`` section.
    plan : dict, optional
        Plan summary for denominators.
    errors : list, optional
        Campaign errors.

    Returns
    -------
    dict
        Per-trial metrics, planned/valid/failed/skipped counts, conditional
        success, per-solver groups, reference coverage, and a manifest-ready
        summary.
    """
    refs = reference_index(references)
    annotated = [annotate_trial_quality(record, refs, metrics_config) for record in trials]
    valid = [
        record
        for record in annotated
        if record.get("status") in ("completed", "stopped")
        and _finite(record.get("cut_value"))
    ]
    failed = [record for record in annotated if record.get("status") == "failed"]
    skipped = [
        record
        for record in annotated
        if record.get("status") in ("skipped", "rejected", "pending")
    ]
    success_values = [
        record["metrics"]["success"]
        for record in valid
        if record["metrics"].get("success") is not None
    ]
    gaps = [
        record["metrics"]["optimality_gap"]
        for record in valid
        if _finite(record["metrics"].get("optimality_gap"))
    ]
    cuts = [record.get("cut_value") for record in valid]
    by_solver = {}
    for name, group in _group(annotated, "solver_case").items():
        group_valid = [
            item
            for item in group
            if item.get("status") in ("completed", "stopped") and _finite(item.get("cut_value"))
        ]
        group_success = [
            item["metrics"]["success"]
            for item in group_valid
            if item["metrics"].get("success") is not None
        ]
        by_solver[name] = {
            "planned": len(group),
            "valid": len(group_valid),
            "failed": sum(item.get("status") == "failed" for item in group),
            "skipped": sum(item.get("status") in ("skipped", "rejected", "pending") for item in group),
            "cut": _summary([item.get("cut_value") for item in group_valid]),
            "gap": _summary(
                [
                    item["metrics"].get("optimality_gap")
                    for item in group_valid
                    if _finite(item["metrics"].get("optimality_gap"))
                ]
            ),
            "success": (sum(bool(value) for value in group_success) / len(group_success))
            if group_success
            else None,
        }
    return {
        "schema_version": 1,
        "record_type": "series_analysis",
        "planned": len(trials),
        "valid": len(valid),
        "failed": len(failed),
        "skipped": len(skipped),
        "stopped": sum(record.get("status") == "stopped" for record in annotated),
        "success_conditional_on_valid": (sum(bool(v) for v in success_values) / len(success_values))
        if success_values
        else None,
        "success_count": sum(bool(v) for v in success_values),
        "success_defined": len(success_values),
        "cut": _summary(cuts),
        "optimality_gap": _summary(gaps),
        "best_observed": max(cuts) if cuts else None,
        "by_solver": by_solver,
        "references": {
            "records": len(references),
            "completed": sum(record.get("status") == "completed" for record in references),
            "failed": sum(record.get("status") == "failed" for record in references),
            "kinds": {kind: sum(record.get("kind") == kind for record in references) for kind in sorted({record.get("kind") for record in references})},
        },
        "errors": deepcopy(errors or []),
        "trials": annotated,
    }
