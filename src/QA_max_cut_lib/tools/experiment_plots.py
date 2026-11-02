"""Plots for experiment series and parallelism profiles."""

from __future__ import annotations

from copy import deepcopy
import base64
import html
import io
import json
import math
from pathlib import Path

import numpy as np

from ..core.records import RunRecord
from ..core.saving import publish_files, utc_now
from .plotting import _finish, _theme

__all__ = [
    "plot_series_coverage",
    "plot_series_quality",
    "plot_parallelism",
    "parallelism_html",
]


def _analysis(source):
    if hasattr(source, "analysis"):
        return deepcopy(source.analysis)
    if isinstance(source, dict) and "analysis" in source:
        return deepcopy(source["analysis"])
    if isinstance(source, dict) and "aggregate" in source:
        return deepcopy(source)
    raise ValueError("Supply a profile/result with an analysis section.")


def _summary_record(source, kind):
    return RunRecord(
        {
            "schema_version": 1,
            "algorithm_type": kind,
            "package_version": "0.1.0a9",
            "status": "completed",
            "finished_at": utc_now(),
            "config": {},
            "graph": None,
            "source": getattr(source, "summary", None),
        }
    )


def _save_variant(fig, config, suffix):
    config = deepcopy(config or {})
    save = config.get("save")
    if save is None:
        return fig
    save = deepcopy(save)
    base = save.get("filename", "figure")
    save["filename"] = f"{base}_{suffix}"
    config["save"] = save
    return _finish(fig, config, _summary_record(None, "plot"), f"experiment_{suffix}")


def plot_series_coverage(source, config=None):
    """Plot planned/valid/failed/skipped counts and per-solver cut quality."""
    import matplotlib.pyplot as plt

    analysis = source.analysis if hasattr(source, "analysis") else source.get("analysis", source)
    config = deepcopy(config or {})
    with _theme(config):
        fig, axes = plt.subplots(1, 2, figsize=config.get("style", {}).get("figsize", [12, 5]))
        labels = ["planned", "valid", "failed", "skipped"]
        values = [analysis.get(key, 0) for key in labels]
        axes[0].bar(labels, values, color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"])
        axes[0].set_title("Campaign coverage")
        axes[0].set_ylabel("trials")
        by_solver = analysis.get("by_solver", {})
        names = list(by_solver)
        medians = [by_solver[name].get("cut", {}).get("median") for name in names]
        positions = np.arange(len(names))
        values = [0.0 if value is None else value for value in medians]
        axes[1].bar(positions, values, color="#4c72b0")
        axes[1].set_xticks(positions, names, rotation=25, ha="right")
        axes[1].set_title("Median cut by solver")
        axes[1].set_ylabel("C")
        return _save_variant(fig, config, "coverage")


def plot_series_quality(source, config=None):
    """Plot valid cut values by solver and reference kind."""
    import matplotlib.pyplot as plt

    if hasattr(source, "trials"):
        trials = source.trials
    elif isinstance(source, dict):
        trials = source.get("trials", [])
    else:
        raise ValueError("No trial records available.")
    analysis = source.analysis if hasattr(source, "analysis") else source.get("analysis", {})
    refs = {}
    for item in (source.references if hasattr(source, "references") else source.get("references", [])):
        refs.setdefault((item.get("graph_content_id"), item.get("kind")), item)
    config = deepcopy(config or {})
    with _theme(config):
        fig, ax = plt.subplots(figsize=config.get("style", {}).get("figsize", [10, 6]))
        groups = {}
        for trial in trials:
            if trial.get("status") not in ("completed", "stopped") or trial.get("cut_value") is None:
                continue
            groups.setdefault(trial.get("solver_case", "unknown"), []).append(float(trial["cut_value"]))
        names = list(groups)
        ax.boxplot([groups[name] for name in names], labels=names)
        ax.set_ylabel("C on original graph")
        ax.set_title("Valid cuts by solver")
        ax.tick_params(axis="x", rotation=25)
        return _save_variant(fig, config, "quality")


def _curve_records(analysis):
    curves = analysis.get("case_curves", {})
    return curves


def plot_parallelism(source, config=None):
    """Draw six views of a parallelism profile.

    The function uses only the saved profile analysis.  It never calls a
    solver and never consumes shots.  The Amdahl line is labelled as a fit,
    not as an exact code fraction.
    """
    import matplotlib.pyplot as plt

    analysis = _analysis(source)
    config = deepcopy(config or {})
    curves = _curve_records(analysis)
    aggregate = analysis.get("aggregate", {})
    ns = sorted(int(n) for n in aggregate)
    with _theme(config):
        fig, axes = plt.subplots(2, 3, figsize=config.get("style", {}).get("figsize", [16, 9]))
        ax = axes[0, 0]
        for case_id, curve in curves.items():
            points = curve.get("candidates", {})
            xs = sorted(points)
            ys = [points[n].get("median_seconds") for n in xs]
            mask = [value is not None for value in ys]
            if any(mask):
                ax.plot(np.asarray(xs)[mask], np.asarray(ys)[mask], color="#9aa0a6", alpha=0.35, linewidth=0.8)
        ys = [aggregate[n].get("total_seconds") for n in ns]
        ax.plot(ns, ys, marker="o", color="#2563eb", linewidth=2.5, label="weighted total")
        ax.set_xlabel("num_treads")
        ax.set_ylabel("seconds")
        ax.set_title("Total time over full profile")
        ax.legend()
        ax = axes[0, 1]
        speedups = [aggregate[n].get("speedup") for n in ns]
        ax.plot(ns, speedups, marker="o", color="#55a868", label="measured speedup")
        ax.plot(ns, ns, linestyle="--", color="#7f7f7f", label="ideal")
        ax.set_xlabel("num_treads")
        ax.set_ylabel("S(n)")
        ax.set_title("Speedup over same implementation at 1")
        ax.legend()
        ax = axes[0, 2]
        eff = [aggregate[n].get("efficiency") for n in ns]
        ax.plot(ns, eff, marker="o", color="#c44e52")
        ax.axhline(1.0, linestyle="--", color="#7f7f7f")
        ax.set_xlabel("num_treads")
        ax.set_ylabel("eta(n)")
        ax.set_title("Efficiency of CPU budget")
        ax = axes[1, 0]
        case_names = [curves[key].get("display_name", key)[:18] for key in curves]
        matrix = []
        for key in curves:
            points = curves[key].get("candidates", {})
            matrix.append([points[n].get("median_seconds") for n in ns])
        matrix = np.asarray(
            [[float(value) if value is not None else np.nan for value in row] for row in matrix],
            dtype=float,
        )
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(ns)), [str(n) for n in ns])
        ax.set_yticks(range(len(case_names)), case_names, fontsize=6)
        ax.set_xlabel("num_treads")
        ax.set_title("Median time by case")
        fig.colorbar(image, ax=ax, label="seconds")
        ax = axes[1, 1]
        for key, curve in curves.items():
            points = curve.get("candidates", {})
            xs = [n for n in ns if points.get(n, {}).get("f_eff") is not None]
            ys = [points[n]["f_eff"] for n in xs]
            if xs:
                ax.plot(xs, ys, marker=".", color="#8172b2", alpha=0.5, linewidth=0.8)
        ax.axhline(0.0, color="#7f7f7f", linewidth=0.8)
        ax.set_xlabel("num_treads")
        ax.set_ylabel("f_eff")
        ax.set_title("Karp-Flatt experimental serial fraction")
        ax = axes[1, 2]
        fit = analysis.get("fit_amdahl", {})
        ax.plot(ns, [aggregate[n].get("total_seconds") for n in ns], marker="o", label="measured")
        if fit.get("a") is not None:
            fitted = [fit["a"] + fit["b"] / n for n in ns]
            ax.plot(ns, fitted, linestyle="--", label="Amdahl fit")
        rec = analysis.get("recommendation", {})
        if rec.get("recommended_num_treads") is not None:
            ax.axvline(rec["recommended_num_treads"], color="#c44e52", linestyle=":", label="recommendation")
        ax.set_xlabel("num_treads")
        ax.set_ylabel("seconds")
        ax.set_title("Global recommendation")
        ax.legend()
        fig.tight_layout()
        return _save_variant(fig, config, "parallelism")


def _figure_png_base64(fig):
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=140, bbox_inches="tight", facecolor="white")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def parallelism_html(source, config):
    """Write an autonomous HTML report for a saved profile.

    Parameters
    ----------
    source : ParallelismProfile or mapping
        A profile result with analysis and measurement records.
    config : dict
        ``save.path``, ``save.filename``, ``makedir``, and optional
        ``format`` are used.  The report embeds a PNG and a JSON table.
    """
    import matplotlib.pyplot as plt

    analysis = _analysis(source)
    config = deepcopy(config or {})
    save = config.get("save")
    if save is None:
        raise ValueError("HTML output needs save.path and save.filename.")
    fig = plot_parallelism(source, {"style": config.get("style", {})})
    png = _figure_png_base64(fig)
    plt.close(fig)
    rows = []
    for n, item in sorted(analysis.get("aggregate", {}).items()):
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(value))}</td>"
                for value in (
                    n,
                    item.get("total_seconds"),
                    item.get("speedup"),
                    item.get("efficiency"),
                    item.get("f_eff"),
                    item.get("complete"),
                )
            )
            + "</tr>"
        )
    body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Parallelism profile</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;color:#222}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:.35rem .6rem}}
img{{max-width:100%;height:auto}}</style></head><body>
<h1>Parallelism profile</h1>
<p>{html.escape(str(getattr(source, "summary", "")))}</p>
<img src="data:image/png;base64,{png}" alt="parallelism profile">
<h2>Aggregate</h2>
<table><thead><tr><th>num_treads</th><th>total seconds</th><th>speedup</th><th>efficiency</th><th>f_eff</th><th>complete</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>Recommendation</h2><pre>{html.escape(json.dumps(analysis.get("recommendation", {}), ensure_ascii=False, indent=2))}</pre>
</body></html>"""
    base = Path
    path = base(save["path"]).expanduser()
    if not path.exists():
        if not save.get("makedir", False):
            raise FileNotFoundError(path)
        path.mkdir(parents=True, exist_ok=True)
    name = str(save.get("filename", "parallelism")) + ".html"
    files = publish_files(
        {"path": str(path), "filename": name, "makedir": True, "savedir": False},
        {name: body.encode("utf-8")},
        name,
    )
    return files
