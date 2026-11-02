"""Analysis plots for reductions and RQAOA parameter trajectories.

The functions read stored result records.  They never run a solver,
optimizer, correlation source, or new measurement.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
import math
from datetime import datetime, timezone

import numpy as np

from .._version import __version__
from ..core.records import RunRecord, resolve_record
from ..core.saving import publish_files, _result_payloads, utc_now
from .plotting import _finish, _text, _theme

__all__ = [
    "TrajectoryPanelCollection",
    "plot_analytic_gamma_profile",
    "plot_ls_coverage",
    "plot_parameter_trajectories",
    "plot_parameter_trajectories_3d",
    "plot_reduction_energy",
    "plot_reduction_weights",
]


def _history(record):
    history = record.get("reduction_history")
    if not history:
        raise ValueError(
            "No reduction history was collected. "
            "Enable collect.reduction_history before solve()."
        )
    return history


def _trajectory(record):
    trajectory = record.get("parameter_trajectory")
    if not trajectory:
        raise ValueError(
            "No parameter trajectory was collected. "
            "Enable collect.parameter_trajectory before solve()."
        )
    layers = int(trajectory[0]["layers"])
    if layers < 1:
        raise ValueError("The trajectory has an invalid layer count.")
    for row in trajectory:
        if int(row.get("layers", -1)) != layers:
            raise ValueError("Trajectory layers change between reductions.")
        if len(row.get("angles", [])) != 2 * layers:
            raise ValueError("A trajectory row has the wrong number of angles.")
    return trajectory, layers


def _angle_label(index, layers):
    if index < layers:
        return rf"$\gamma_{{{index}}}$"
    return rf"$\beta_{{{index - layers}}}$"


def _angle_text(index, layers):
    if index < layers:
        return f"gamma_{index}"
    return f"beta_{index - layers}"


def _angle_unicode(index, layers):
    subscript = "₀₁₂₃₄₅₆₇₈₉"
    power = "⁰¹²³⁴⁵⁶⁷⁸⁹"
    number = index if index < layers else index - layers
    digits = "".join(subscript[int(d)] for d in str(number))
    symbol = "γ" if index < layers else "β"
    return symbol + digits


def _pair_list(layers):
    return list(combinations(range(2 * layers), 2))


def _trajectory_groups(trajectory):
    """Return local-task groups without inventing a global average."""
    if not any("task_id" in row for row in trajectory):
        return [("", list(trajectory))]
    groups = {}
    for row in trajectory:
        groups.setdefault(str(row["task_id"]), []).append(row)
    return list(groups.items())


def plot_ls_coverage(source, config=None):
    """Plot observed allowed-pair coverage for each LS reduction step.

    Parameters
    ----------
    source : solver, RunRecord, or result dict
        A completed LS-RQAOA result with ``coverage_history``.
    config : dict, optional
        Plot style, axes, title, output, and save settings.

    Returns
    -------
    matplotlib.figure.Figure
        Two panels: coverage fraction and allowed/observed pair counts.
        A zero allowed domain is drawn as a gap, not as an invented ratio.
    """
    record = resolve_record(source)
    history = record.get("coverage_history")
    if not history:
        raise ValueError(
            "No LS coverage history is available. Run ls_rqaoa_cpu first."
        )
    config = deepcopy(config or {})
    steps = list(range(1, len(history) + 1))
    fractions = [
        None if row.get("fraction") is None else float(row["fraction"])
        for row in history
    ]
    allowed = [int(row["allowed_pairs"]) for row in history]
    observed = [int(row["observed_pairs"]) for row in history]
    missing = [int(row["missing_pairs"]) for row in history]
    with _theme(config):
        import matplotlib.pyplot as plt

        figsize = config.get("style", {}).get("figsize", [12, 5])
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].plot(
            steps,
            [np.nan if value is None else value for value in fractions],
            color="#2563eb",
            marker="o",
        )
        axes[0].set_ylim(0, 1.05)
        axes[0].set_xlabel(_text("Reduction step", config))
        axes[0].set_ylabel(_text("Observed fraction", config))
        axes[0].set_title(_text("Pair coverage fraction", config))
        axes[1].plot(steps, allowed, color="#64748b", marker="o", label="allowed")
        axes[1].plot(steps, observed, color="#059669", marker="s", label="observed")
        axes[1].plot(steps, missing, color="#dc2626", marker="^", label="missing")
        axes[1].set_xlabel(_text("Reduction step", config))
        axes[1].set_ylabel(_text("Pair count", config))
        axes[1].set_title(_text("Allowed, observed, and missing pairs", config))
        axes[1].legend()
        for axis in axes:
            axis.grid(alpha=0.3)
        return _finish(
            fig,
            config,
            record,
            "plot_ls_coverage",
            {
                "steps": steps,
                "fraction": fractions,
                "allowed_pairs": allowed,
                "observed_pairs": observed,
                "missing_pairs": missing,
            },
        )


def plot_reduction_weights(source, config=None):
    """Plot selected weight and mean nonzero weight against reduction step.

    Parameters
    ----------
    source : solver, RunRecord, or result dict
        A completed RQAOA result with ``reduction_history``.
    config : dict, optional
        Plot style, axes, title, output, and save settings.

    Returns
    -------
    matplotlib.figure.Figure
        Two panels.  Missing mean values are shown as gaps, not zero.
    """
    record = resolve_record(source)
    history = _history(record)
    config = deepcopy(config or {})
    steps = [int(row["step"]) for row in history]
    selected = [float(row["selected_weight"]) for row in history]
    mean = [
        None if row.get("mean_abs_weight") is None else float(row["mean_abs_weight"])
        for row in history
    ]
    with _theme(config):
        import matplotlib.pyplot as plt

        figsize = config.get("style", {}).get("figsize", [12, 5])
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes = np.asarray(axes).reshape(-1)
        axes[0].plot(steps, selected, color="#2563eb", marker="o")
        axes[0].set_xlabel(_text("Reduction step", config))
        axes[0].set_ylabel(_text("Selected weight $w_t$", config))
        axes[0].set_title(_text("Selected pair weight", config))
        axes[1].plot(
            steps,
            [np.nan if value is None else value for value in mean],
            color="#d97706",
            marker="o",
        )
        axes[1].set_xlabel(_text("Reduction step", config))
        axes[1].set_ylabel(_text(r"Mean $|w|$", config))
        axes[1].set_title(_text("Mean absolute nonzero weight", config))
        for axis in axes:
            axis.grid(alpha=0.3)
        return _finish(
            fig,
            config,
            record,
            "plot_reduction_weights",
            {"steps": steps, "selected_weight": selected, "mean_abs_weight": mean},
        )


def plot_reduction_energy(source, config=None):
    """Plot fixed, expected, cumulative, offset, and Ising constants.

    Parameters
    ----------
    source : solver, RunRecord, or result dict
        A completed RQAOA result with ``reduction_history``.
    config : dict, optional
        Plot style, axes, title, output, and save settings.

    Returns
    -------
    matplotlib.figure.Figure
        Four labeled panels.  Empty expected values are gaps.
    """
    record = resolve_record(source)
    history = _history(record)
    config = deepcopy(config or {})
    steps = [int(row["step"]) for row in history]
    fixed = [float(row["fixed_energy"]) for row in history]
    expected = [
        None if row.get("expected_fixed_energy") is None else float(row["expected_fixed_energy"])
        for row in history
    ]
    cumulative = np.cumsum(fixed)
    cumulative_expected = []
    total = 0.0
    for value in expected:
        if value is None:
            cumulative_expected.append(np.nan)
        else:
            total += value
            cumulative_expected.append(total)
    offsets = [float(row["cut_offset"]) for row in history]
    constants = [float(row["ising_constant"]) for row in history]

    with _theme(config):
        import matplotlib.pyplot as plt

        figsize = config.get("style", {}).get("figsize", [12, 8])
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = np.asarray(axes).reshape(-1)
        axes[0].plot(steps, fixed, color="#2563eb", marker="o", label="fixed")
        axes[0].plot(
            steps,
            [np.nan if value is None else value for value in expected],
            color="#d97706",
            marker="s",
            label="expected",
        )
        axes[0].set_xlabel(_text("Reduction step", config))
        axes[0].set_ylabel(_text("Energy contribution", config))
        axes[0].set_title(_text("Selected-edge contributions", config))
        axes[0].legend()
        axes[1].plot(steps, cumulative, color="#2563eb", marker="o", label="fixed sum")
        axes[1].plot(
            steps,
            cumulative_expected,
            color="#d97706",
            marker="s",
            label="expected sum",
        )
        axes[1].set_xlabel(_text("Reduction step", config))
        axes[1].set_ylabel(_text("Cumulative contribution", config))
        axes[1].set_title(_text("Cumulative event sums", config))
        axes[1].legend()
        axes[2].plot(steps, offsets, color="#059669", marker="o")
        axes[2].set_xlabel(_text("Reduction step", config))
        axes[2].set_ylabel(_text("Cut offset $b_t$", config))
        axes[2].set_title(_text("Accumulated cut offset", config))
        axes[3].plot(steps, constants, color="#7c3aed", marker="o")
        axes[3].set_xlabel(_text("Reduction step", config))
        axes[3].set_ylabel(_text("Full Ising constant $c_t$", config))
        axes[3].set_title(_text("Full Ising constant", config))
        for axis in axes:
            axis.grid(alpha=0.3)
        return _finish(
            fig,
            config,
            record,
            "plot_reduction_energy",
            {
                "steps": steps,
                "fixed_energy": fixed,
                "expected_fixed_energy": expected,
                "cumulative_fixed": cumulative.tolist(),
                "cut_offset": offsets,
                "ising_constant": constants,
            },
        )


class TrajectoryPanelCollection:
    """Hold several Matplotlib pages of parameter-trajectory panels."""

    def __init__(self, figures):
        self.figures = tuple(figures)

    def __len__(self):
        return len(self.figures)

    def __iter__(self):
        return iter(self.figures)

    def __getitem__(self, index):
        return self.figures[index]

    @property
    def qa_saved_files(self):
        """Return saved-file records for all pages that had a save section."""
        return tuple(
            getattr(figure, "qa_saved_files", None) for figure in self.figures
        )


def plot_parameter_trajectories(source, config=None):
    """Plot every unordered pair of the ``2*p`` parameters in 2D.

    Parameters
    ----------
    source : solver, RunRecord, or result dict
        A completed RQAOA result with ``parameter_trajectory``.
    config : dict, optional
        Style, axes, title, panels, output, and save settings.  When
        ``layout.panels_per_page`` is a positive integer, the pair grid is
        split into deterministic pages.  All pairs remain present.

    Returns
    -------
    matplotlib.figure.Figure or TrajectoryPanelCollection
        One panel for every pair.  For p=1, 2, 3 the counts are 1, 6, 15.
        Multiple pages are returned as ``TrajectoryPanelCollection``.
    """
    record = resolve_record(source)
    trajectory, layers = _trajectory(record)
    config = deepcopy(config or {})
    pairs = _pair_list(layers)
    steps = [int(row["reduction_step"]) for row in trajectory]
    panels_per_page = config.get("layout", {}).get("panels_per_page")
    if panels_per_page is None:
        chunks = [pairs]
    else:
        if (
            isinstance(panels_per_page, bool)
            or not isinstance(panels_per_page, int)
            or panels_per_page < 1
        ):
            raise ValueError("layout.panels_per_page must be a positive integer.")
        chunks = [
            pairs[start : start + panels_per_page]
            for start in range(0, len(pairs), panels_per_page)
        ]
    pages = []
    for page_index, chunk in enumerate(chunks, start=1):
        page_config = deepcopy(config)
        if len(chunks) > 1:
            save = page_config.get("save")
            if save is not None:
                save["filename"] = f"{save['filename']}_page_{page_index:03d}"
        ncols = min(3, len(chunk))
        nrows = int(math.ceil(len(chunk) / ncols))
        style = page_config.setdefault("style", {})
        if "figsize" not in style:
            style["figsize"] = [max(10.0, 4.0 * ncols), max(6.0, 3.0 * nrows)]
        with _theme(page_config):
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(nrows, ncols, squeeze=False)
            flat = axes.reshape(-1)
            groups = _trajectory_groups(trajectory)
            for axis, (i, j) in zip(flat, chunk):
                for group_name, rows in groups:
                    x = [float(row["angles"][i]) for row in rows]
                    y = [float(row["angles"][j]) for row in rows]
                    axis.plot(
                        x,
                        y,
                        marker="o",
                        linewidth=1.2,
                        label=group_name if group_name else None,
                    )
                    axis.scatter([x[0]], [y[0]], color="#059669", zorder=3)
                    axis.scatter([x[-1]], [y[-1]], color="#d97706", zorder=3)
                if len(groups) > 1:
                    axis.legend(fontsize=7, title="local task")
                axis.set_xlabel(_angle_label(i, layers))
                axis.set_ylabel(_angle_label(j, layers))
                axis.set_title(
                    f"{_angle_label(i, layers)} vs {_angle_label(j, layers)}"
                )
                axis.grid(alpha=0.3)
            for axis in flat[len(chunk) :]:
                axis.set_visible(False)
            title = _text(
                config.get("title", "RQAOA parameter trajectories"),
                config,
            )
            if len(chunks) > 1:
                title = f"{title} (page {page_index}/{len(chunks)})"
            fig.suptitle(title)
            pages.append(
                _finish(
                    fig,
                    page_config,
                    record,
                    "plot_parameter_trajectories",
                    {
                        "steps": steps,
                        "pairs": [[i, j] for i, j in chunk],
                        "page": page_index,
                        "page_count": len(chunks),
                    },
                )
            )
    return pages[0] if len(pages) == 1 else TrajectoryPanelCollection(pages)


def plot_analytic_gamma_profile(source, config=None):
    """Plot one saved analytic gamma profile and its exact-beta angles.

    Parameters
    ----------
    source : solver, RunRecord, or result dict
        A completed analytic RQAOA result with
        ``local_optimizer_history`` collected.
    config : dict, optional
        ``selection.reduction_step`` chooses the reduction (default 1).
        Style, title, output, and save settings use the common format.

    Returns
    -------
    matplotlib.figure.Figure
        Two panels: profile energy and ``beta_star`` against gamma.
    """
    record = resolve_record(source)
    histories = record.get("local_optimizer_history")
    if not histories:
        raise ValueError(
            "No local optimizer history was collected. Enable "
            "collect.local_optimizer_history before solve()."
        )
    config = deepcopy(config or {})
    wanted = int(config.get("selection", {}).get("reduction_step", 1))
    row = next(
        (item for item in histories if int(item.get("reduction_step", -1)) == wanted),
        None,
    )
    if row is None:
        raise ValueError(f"No local history for reduction step {wanted}.")
    profile = row.get("gamma_profile")
    if not profile:
        raise ValueError("The selected local history has no analytic gamma profile.")
    gammas = [float(value) for value in profile["gamma_values"]]
    energies = [float(value) for value in profile.get(
        "profile_energy_original" if profile.get("objective_reference") == "original_graph" else "profile_energy",
        [],
    )]
    betas = [float(value) for value in profile["beta_star"]]
    if len(gammas) != len(energies) or len(gammas) != len(betas):
        raise ValueError("The saved gamma profile arrays have different lengths.")
    with _theme(config):
        import matplotlib.pyplot as plt

        figsize = config.get("style", {}).get("figsize", [12, 7])
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        axes[0].plot(gammas, energies, color="#2563eb", marker="o", markersize=3)
        axes[0].set_xlabel(_text(r"$\gamma$", config))
        axes[0].set_ylabel(_text("Mean energy", config))
        axes[0].set_title(_text("Analytic p=1 gamma profile", config))
        best = int(profile["best_index"])
        axes[0].scatter(
            [gammas[best]], [energies[best]], color="#d97706", zorder=3, label="best grid point"
        )
        axes[0].legend()
        axes[1].plot(gammas, betas, color="#059669", marker="o", markersize=3)
        axes[1].set_xlabel(_text(r"$\gamma$", config))
        axes[1].set_ylabel(_text(r"$\beta_*(\gamma)$", config))
        axes[1].set_title(_text("Exact beta minimizer", config))
        for axis in axes:
            axis.grid(alpha=0.3)
        return _finish(
            fig,
            config,
            record,
            "plot_analytic_gamma_profile",
            {
                "reduction_step": wanted,
                "gamma": gammas,
                "energy": energies,
                "beta_star": betas,
                "best_index": best,
            },
        )


def _finish_plotly(fig, config, source, kind, plot_data):
    """Save one autonomous Plotly HTML file with the common save protocol."""
    save = config.get("save")
    if save is not None:
        fmt = save.get("format", "html").lower()
        if fmt != "html":
            raise ValueError("Interactive 3D trajectories require format='html'.")
        if save.get("metadata", False):
            record = source
            graph_id = None
            if isinstance(record, RunRecord) and record.get("graph"):
                graph_id = record.graph.get("fingerprint")
            caption = (
                f"QA_max_cut_lib {__version__} | {kind} | "
                f"{datetime.now(timezone.utc).isoformat()} | "
                f"graph={graph_id}"
            )
            fig.add_annotation(
                text=caption,
                xref="paper",
                yref="paper",
                x=0.0,
                y=-0.12,
                showarrow=False,
                font={"size": 9},
            )
        html = fig.to_html(
            include_plotlyjs=True,
            full_html=True,
            config={"displaylogo": False, "responsive": True},
        )
        filename = save["filename"] + ".html"
        payloads = {filename: html.encode("utf-8")}
        primary = filename
        source_fields = (
            {}
            if source is None
            else {
                key: source.get(key)
                for key in ("algorithm_type", "run_id", "config", "graph", "finished_at")
                if key in source.field_names
            }
        )
        tool_record = RunRecord(
            {
                "schema_version": 1,
                "algorithm_type": kind,
                "package_version": __version__,
                "status": "completed",
                "finished_at": utc_now(),
                "config": deepcopy(config),
                "source": source_fields,
                "graph": source_fields.get("graph"),
                "plot_data": plot_data,
            }
        )
        if "atts" in save or "save_original_data" in save:
            extra, primary = _result_payloads(tool_record, save)
            payloads.update(extra)
        saved = publish_files(save, payloads, primary)
        fig.layout.meta = {
            "qa_saved_files": [str(path) for path in saved.files],
            "qa_primary": str(saved.primary),
            "qa_record": tool_record.result,
        }
        try:
            object.__setattr__(fig, "qa_saved_files", saved)
        except (AttributeError, TypeError):
            pass
    return fig


def plot_parameter_trajectories_3d(source, config=None):
    """Build an autonomous interactive 3D trajectory collection.

    Parameters
    ----------
    source : solver, RunRecord, or result dict
        A completed RQAOA result with ``parameter_trajectory``.
    config : dict, optional
        Plot style, title, and save settings.  ``save.format`` must be
        ``html`` when saving.

    Returns
    -------
    plotly.graph_objects.Figure
        One 3D trace for every unordered parameter pair.  Reduction step
        is always the Z coordinate.  Plotly.js and all data are embedded
        in the saved HTML file.
    """
    import plotly.graph_objects as go

    record = resolve_record(source)
    trajectory, layers = _trajectory(record)
    config = deepcopy(config or {})
    pairs = _pair_list(layers)
    groups = _trajectory_groups(trajectory)
    traces = []
    pair_trace_indices = []
    for i, j in pairs:
        indices = []
        for group_name, rows in groups:
            x = [float(row["angles"][i]) for row in rows]
            y = [float(row["angles"][j]) for row in rows]
            z = [int(row["reduction_step"]) for row in rows]
            suffix = f" | {group_name}" if group_name else ""
            indices.append(len(traces))
            traces.append(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines+markers",
                    name=f"{_angle_unicode(i, layers)}–{_angle_unicode(j, layers)}{suffix}",
                    hovertemplate=(
                        f"{_angle_unicode(i, layers)}=%{{x:.6g}}<br>"
                        f"{_angle_unicode(j, layers)}=%{{y:.6g}}<br>"
                        "reduction step=%{z}<extra></extra>"
                    ),
                )
            )
        pair_trace_indices.append(indices)
    fig = go.Figure(data=traces)
    n_traces = len(traces)
    buttons = []
    for pair_index, (i, j) in enumerate(pairs):
        visible = [position in pair_trace_indices[pair_index] for position in range(n_traces)]
        buttons.append(
            {
                "label": f"{_angle_unicode(i, layers)}–{_angle_unicode(j, layers)}",
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "scene.xaxis.title.text": _angle_unicode(i, layers),
                        "scene.yaxis.title.text": _angle_unicode(j, layers),
                        "title.text": (
                            f"RQAOA trajectory: {_angle_unicode(i, layers)} vs "
                            f"{_angle_unicode(j, layers)}"
                        ),
                    },
                ],
            }
        )
    fig.update_layout(
        title=_text(
            config.get("title", "RQAOA parameter trajectories (3D)"),
            config,
        ),
        scene={
            "xaxis_title": _angle_unicode(pairs[0][0], layers),
            "yaxis_title": _angle_unicode(pairs[0][1], layers),
            "zaxis_title": "Reduction step",
        },
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 1.0,
                "xanchor": "right",
                "y": 1.15,
                "yanchor": "top",
            }
        ],
        font={"family": "Computer Modern, serif"},
        legend={"title": {"text": "Parameter pair"}},
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    return _finish_plotly(
        fig,
        config,
        record,
        "plot_parameter_trajectories_3d",
        {
            "steps": [int(row["reduction_step"]) for row in trajectory],
            "pairs": [[i, j] for i, j in pairs],
        },
    )
