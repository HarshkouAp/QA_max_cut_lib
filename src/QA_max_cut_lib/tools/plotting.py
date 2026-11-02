"""Plot completed result data in a local Seaborn/Computer Modern theme."""

from contextlib import contextmanager
from copy import deepcopy
import io
import json
import textwrap
import numpy as np
from ..core.graph import WeightedGraph
from ..core.records import RunRecord, resolve_record
from ..core.saving import publish_files, _result_payloads, utc_now
from .numerics import validate_probabilities


@contextmanager
def _theme(config):
    import matplotlib as mpl
    import seaborn as sns

    style = config.get("style", {})
    engine = style.get("text_engine", "mathtext")
    if engine not in ("latex", "mathtext"):
        raise ValueError("text_engine must be latex or mathtext.")
    rc = {
        **sns.axes_style("whitegrid"),
        **sns.plotting_context("paper"),
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman" if engine == "latex" else "cmr10"],
        "text.usetex": engine == "latex",
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.labelsize": style.get("label_size", 19),
        "xtick.labelsize": style.get("tick_size", 16),
        "ytick.labelsize": style.get("tick_size", 16),
        "legend.fontsize": style.get("legend_size", 15),
        "axes.unicode_minus": False,
        "grid.alpha": 0.3,
        "figure.facecolor": "white",
    }
    if "latex_preamble" in style:
        rc["text.latex.preamble"] = style["latex_preamble"]
    # Restore global plotting settings after both creation and export.
    with mpl.rc_context(rc):
        yield


def _text(text, config):
    if config.get("style", {}).get("text_engine", "mathtext") != "latex":
        return str(text)
    table = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(table.get(char, char) for char in str(text))


def _figure(config):
    import matplotlib.pyplot as plt

    return plt.subplots(figsize=config.get("style", {}).get("figsize", [10, 6]))


def _record_for_graph(source):
    if isinstance(source, WeightedGraph):
        return source.copy(), None
    record = resolve_record(source)
    return WeightedGraph.from_snapshot(record.graph), record


def _finish(fig, config, source, kind, plot_data=None):
    import matplotlib.pyplot as plt

    save = config.get("save")
    source_fields = (
        {}
        if source is None
        else {
            key: source.get(key)
            for key in ("algorithm_type", "run_id", "config", "graph", "finished_at")
            if key in source.field_names
        }
    )
    record = RunRecord(
        {
            "schema_version": 1,
            "algorithm_type": kind,
            "package_version": "0.1.0a1",
            "status": "completed",
            "finished_at": utc_now(),
            "config": deepcopy(config),
            "source": source_fields,
            "graph": source_fields.get("graph"),
            "plot_data": plot_data or {},
        }
    )
    if save and save.get("metadata", False):
        provenance = (
            source_fields
            if source_fields
            else {"algorithm_type": kind, "finished_at": utc_now()}
        )
        raw = json.dumps(provenance, sort_keys=True, ensure_ascii=True)
        # The graph snapshot includes all edge data. The visible caption
        # needs the task parameters and identity, not a full edge table.
        if provenance.get("graph"):
            graph = provenance["graph"]
            provenance = deepcopy(provenance)
            provenance["graph"] = {
                "nodes": len(graph["nodes"]),
                "edges": len(graph["edges"]),
                "fingerprint": graph["fingerprint"],
                "recipe": graph.get("recipe"),
            }
            raw = json.dumps(provenance, sort_keys=True, ensure_ascii=True)
        lines = textwrap.wrap(raw, width=125, break_long_words=True)
        caption_height = max(0.5, 0.13 * len(lines))
        width, height = fig.get_size_inches()
        fig.set_size_inches(width, height + caption_height)
        fig.tight_layout(rect=(0, caption_height / (height + caption_height), 1, 1))
        fig.text(0.02, 0.012, _text("\n".join(lines), config), fontsize=7, va="bottom")
    else:
        fig.tight_layout()
    fig.qa_record = record
    if save is not None:
        fmt = save.get("format", "png").lower()
        if fmt not in ("png", "jpeg", "pdf", "svg"):
            raise ValueError(
                "This milestone exports static png, jpeg, pdf, or svg. Interactive HTML is pending."
            )
        dpi = save.get("DPI", 250)
        if (
            isinstance(dpi, bool)
            or not isinstance(dpi, (float, int))
            or not np.isfinite(dpi)
            or dpi <= 0
        ):
            raise ValueError("DPI must be a finite positive number.")
        stream = io.BytesIO()
        fig.savefig(
            stream,
            format=fmt,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.1,
            facecolor="white",
        )
        filename = save["filename"] + "." + fmt
        payloads = {filename: stream.getvalue()}
        primary = filename
        if "atts" in save or "save_original_data" in save:
            extra, primary = _result_payloads(record, save)
            payloads.update(extra)
        fig.qa_saved_files = publish_files(save, payloads, primary)
    # Force text rendering inside the same theme context as creation.
    # A missing TeX installation is an explicit rendering error.
    fig.canvas.draw()
    if config.get("output", {}).get("show", False):
        plt.show()
    return fig


def _plot_graph(source, config, solution, info):
    import networkx as nx

    graph, record = _record_for_graph(source)
    network = nx.Graph()
    network.add_nodes_from(graph.nodes)
    network.add_weighted_edges_from(graph.edges)
    layout = config.get("graph", {}).get("layout", "spring")
    if layout == "spring":
        positions = nx.spring_layout(
            network, seed=config.get("graph", {}).get("seed", 0), weight=None
        )
    elif layout == "circular":
        positions = nx.circular_layout(network)
    else:
        raise ValueError("layout must be spring or circular.")
    if solution is not None:
        graph.cut_value(solution)
    with _theme(config):
        fig, ax = _figure(config)
        colors = [
            "#2563eb" if solution is None or solution[node] == 1 else "#d97706"
            for node in graph.nodes
        ]
        nx.draw_networkx_nodes(
            network,
            positions,
            ax=ax,
            node_color=colors,
            edgecolors="#202020",
            node_size=550,
        )
        nx.draw_networkx_labels(
            network,
            positions,
            labels={n: str(n) for n in graph.nodes},
            ax=ax,
            font_color="white",
            font_family="serif",
            font_size=12,
        )
        for u, v, weight in graph.edges:
            crossing = solution is not None and solution[u] != solution[v]
            nx.draw_networkx_edges(
                network,
                positions,
                ax=ax,
                edgelist=[(u, v)],
                width=2.4 if crossing else 1.0,
                edge_color="#2563eb" if crossing else "#737373",
                style="dashed" if weight < 0 else "solid",
                alpha=0.9 if crossing else 0.6,
            )
        if config.get("graph", {}).get("weight_labels", True):
            nx.draw_networkx_edge_labels(
                network,
                positions,
                ax=ax,
                edge_labels={(u, v): f"{w:g}" for u, v, w in graph.edges},
                font_size=10,
            )
        ax.set_title(
            _text(
                config.get(
                    "title",
                    "Weighted graph" if solution is None else "Max-Cut solution",
                ),
                config,
            )
        )
        if solution is not None:
            ax.text(
                0.01,
                0.01,
                _text(
                    f"E = {-graph.cut_value(solution):.6g}; C = {graph.cut_value(solution):.6g}; blue: +1; orange: -1; dashed: negative weight",
                    config,
                ),
                transform=ax.transAxes,
                fontsize=10,
            )
        if info:
            # Keep the title above the statistics line. Both texts are
            # outside the graph axes and need separate vertical space.
            ax.set_title(ax.get_title(), pad=32)
            weights = np.array([e[2] for e in graph.edges], dtype=float)
            text = f"N={len(graph)}; edges={len(graph.edges)}; density={nx.density(network):.3g}; "
            text += f"components={nx.number_connected_components(network)}; isolated={len(list(nx.isolates(network)))}"
            if len(weights):
                text += f"; weights=[{weights.min():g}, {weights.max():g}]; mean |w|={np.abs(weights).mean():.3g}"
            ax.text(
                0.5,
                1.02,
                _text(text, config),
                transform=ax.transAxes,
                ha="center",
                fontsize=10,
            )
        ax.set_axis_off()
        if record is None:
            record = RunRecord(
                {
                    "algorithm_type": "graph",
                    "graph": graph.snapshot(),
                    "config": {},
                    "finished_at": utc_now(),
                }
            )
        return _finish(fig, config, record, "plot_graph", {"solution": solution})


def plot_graph(source, config=None):
    """Draw a graph or the input snapshot of a completed solver."""
    return _plot_graph(source, deepcopy(config or {}), None, False)


def plot_graph_solution(source, solution=None, config=None):
    """Draw spins and cut edges. Accept a graph plus an explicit solution."""
    if solution is None:
        solution = resolve_record(source).solution
    return _plot_graph(source, deepcopy(config or {}), solution, False)


def plot_graph_info(source, config=None):
    """Draw the graph with node, edge, connectivity, and weight statistics."""
    return _plot_graph(source, deepcopy(config or {}), None, True)


def plot_probabilities(source, config=None):
    """Draw basis-state probabilities without renormalizing a top-K view.

    config.selection.top_k limits the number of visible states; the
    default is 16. The title reports visible and remaining mass.
    """
    import seaborn as sns

    record = resolve_record(source)
    config = deepcopy(config or {})
    if record.get("probabilities") is None:
        raise ValueError(
            "No probabilities were collected. Enable collect.probabilities before solve()."
        )
    p = validate_probabilities(record.probabilities)
    k = config.get("selection", {}).get("top_k", 16)
    if type(k) is not int or k < 1:
        raise ValueError("top_k must be a positive integer.")
    order = np.argsort(-p, kind="stable")[:k]
    n = (len(p) - 1).bit_length()
    labels = [format(int(index), f"0{n}b")[::-1] for index in order]
    with _theme(config):
        fig, ax = _figure(config)
        sns.barplot(x=labels, y=p[order], color="#2563eb", ax=ax, errorbar=None)
        ax.set_xlabel("Bits in node order (leftmost bit = first node)")
        ax.set_ylabel("Probability")
        ax.tick_params(axis="x", labelsize=10, rotation=45)
        ax.set_title(
            _text(
                config.get(
                    "title",
                    f"State probabilities: shown mass={p[order].sum():.4f}; remainder={1-p[order].sum():.4f}",
                ),
                config,
            )
        )
        return _finish(
            fig,
            config,
            record,
            "plot_probabilities",
            {"indices": order, "probabilities": p[order]},
        )


def plot_optimization(source, config=None):
    """Plot accepted steps from optimizer_history. Never rerun training."""
    config = deepcopy(config or {})
    record = resolve_record(source)
    history = record.optimizer_history
    if not history:
        raise ValueError(
            "No optimization steps were collected. Enable collect.optimizer_history before solve()."
        )
    x = config.get("axes", {}).get("x", "optimizer_step")
    y = config.get("axes", {}).get("y", "optimizer_loss")
    if x not in ("optimizer_step", "objective_evaluations", "shots_used"):
        raise ValueError("Unsupported optimization x coordinate.")
    if y not in ("objective_value", "optimizer_loss", "gradient_norm"):
        raise ValueError("Unsupported optimization y coordinate.")
    with _theme(config):
        fig, ax = _figure(config)
        ax.plot(
            [row[x] for row in history],
            [row[y] for row in history],
            color="#2563eb",
            marker="o",
            markersize=3,
        )
        labels = {
            "optimizer_step": "Optimization step",
            "objective_evaluations": "Objective evaluations",
            "shots_used": "Training shots",
        }
        ylabel = {
            "objective_value": "Objective value",
            "gradient_norm": "Gradient norm",
        }.get(y, y)
        if y == "optimizer_loss":
            objective = record.config.get("objective", {}).get("type", "mean_cut")
            ylabel = (
                "Energy expectation"
                if objective == "mean_cut"
                else "Loss (minus objective)"
            )
        if (
            y == "optimizer_loss"
            and record.config.get("estimation", {}).get("mode") == "shots"
            and objective == "mean_cut"
        ):
            ylabel = "Estimated mean energy"
        ax.set_xlabel(_text(labels[x], config))
        ax.set_ylabel(_text(ylabel, config))
        ax.set_title(_text(config.get("title", "QAOA optimization"), config))
        return _finish(fig, config, record, "plot_optimization", {"history": history})


def plot_landscape(source, config=None):
    """Plot a p=1 landscape with gamma on X and beta on Y."""
    config = deepcopy(config or {})
    record = resolve_record(source)
    with _theme(config):
        fig, ax = _figure(config)
        image = ax.pcolormesh(
            record.gamma, record.beta, record.values, shading="auto", cmap="viridis"
        )
        ax.set_xlabel(r"$\gamma_0$")
        ax.set_ylabel(r"$\beta_0$")
        ax.set_title(_text(config.get("title", "QAOA objective landscape"), config))
        label = (
            "Objective value"
            if record.config.get("quantity") == "objective_value"
            else "Loss (minus objective)"
        )
        if (
            record.config.get("objective", {}).get("type") == "mean_cut"
            and label != "Objective value"
        ):
            label = "Energy expectation"
        fig.colorbar(image, ax=ax, label=label)
        return _finish(
            fig,
            config,
            record,
            "plot_landscape",
            {"gamma": record.gamma, "beta": record.beta, "values": record.values},
        )
