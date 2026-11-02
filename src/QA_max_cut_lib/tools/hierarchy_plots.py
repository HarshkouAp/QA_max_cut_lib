"""Plots for the QAOA-in-QAOA hierarchy.

The functions read a completed solver, a ``RunRecord``, or a result dict.
They never run a local solver, projection, or new measurement.
"""

from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from ..core.records import resolve_record
from .plotting import _finish, _theme

__all__ = [
    "plot_hierarchy_overview",
    "plot_hierarchy_parameters",
    "plot_hierarchy_tree",
]


def _record(source):
    record = resolve_record(source)
    if record.get("hierarchy_history") is None:
        raise ValueError(
            "This result has no hierarchy_history; set collect.hierarchy_history=true."
        )
    return record


def plot_hierarchy_overview(source, config=None):
    """Plot sizes, offset, calls, and shots for every accepted level."""
    import matplotlib.pyplot as plt

    config = deepcopy(config or {})
    record = _record(source)
    history = record.get("hierarchy_history")
    if not history:
        raise ValueError("The hierarchy history is empty.")
    levels = [int(item["level"]) for item in history]
    before = [len(item["parent_nodes"]) for item in history]
    after = [len(item["nodes_after"]) for item in history]
    blocks = [len(item["blocks"]) for item in history]
    delta = [float(item["delta_cut_offset"]) for item in history]
    shots = [int(item.get("local_tasks_total", 0)) for item in history]
    with _theme(config):
        fig, axes = plt.subplots(2, 2, figsize=config.get("style", {}).get("figsize", [12, 8]))
        ax = axes[0, 0]
        ax.plot(levels, before, marker="o", label="nodes before")
        ax.plot(levels, after, marker="s", label="nodes after")
        ax.set_xlabel("hierarchy level")
        ax.set_ylabel("nodes")
        ax.set_title("Level size")
        ax.legend()
        ax = axes[0, 1]
        ax.bar(levels, blocks, color="#4c72b0")
        ax.set_xlabel("hierarchy level")
        ax.set_ylabel("blocks")
        ax.set_title("Blocks per level")
        ax = axes[1, 0]
        ax.bar(levels, delta, color="#55a868")
        ax.set_xlabel("hierarchy level")
        ax.set_ylabel("Delta b")
        ax.set_title("Cut offset added by a level")
        ax = axes[1, 1]
        ax.bar(levels, shots, color="#c44e52")
        ax.set_xlabel("hierarchy level")
        ax.set_ylabel("local calls")
        ax.set_title("Nontrivial local readouts")
        return _finish(fig, config, record, "qaoa_in_qaoa_hierarchy_overview")


def plot_hierarchy_parameters(source, config=None):
    """Plot local angle parameters by hierarchy level and block.

    Angles from different blocks are never averaged into one global vector.
    """
    import matplotlib.pyplot as plt

    config = deepcopy(config or {})
    record = _record(source)
    trajectory = record.get("parameter_trajectory")
    if not trajectory:
        raise ValueError(
            "This result has no parameter_trajectory; enable it in collect."
        )
    groups = {}
    for item in trajectory:
        angles = item.get("angles")
        if angles is None:
            continue
        key = (int(item["level"]), int(item["block"]))
        groups[key] = [float(value) for value in angles]
    if not groups:
        raise ValueError("The hierarchy has no QAOA parameter vectors.")
    layers = len(next(iter(groups.values()))) // 2
    with _theme(config):
        fig, axes = plt.subplots(1, 2, figsize=config.get("style", {}).get("figsize", [13, 5]))
        for key in sorted(groups):
            values = groups[key]
            axes[0].plot(
                range(layers), values[:layers], marker="o", alpha=0.7,
                label=f"L{key[0]} B{key[1]}",
            )
            axes[1].plot(
                range(layers), values[layers:], marker="o", alpha=0.7,
                label=f"L{key[0]} B{key[1]}",
            )
        axes[0].set_xlabel("QAOA layer")
        axes[0].set_ylabel("gamma")
        axes[0].set_title("Gamma by local block")
        axes[1].set_xlabel("QAOA layer")
        axes[1].set_ylabel("beta")
        axes[1].set_title("Beta by local block")
        axes[1].legend(fontsize=7, ncol=2)
        return _finish(fig, config, record, "qaoa_in_qaoa_parameters")


def plot_hierarchy_parameters_3d(source, config=None):
    """Plot local angle vectors with hierarchy level as the Z coordinate."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    config = deepcopy(config or {})
    record = _record(source)
    trajectory = record.get("parameter_trajectory")
    if not trajectory:
        raise ValueError(
            "This result has no parameter_trajectory; enable it in collect."
        )
    groups = {}
    for item in trajectory:
        angles = item.get("angles")
        if angles is None:
            continue
        groups[(int(item["level"]), int(item["block"]))] = [
            float(value) for value in angles
        ]
    if not groups:
        raise ValueError("The hierarchy has no QAOA parameter vectors.")
    layers = len(next(iter(groups.values()))) // 2
    with _theme(config):
        fig = plt.figure(figsize=config.get("style", {}).get("figsize", [12, 8]))
        ax = fig.add_subplot(111, projection="3d")
        for (level, block), values in sorted(groups.items()):
            xs = list(range(layers))
            ys = [level] * layers
            ax.plot(xs, ys, values[:layers], marker="o", alpha=0.75, label=f"gamma L{level}B{block}")
            ax.plot(xs, ys, values[layers:], marker="s", alpha=0.75, label=f"beta L{level}B{block}")
        ax.set_xlabel("QAOA layer")
        ax.set_ylabel("hierarchy level")
        ax.set_zlabel("angle")
        ax.set_title("Local QAOA parameters by hierarchy level")
        ax.legend(fontsize=6, ncol=2)
        return _finish(fig, config, record, "qaoa_in_qaoa_parameters_3d")


def _walk_tree(nodes, level, parent, positions, edges, max_nodes):
    for node in nodes:
        if len(positions) >= max_nodes:
            return
        key = (level, str(node.get("id")))
        positions[key] = node
        if parent is not None:
            edges.append((parent, key))
        children = node.get("children", [])
        child_nodes = []
        for item in children:
            if "child" in item:
                child_nodes.append(item["child"])
            elif node.get("kind") == "block":
                # The compact tree stores leaf children without a nested node.
                child_nodes.append(
                    {
                        "id": item.get("id"),
                        "kind": "leaf",
                        "node": item.get("id"),
                        "children": [],
                    }
                )
        if child_nodes:
            _walk_tree(child_nodes, level - 1, key, positions, edges, max_nodes)


def plot_hierarchy_tree(source, config=None, *, max_nodes=160, anchor=None):
    """Plot a compact anchor/leaf overview of the hierarchy tree.

    For a large hierarchy the plot stops at ``max_nodes`` and reports the
    visible and total counts in the title.  It does not draw all graph edges.
    ``anchor`` selects one top-level block subtree.
    """
    import matplotlib.pyplot as plt

    config = deepcopy(config or {})
    record = _record(source)
    tree = record.get("hierarchy_tree")
    if tree is None:
        raise ValueError("This result has no hierarchy_tree.")
    roots = tree.get("roots", [])
    if anchor is not None:
        roots = [item for item in roots if int(item.get("anchor")) == int(anchor)]
        if not roots:
            raise ValueError(f"anchor {anchor!r} is not a top-level hierarchy block.")
    positions = {}
    edges = []
    _walk_tree(roots, int(tree.get("root_level", 0)), None, positions, edges, int(max_nodes))
    if not positions:
        raise ValueError("The hierarchy tree is empty.")
    levels = sorted({level for level, _ in positions})
    y_for = {}
    for level, key in positions:
        y_for.setdefault(level, []).append(key)
    coords = {}
    for level, keys in y_for.items():
        for index, key in enumerate(sorted(keys)):
            coords[(level, key)] = (level, index - (len(keys) - 1) / 2)
    with _theme(config):
        fig, ax = plt.subplots(figsize=config.get("style", {}).get("figsize", [12, 8]))
        for (level_a, key_a), (level_b, key_b) in edges:
            if key_a in coords and key_b in coords:
                xa, ya = coords[key_a]
                xb, yb = coords[key_b]
                ax.plot([xa, xb], [ya, yb], color="#999999", linewidth=0.7, zorder=1)
        for key, node in positions.items():
            x, y = coords[key]
            kind = node.get("kind")
            color = "#4c72b0" if kind == "block" else "#55a868"
            marker = "s" if kind == "block" else "o"
            ax.scatter([x], [y], color=color, marker=marker, s=28, zorder=2)
            if kind == "block":
                label = f"A{node.get('anchor')} ({node.get('size')})"
            else:
                label = str(node.get("node"))
            ax.text(x, y + 0.12, label, fontsize=7, ha="center", va="bottom")
        ax.set_xlabel("hierarchy level")
        ax.set_ylabel("anchor order (compact)")
        ax.set_title(
            f"Hierarchy tree: {len(positions)} visible nodes"
            + (" (truncated)" if len(positions) >= int(max_nodes) else "")
        )
        return _finish(fig, config, record, "qaoa_in_qaoa_tree")
