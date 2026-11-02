"""Lightweight MI plots for saved observer records.

Matplotlib is imported only when a plot is requested.  No plot function
runs a solver, sampler, or quantum circuit.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "plot_mutual_information_map",
    "plot_mutual_information_frames",
    "plot_mutual_information_distance",
]


def _frames(record):
    result = record.result if hasattr(record, "result") else record
    payload = result.get("mutual_information") if isinstance(result, dict) else None
    if payload is None:
        raise ValueError("record has no mutual_information payload.")
    frames = payload.get("frames", [])
    if not frames:
        raise ValueError("record has no MI frames.")
    return frames


def _plt():
    import matplotlib.pyplot as plt

    return plt


def plot_mutual_information_map(record, *, frame=0, measure="quantum", ax=None):
    """Plot one QMI or classical-Z-MI dense map."""
    frames = _frames(record)
    item = frames[int(frame)]
    label = "quantum_mi" if measure in ("quantum", "qmi") else "classical_z_mi"
    matrix = np.asarray(item["matrix"][label], dtype=float)
    mask = np.asarray(item["matrix"]["valid_mask"], dtype=bool)
    shown = np.where(mask, matrix, np.nan)
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(shown, cmap="viridis", vmin=0.0, vmax=2.0 if label == "quantum_mi" else 1.0)
    labels = item["matrix"]["labels"]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([str(label) for label in labels])
    ax.set_yticklabels([str(label) for label in labels])
    ax.set_title(f"{'QMI' if label == 'quantum_mi' else 'Z-MI'}, layer {item.get('layer')}")
    plt.colorbar(image, ax=ax, label="bits")
    return ax


def plot_mutual_information_frames(record, *, measure="quantum", ax=None):
    """Plot mean/min/max over snapshot layers; no monotonicity is implied."""
    frames = _frames(record)
    key = "quantum_summary" if measure in ("quantum", "qmi") else "classical_z_summary"
    x = [frame.get("layer", index) for index, frame in enumerate(frames)]
    mean = [frame[key]["mean"] if frame[key]["mean"] is not None else np.nan for frame in frames]
    low = [frame[key]["min"] if frame[key]["min"] is not None else np.nan for frame in frames]
    high = [frame[key]["max"] if frame[key]["max"] is not None else np.nan for frame in frames]
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(x, mean, marker="o", label="mean")
    ax.fill_between(x, low, high, alpha=0.2, label="min..max")
    ax.set_xlabel("prefix layer")
    ax.set_ylabel("bits")
    ax.set_title("MI over final-angle prefixes")
    ax.legend()
    return ax


def plot_mutual_information_distance(record, *, measure="quantum", ax=None):
    """Plot mean MI grouped by integer hop distance."""
    frames = _frames(record)
    key = "quantum_summary" if measure in ("quantum", "qmi") else "classical_z_summary"
    profiles = [frame.get("distance_profile", {}) for frame in frames]
    keys = sorted({key for profile in profiles for key in profile if key != "disconnected"})
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 3.8))
    for index, profile in enumerate(profiles):
        xs = [int(key) for key in keys if key in profile]
        ys = [profile[key]["mean"] for key in keys if key in profile]
        ax.plot(xs, ys, marker="o", label=f"frame {index}")
    ax.set_xlabel("hop distance")
    ax.set_ylabel("mean MI, bits")
    ax.set_title("MI versus unweighted hop distance")
    ax.legend()
    return ax
