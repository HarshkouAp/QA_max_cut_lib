"""Pure metrics and plots for classical Max-Cut solvers and refinement."""

from __future__ import annotations

from copy import deepcopy
import math

from ..core.graph import WeightedGraph
from ..core.records import resolve_record
from .plotting import _finish, _theme

__all__ = [
    "analyze_solution",
    "plot_classical_comparison",
    "plot_gw_convergence",
    "plot_refinement",
    "plot_sa_reads",
]


def analyze_solution(graph, solution, *, reference_cut=None, reference_kind="provided"):
    """Return cut metrics for one complete spin assignment.

    Parameters
    ----------
    graph : WeightedGraph
        Original graph in cut units.
    solution : mapping of int to int
        Complete assignment with one integer spin for every node.
    reference_cut : float, optional
        A separately supplied exact or heuristic cut reference.  It is never
        silently raised to one or treated as a proof of optimality.
    reference_kind : str, optional
        Label for the reference.  A heuristic reference is not certified.

    Returns
    -------
    dict
        Complete assignment flags, ``cut_value``, ``solution_energy``,
        number and weight of separated edges, and optional gap/ratio.
        ``ratio`` is null when the reference is not positive.
    """
    if not isinstance(graph, WeightedGraph):
        raise TypeError("graph must be a WeightedGraph.")
    if set(solution) != set(graph.nodes):
        raise ValueError("solution must contain every graph node exactly once.")
    complete = True
    valid = True
    for value in solution.values():
        if isinstance(value, bool) or not isinstance(value, int) or value not in (-1, 1):
            valid = False
    if not valid:
        raise ValueError("Every spin must be an integer +1 or -1.")
    cut_edges = [
        (u, v, float(weight))
        for u, v, weight in graph.edges
        if solution[u] != solution[v]
    ]
    cut = float(graph.cut_value(solution))
    result = {
        "complete": complete,
        "valid": valid,
        "node_count": len(graph),
        "edge_count": len(graph.edges),
        "spin_values": sorted({int(value) for value in solution.values()}),
        "cut_value": cut,
        "solution_energy": -cut,
        "separated_edge_count": len(cut_edges),
        "cut_weight": float(math.fsum(item[2] for item in cut_edges)),
        "cut_edges": cut_edges,
        "reference_cut": reference_cut,
        "reference_kind": reference_kind if reference_cut is not None else None,
        "optimality_gap": None,
        "ratio": None,
    }
    if reference_cut is not None:
        reference = float(reference_cut)
        if not math.isfinite(reference):
            raise ValueError("reference_cut must be finite.")
        result["optimality_gap"] = float(reference - cut)
        result["ratio"] = float(cut / reference) if reference > 0 else None
    return result


def _record(source):
    return resolve_record(source)


def plot_gw_convergence(source, config=None):
    """Plot GW hyperplane candidates and the running best.

    The SDP objective is labelled as a numerical relaxation value, not a
    certified upper bound.
    """
    import matplotlib.pyplot as plt

    config = deepcopy(config or {})
    record = _record(source)
    history = record.get("candidate_history")
    if not history:
        raise ValueError("GW candidate_history is not collected.")
    trials = [int(item["trial_id"]) for item in history]
    cuts = [float(item["cut_value"]) for item in history]
    best = [float(item["running_best"]) for item in history]
    with _theme(config):
        fig, ax = plt.subplots(figsize=config.get("style", {}).get("figsize", [10, 6]))
        ax.plot(trials, cuts, marker="o", markersize=3, label="hyperplane C")
        ax.plot(trials, best, label="running best C")
        raw = record.get("sdp_objective_raw")
        if raw is not None:
            ax.axhline(raw, color="#c44e52", linestyle="--", label="numerical SDP relaxation")
        ax.set_xlabel("hyperplane id")
        ax.set_ylabel("C")
        ax.set_title("Goemans-Williamson rounding")
        ax.legend()
        return _finish(fig, config, record, "gw_convergence")


def plot_sa_reads(source, config=None):
    """Plot final C of logical SA reads and their running best."""
    import matplotlib.pyplot as plt

    config = deepcopy(config or {})
    record = _record(source)
    history = record.get("candidate_history")
    if not history:
        raise ValueError("SA candidate_history is not collected.")
    ids = [int(item["read_id"]) for item in history]
    cuts = [float(item["cut_value"]) for item in history]
    best = [float(item["running_best"]) for item in history]
    with _theme(config):
        fig, ax = plt.subplots(figsize=config.get("style", {}).get("figsize", [10, 6]))
        ax.plot(ids, cuts, marker="o", markersize=4, label="final read C")
        ax.plot(ids, best, marker="s", markersize=4, label="running best C")
        ax.set_xlabel("read id")
        ax.set_ylabel("C")
        ax.set_title("Classical simulated-annealing reads")
        ax.legend()
        return _finish(fig, config, record, "sa_reads")


def plot_refinement(source, config=None):
    """Plot accepted and rejected conditional blocks."""
    import matplotlib.pyplot as plt

    config = deepcopy(config or {})
    record = _record(source)
    refinement = record.get("refinement_run")
    if not refinement or not refinement.get("history"):
        raise ValueError("Refinement history is not collected.")
    history = refinement["history"]
    ids = [int(item.get("block_id", index)) for index, item in enumerate(history)]
    before = [float(item["c_before"]) for item in history]
    after = [float(item["c_after"]) for item in history]
    accepted = [bool(item.get("accepted")) for item in history]
    with _theme(config):
        fig, ax = plt.subplots(figsize=config.get("style", {}).get("figsize", [11, 6]))
        ax.plot(ids, before, marker="o", markersize=4, label="C before block")
        ax.plot(ids, after, marker="s", markersize=4, label="C after block")
        for block_id, accepted_flag in zip(ids, accepted):
            if accepted_flag:
                ax.axvspan(block_id - 0.35, block_id + 0.35, color="#55a868", alpha=0.15)
        ax.set_xlabel("block id")
        ax.set_ylabel("cut value")
        ax.set_title("Exact block refinement (green = accepted)")
        ax.legend()
        return _finish(fig, config, record, "refinement_blocks")


def plot_classical_comparison(source, config=None):
    """Compare base and refined cuts for a refined solver result."""
    import matplotlib.pyplot as plt

    config = deepcopy(config or {})
    record = _record(source)
    base = record.get("base_cut_value")
    final = record.get("cut_value")
    if base is None or final is None:
        raise ValueError("This result has no base_cut_value and final cut.")
    with _theme(config):
        fig, ax = plt.subplots(figsize=config.get("style", {}).get("figsize", [8, 5]))
        ax.bar(["base", "refined"], [float(base), float(final)], color=["#4c72b0", "#55a868"])
        ax.set_ylabel("C")
        ax.set_title("Base and refined complete solution")
        return _finish(fig, config, record, "classical_comparison")
