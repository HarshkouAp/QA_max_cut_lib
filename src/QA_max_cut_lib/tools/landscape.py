"""Calculate bounded exact p=1 parameter maps independently of plotting."""

from copy import deepcopy
from time import perf_counter
from uuid import uuid4
import numpy as np
from .._version import __version__
from ..core.records import RunRecord
from ..core.saving import utc_now, save_result
from ..solvers.qaoa_cpu import cut_diagonal, state_at_angles
from ..solvers.objectives import objective_value, validate_objective


def scan_landscape(graph, config):
    """Calculate an exact p=1 map on explicitly supplied gamma and beta grids.

    Parameters
    ----------
    graph : WeightedGraph
        A small weighted graph, with at most 16 nodes in this tool.
    config : dict
        gamma and beta are finite lists of at least two coordinates.
        objective selects one of the five supported criteria. save is
        optional and uses the common result-export protocol.

    Returns
    -------
    RunRecord
        gamma, beta, and values. The array order is [beta, gamma].

    Notes
    -----
    A chosen gamma window is a research input. It is not assumed to be
    a universal period for arbitrary real graph weights.
    """
    config = deepcopy(config)
    graph = graph.copy()
    if set(config) - {"gamma", "beta", "objective", "quantity", "save"}:
        raise ValueError("Unknown landscape configuration field.")
    if not 1 <= len(graph) <= 16:
        raise ValueError("The first scanner supports 1..16 nodes.")
    gamma = np.asarray(config["gamma"], dtype=float)
    beta = np.asarray(config["beta"], dtype=float)
    for axis in (gamma, beta):
        if (
            axis.ndim != 1
            or not 2 <= len(axis) <= 256
            or not np.isfinite(axis).all()
            or np.any(np.diff(axis) <= 0)
        ):
            raise ValueError(
                "Grid axes must be strictly increasing finite vectors, with 2..256 points."
            )
    objective = validate_objective(config.get("objective", {}))
    config["objective"] = objective
    config.setdefault("quantity", "optimizer_loss")
    if config["quantity"] not in ("objective_value", "optimizer_loss"):
        raise ValueError("quantity must be objective_value or optimizer_loss.")
    started = utc_now()
    clock = perf_counter()
    cuts = cut_diagonal(graph)
    values = np.empty((len(beta), len(gamma)))
    for j, b in enumerate(beta):
        for i, g in enumerate(gamma):
            _, probabilities = state_at_angles(cuts, [g, b], 1)
            value = objective_value(cuts, probabilities, objective)
            values[j, i] = value if config["quantity"] == "objective_value" else -value
    record = RunRecord(
        {
            "schema_version": 1,
            "algorithm_type": "landscape_scanner_cpu",
            "package_version": __version__,
            "run_id": str(uuid4()),
            "status": "completed",
            "termination_reason": "grid_complete",
            "started_at": started,
            "finished_at": utc_now(),
            "config": config,
            "graph": graph.snapshot(),
            "gamma": gamma,
            "beta": beta,
            "values": values,
            "timing": {"compute_seconds": perf_counter() - clock},
            "resources": {
                "shots_used": 0,
                "grid_points": int(values.size),
                "num_stages": int(values.size),
            },
        }
    )
    if "save" in config:
        save_result(record, config["save"])
    return record
