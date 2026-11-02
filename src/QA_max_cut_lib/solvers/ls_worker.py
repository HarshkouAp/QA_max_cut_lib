"""Serializable worker function for local LS-RQAOA tasks.

The coordinator sends only an induced graph, a source configuration, one
stable seed, and one local quota.  Importing this module does not create a
process pool or perform a measurement.
"""

from __future__ import annotations

from copy import deepcopy
from time import perf_counter, time

from ..core.budget import ShotBudget
from ..core.graph import WeightedGraph
from ..core.reduction import GraphReducer
from .ls_components import build_correlation_source

__all__ = ["solve_local_task"]


def solve_local_task(payload):
    """Solve one independent local correlation task.

    Parameters
    ----------
    payload : dict
        Serializable task data.  Required keys are ``task_id``, ``nodes``,
        ``edges``, ``parent_version``, ``source_config``, ``seed``,
        ``pair_domain``, ``shot_quota`` and the two collection flags.

    Returns
    -------
    dict
        Compact local table and resource counters.  No parent solver,
        graph, plot, or lock is referenced.

    Notes
    -----
    The local source is created inside the worker.  Its random generator
    and buffers never cross a process boundary or a task boundary.
    """
    started = perf_counter()
    started_unix = time()
    graph = WeightedGraph(edges=payload["edges"], nodes=payload["nodes"])
    state = GraphReducer().start(graph)
    source_config = deepcopy(payload["source_config"])
    initial_angles = None
    initialization = source_config.get("initialization")
    if isinstance(initialization, dict):
        initial_angles = deepcopy(initialization.get("parameters", {}).get("values"))
    steps = int(payload.get("training_steps", 0))
    if source_config.get("type") == "qaoa_statevector_cpu":
        source_config.setdefault("optimizer", {}).setdefault("parameters", {})[
            "max_steps"
        ] = steps
    source = build_correlation_source(source_config, seed=payload.get("seed"))
    quota = payload.get("shot_quota")
    budget = ShotBudget(None if quota is None else int(quota))
    try:
        from threadpoolctl import threadpool_limits

        native_context = threadpool_limits(limits=1)
    except ImportError:  # pragma: no cover - optional dependency
        from contextlib import nullcontext

        native_context = nullcontext()
    with native_context:
        result = source.estimate(
            state,
            budget=budget,
            pair_domain=payload["pair_domain"],
            collect_optimizer_history=bool(payload.get("collect_optimizer_history", False)),
            collect_objective_history=bool(payload.get("collect_objective_history", False)),
        )
        native_pools = None
        try:
            from threadpoolctl import threadpool_info

            native_pools = [
                {
                    "internal_api": item.get("internal_api"),
                    "num_threads": item.get("num_threads"),
                }
                for item in threadpool_info()
            ]
        except ImportError:  # pragma: no cover
            pass
    values = {}
    for pair, value in result.values.items():
        values[(int(pair[0]), int(pair[1]))] = float(value)
    return {
        "task_id": str(payload["task_id"]),
        "parent_version": int(payload["parent_version"]),
        "nodes": [int(node) for node in state.current_nodes],
        "pairs": [list(pair) for pair in result.pairs],
        "values": values,
        "mode": result.mode,
        "angles": [float(value) for value in result.angles],
        "layers": int(result.layers),
        "training": deepcopy(result.training),
        "resources": deepcopy(result.resources),
        "objective_reference": result.objective_reference,
        "seconds": perf_counter() - started,
        "started_unix": started_unix,
        "finished_unix": time(),
        "initial_angles": initial_angles,
        "native_pools": native_pools,
        "status": "completed",
    }
