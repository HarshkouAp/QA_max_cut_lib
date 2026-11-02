"""Components shared by the CPU and NVIDIA QAOA-in-QAOA coordinators.

The module contains three independent pieces:

* validation and sizing for a local readout solver;
* the exact one-level block projection used as an independent check;
* one serializable local-task function used by CPU worker processes.

The shared RQAOA/LS reduction core is not copied here.  The coordinator
applies the block constraints through :class:`GraphReducer`.
"""

from __future__ import annotations

from copy import deepcopy
import math
from numbers import Integral
from time import perf_counter, time
import os

import numpy as np

from ..core.graph import WeightedGraph

__all__ = [
    "local_solver_config",
    "local_task_requirements",
    "normalize_block_pattern",
    "project_blocks",
    "validate_local_solver",
    "solve_hierarchy_local_task",
]

_LOCAL_QAOA_FIELDS = {
    "type",
    "parameters",
    "objective",
    "estimation",
    "optimizer",
    "initialization",
    "readout",
}
_LOCAL_GPU_FIELDS = _LOCAL_QAOA_FIELDS | {"gpu"}
_EXACT_FIELDS = {"type", "parameters"}


def _int(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def validate_local_solver(config, *, backend):
    """Validate one local readout solver section.

    Parameters
    ----------
    config : dict
        Local solver section.  It contains no root shot limit, save block,
        output block, report/log switch, or independent seed.
    backend : {"cpu", "gpu"}
        Explicit hierarchy backend.  A GPU hierarchy accepts only the GPU
        QAOA local solver.

    Returns
    -------
    dict
        Independent normalized section.

    Raises
    ------
    ValueError
        The type, field set, or backend combination is unsupported.
    """
    if not isinstance(config, dict):
        raise TypeError("local_solver must be a mapping.")
    config = deepcopy(config)
    kind = config.get("type")
    if kind not in ("qaoa_cpu", "qaoa_gpu", "exact_cpu"):
        raise ValueError("Unsupported local_solver.type.")
    if kind == "qaoa_cpu" and backend != "cpu":
        raise ValueError("qaoa_cpu local solver requires the CPU hierarchy.")
    if kind == "qaoa_gpu" and backend != "gpu":
        raise ValueError("qaoa_gpu local solver requires the GPU hierarchy.")
    if kind == "exact_cpu" and backend != "cpu":
        raise ValueError("exact_cpu is an explicit CPU test component.")
    allowed = _EXACT_FIELDS if kind == "exact_cpu" else (
        _LOCAL_GPU_FIELDS if kind == "qaoa_gpu" else _LOCAL_QAOA_FIELDS
    )
    if set(config) - allowed:
        raise ValueError(
            f"Unknown local_solver fields: {sorted(set(config) - allowed)}"
        )
    if kind == "exact_cpu":
        params = config.setdefault("parameters", {})
        if set(params) - {"max_vertices", "batch_size", "tie_tolerance"}:
            raise ValueError("Unknown exact local solver parameter.")
        params.setdefault("max_vertices", 16)
        _int(params["max_vertices"], "local max_vertices", minimum=1)
        return config
    params = config.setdefault("parameters", {})
    if set(params) - {"layers", "max_vertices", "dtype", "representation"}:
        raise ValueError("Unknown local QAOA parameter.")
    params.setdefault("layers", 1)
    params.setdefault("max_vertices", 16)
    params.setdefault("dtype", "complex64" if kind == "qaoa_gpu" else "complex128")
    if kind == "qaoa_gpu":
        params.setdefault("representation", "spin_flip_pairs")
    elif "representation" in params:
        raise ValueError("QAOA CPU has no state representation option.")
    _int(params["layers"], "local layers", minimum=1)
    _int(params["max_vertices"], "local max_vertices", minimum=1)
    objective = config.setdefault("objective", {"type": "mean_cut", "parameters": {}})
    estimation = config.setdefault("estimation", {})
    if set(estimation) - {"mode", "shots_per_evaluation"}:
        raise ValueError("Unknown local estimation field.")
    estimation.setdefault("mode", "exact")
    estimation.setdefault("shots_per_evaluation", 256)
    if estimation["mode"] not in ("exact", "shots"):
        raise ValueError("Local estimation must be exact or shots.")
    _int(estimation["shots_per_evaluation"], "shots_per_evaluation", minimum=1)
    optimizer = config.setdefault("optimizer", {})
    if set(optimizer) - {"type", "parameters", "gradient"}:
        raise ValueError("Unknown local optimizer field.")
    optimizer.setdefault("type", "adam")
    if optimizer["type"] != "adam":
        raise ValueError("Only ADAM is supported for a local QAOA readout.")
    op = optimizer.setdefault("parameters", {})
    if set(op) - {"max_steps", "learning_rate"}:
        raise ValueError("Unknown local ADAM parameter.")
    op.setdefault("max_steps", 10)
    op.setdefault("learning_rate", 0.05)
    _int(op["max_steps"], "local max_steps", minimum=0)
    if op["max_steps"] > 0:
        _finite(op["learning_rate"], "local learning_rate")
        if op["learning_rate"] <= 0:
            raise ValueError("local learning_rate must be positive.")
    grad = optimizer.setdefault(
        "gradient", {"type": "central_difference", "parameters": {"step": 0.01}}
    )
    if grad.get("type") not in ("central_difference", "adjoint"):
        raise ValueError("Unsupported local gradient.")
    if grad["type"] == "adjoint" and (
        objective.get("type") != "mean_cut" or estimation["mode"] != "exact"
    ):
        raise ValueError("Local adjoint requires exact Mean Cut.")
    if grad["type"] == "central_difference":
        gp = grad.setdefault("parameters", {})
        if set(gp) != {"step"}:
            raise ValueError("Central differences need only a step.")
        _finite(gp["step"], "local gradient step")
        if gp["step"] <= 0:
            raise ValueError("local gradient step must be positive.")
    init = config.setdefault("initialization", {"type": "fixed_angles", "parameters": {}})
    if init.get("type") not in ("fixed_angles", "random_angles"):
        raise ValueError("Unsupported local initialization.")
    if init["type"] == "fixed_angles":
        ip = init.setdefault("parameters", {})
        ip.setdefault(
            "values",
            [0.5] * params["layers"] + [0.3] * params["layers"],
        )
        if np.shape(ip["values"]) != (2 * params["layers"],):
            raise ValueError("Local fixed angles need exactly 2*layers values.")
    readout = config.setdefault("readout", {"mode": "most_probable"})
    if set(readout) - {"mode", "shots"}:
        raise ValueError("Unknown local readout field.")
    readout.setdefault("mode", "most_probable")
    if readout["mode"] == "best_of_shots":
        _int(readout.get("shots"), "local readout shots", minimum=1)
    elif readout["mode"] == "most_probable":
        if "shots" in readout:
            raise ValueError("most_probable has no readout shots.")
    else:
        raise ValueError("Unsupported local readout mode.")
    if kind == "qaoa_gpu" and "gpu" not in config:
        config["gpu"] = {}
    return config


def local_task_requirements(local_config):
    """Return mandatory readout shots, iteration shots, and step limit."""
    if local_config["type"] == "exact_cpu":
        return 0, 0, 0
    readout = local_config.get("readout", {"mode": "most_probable"})
    mandatory = int(readout.get("shots", 0)) if readout["mode"] == "best_of_shots" else 0
    estimation = local_config.get("estimation", {})
    layers = int(local_config["parameters"]["layers"])
    if estimation.get("mode") == "shots":
        cost = (4 * layers + 1) * int(estimation.get("shots_per_evaluation", 256))
    else:
        cost = 0
    steps = int(local_config.get("optimizer", {}).get("parameters", {}).get("max_steps", 10))
    return mandatory, cost, steps


def normalize_block_pattern(nodes, raw_solution):
    """Return raw and anchor-normalized patterns for one block.

    The anchor is the smallest external integer label in the block.  The
    normalized pattern has ``+1`` at the anchor.
    """
    nodes = tuple(sorted(int(node) for node in nodes))
    if not nodes:
        raise ValueError("A block must not be empty.")
    if not isinstance(raw_solution, dict):
        raise TypeError("A local solution must be a mapping.")
    if set(raw_solution) != set(nodes):
        raise ValueError("A local solution must contain every block node once.")
    raw = {}
    for node in nodes:
        value = raw_solution[node]
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError("Local spins must be integer +1 or -1.")
        spin = int(value)
        if spin not in (-1, 1):
            raise ValueError("Local spins must be integer +1 or -1.")
        raw[node] = spin
    anchor = nodes[0]
    sign = raw[anchor]
    normalized = {node: raw[node] * sign for node in nodes}
    return {
        "anchor": anchor,
        "nodes": list(nodes),
        "raw_pattern": raw,
        "normalized_pattern": normalized,
        "anchor_sign": sign,
    }


def _edges_dict(graph):
    return {(int(u), int(v)): float(weight) for u, v, weight in graph.edges}


def _pattern_for(nodes, normalized):
    return {int(node): int(normalized[node]) for node in nodes}


def project_blocks(graph, blocks, normalized_patterns):
    """Project one validated partition into its exact Max-Cut metagraph.

    Parameters
    ----------
    graph : WeightedGraph
        Current level graph.  It is not changed.
    blocks : list of sequences of int
        Non-overlapping blocks that cover every graph node.
    normalized_patterns : mapping of anchor to mapping of int to ±1
        Anchor-normalized spin pattern for each block.

    Returns
    -------
    tuple
        ``(meta_graph, delta_b, block_records)``.  ``meta_graph`` has one
        anchor per block and weights ``W_ab``.  ``delta_b`` includes all
        internal and boundary edges of the current graph.
    """
    if not isinstance(graph, WeightedGraph):
        raise TypeError("graph must be a WeightedGraph.")
    node_set = set(int(node) for node in graph.nodes)
    used = []
    anchors = []
    blocks = [tuple(sorted(int(node) for node in block)) for block in blocks]
    for block in blocks:
        if not block:
            raise ValueError("A block must not be empty.")
        if len(set(block)) != len(block):
            raise ValueError("A block contains a duplicate node.")
        if not set(block) <= node_set:
            raise ValueError("A block contains a node outside the graph.")
        used.extend(block)
        anchors.append(block[0])
    if len(set(used)) != len(used) or set(used) != node_set:
        raise ValueError("Blocks must be non-overlapping and cover every node.")
    if len(set(anchors)) != len(anchors):
        raise ValueError("Block anchors must be unique.")
    patterns = {}
    for block, anchor in zip(blocks, anchors):
        if anchor not in normalized_patterns:
            raise ValueError(f"Missing normalized pattern for anchor {anchor}.")
        pattern = _pattern_for(block, normalized_patterns[anchor])
        if set(pattern) != set(block):
            raise ValueError("A normalized pattern has the wrong node set.")
        if any(value not in (-1, 1) for value in pattern.values()):
            raise ValueError("Normalized patterns must contain integer ±1.")
        if pattern[anchor] != 1:
            raise ValueError("A normalized pattern must have +1 at its anchor.")
        patterns[anchor] = pattern
    owner = {}
    for block, anchor in zip(blocks, anchors):
        for node in block:
            owner[node] = anchor
    edges = _edges_dict(graph)
    delta_b = math.fsum(
        weight * 0.5 * (1 - patterns[owner[u]][u] * patterns[owner[v]][v])
        for (u, v), weight in edges.items()
    )
    meta_weights = {}
    for (u, v), weight in edges.items():
        a_u = owner[u]
        a_v = owner[v]
        if a_u == a_v:
            continue
        pair = (min(a_u, a_v), max(a_u, a_v))
        meta_weights[pair] = meta_weights.get(pair, 0.0) + weight * (
            patterns[a_u][u] * patterns[a_v][v]
        )
    meta_weights = {
        pair: value
        for pair, value in meta_weights.items()
        if value != 0.0
    }
    meta = WeightedGraph(
        [(u, v, value) for (u, v), value in sorted(meta_weights.items())],
        anchors,
    )
    records = [
        {
            "anchor": anchor,
            "nodes": list(block),
            "normalized_pattern": dict(patterns[anchor]),
            "raw_pattern": dict(patterns[anchor]),
        }
        for block, anchor in zip(blocks, anchors)
    ]
    return meta, float(delta_b), records


def _stable_seed(root_seed, level, block_id):
    """Derive a reproducible integer seed from a root seed and task ID."""
    base = 0 if root_seed is None else int(root_seed)
    sequence = np.random.SeedSequence([base & 0xFFFFFFFF, int(level), int(block_id)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def local_solver_config(local_config, *, seed, quota, collect_optimizer_history=False):
    """Build one child solver configuration without a public root lifecycle."""
    config = deepcopy(local_config)
    config["seed"] = int(seed)
    if config["type"] == "exact_cpu":
        config.setdefault("parameters", {})
        config["limits"] = {"shots_total": 0}
        return config
    config["limits"] = {"shots_total": None if quota is None else int(quota)}
    config["collect"] = {
        "optimizer_history": bool(collect_optimizer_history),
        "objective_history": False,
        "probabilities": False,
        "statevector": False,
        "correlations": False,
    }
    return config


def solve_hierarchy_local_task(payload):
    """Run one readout task in a child context.

    Parameters
    ----------
    payload : dict
        Serializable ``graph``, ``local_config``, ``seed``, ``quota``, and
        ``task_id`` data.

    Returns
    -------
    dict
        The raw local spin solution and compact resource fields.
    """
    started = perf_counter()
    started_unix = time()
    graph = WeightedGraph(payload["edges"], payload["nodes"])
    kind = payload["local_config"]["type"]
    child = local_solver_config(
        payload["local_config"],
        seed=payload["seed"],
        quota=payload.get("quota"),
        collect_optimizer_history=bool(payload.get("collect_optimizer_history", False)),
    )
    try:
        from threadpoolctl import threadpool_limits

        native_context = threadpool_limits(limits=1)
    except ImportError:  # pragma: no cover - optional dependency
        from contextlib import nullcontext

        native_context = nullcontext()
    with native_context:
        if kind == "exact_cpu":
            from .exact_cpu import ExactCPU

            solver = ExactCPU(child)
        elif kind == "qaoa_cpu":
            from .qaoa_cpu import QAOACPU

            solver = QAOACPU(child)
        elif kind == "qaoa_gpu":
            from .qaoa_gpu import QAOAGPU

            solver = QAOAGPU(child)
        else:
            raise ValueError("Unknown local solver type.")
        solution = solver.solve(graph, report=0, logs=0)
        record = solver.last_run
        resources = deepcopy(record.get("resources", {}) or {})
        angles = record.get("optimal_angles")
        local_history = record.get("optimizer_history")
        result = record.result
    return {
        "task_id": str(payload["task_id"]),
        "pid": os.getpid(),
        "level": int(payload["level"]),
        "block": int(payload["block"]),
        "nodes": [int(node) for node in graph.nodes],
        "solution": {int(node): int(spin) for node, spin in solution.items()},
        "angles": None if angles is None else [float(value) for value in angles],
        "layers": int(payload["local_config"].get("parameters", {}).get("layers", 0)),
        "readout_mode": payload["local_config"].get("readout", {}).get("mode"),
        "local_solver_type": kind,
        "resources": resources,
        "optimizer_history": (
            deepcopy(local_history)
            if payload.get("collect_optimizer_history")
            else None
        ),
        "status": result.get("status"),
        "termination_reason": result.get("termination_reason"),
        "seconds": perf_counter() - started,
        "started_unix": started_unix,
        "finished_unix": time(),
    }
