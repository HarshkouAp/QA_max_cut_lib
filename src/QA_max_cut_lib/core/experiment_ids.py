"""Stable identifiers for experiment plans, graphs, and measurements.

The functions in this module use a small, versioned canonical form.  They do
not use Python ``hash()``, object ids, process ids, or completion order.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np

from .graph import WeightedGraph

__all__ = [
    "canonical_json",
    "stable_hash",
    "graph_content_id",
    "graph_instance_id",
    "node_order",
    "measurement_protocol_id",
    "measurement_environment_id",
]


def _canonical(value):
    """Return a deterministic JSON value with explicit float spelling."""
    if isinstance(value, np.ndarray):
        return _canonical(value.tolist())
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Non-finite values are not allowed in identifiers.")
        if number == 0.0:
            number = 0.0
        # repr is round-trip exact for binary64 values.
        return {"$float": repr(number)}
    raise TypeError(f"Unsupported identifier value: {type(value).__name__}.")


def canonical_json(value):
    """Serialize a value with sorted keys and exact float tokens.

    Parameters
    ----------
    value : object
        JSON-like data.  NumPy scalars and arrays are converted explicitly.

    Returns
    -------
    str
        Compact UTF-8 JSON string.  Integer ``1`` and float ``1.0`` remain
        different values.
    """
    return json.dumps(
        _canonical(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_hash(value, *, schema="qa-max-cut-id-v1"):
    """Return a full SHA-256 digest for a canonical value."""
    material = (str(schema) + "|" + canonical_json(value)).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def node_order(graph):
    """Return the state-vector node order without changing the graph."""
    if not isinstance(graph, WeightedGraph):
        raise TypeError("graph must be a WeightedGraph.")
    return [int(node) for node in graph.nodes]


def _weight_token(weight):
    number = float(weight)
    if not math.isfinite(number):
        raise ValueError("Graph weights must be finite.")
    if number == 0.0:
        number = 0.0
    return repr(number)


def graph_content_id(graph):
    """Hash graph content independently of node insertion order.

    The digest includes every original node label (including isolated
    nodes), each undirected edge once, exact weight spelling, and a schema
    version.  Recipe provenance and the current graph version are excluded.
    """
    if not isinstance(graph, WeightedGraph):
        raise TypeError("graph must be a WeightedGraph.")
    nodes = sorted(int(node) for node in graph.nodes)
    edges = [
        (min(int(u), int(v)), max(int(u), int(v)), _weight_token(weight))
        for u, v, weight in graph.edges
    ]
    edges.sort(key=lambda item: (item[0], item[1], item[2]))
    return stable_hash(
        {
            "schema": "graph-content-v1",
            "nodes": nodes,
            "edges": [[u, v, w] for u, v, w in edges],
        }
    )


def graph_instance_id(graph):
    """Hash content plus the state-vector node order.

    Reordering a graph changes this identifier even when its mathematical
    content is unchanged.  A statevector or timing record is therefore never
    reused silently across orders.
    """
    return stable_hash(
        {
            "schema": "graph-instance-v1",
            "content": graph_content_id(graph),
            "node_order": node_order(graph),
        }
    )


def measurement_protocol_id(protocol):
    """Hash a resolved measurement protocol."""
    return stable_hash(
        {"schema": "measurement-protocol-v1", "protocol": dict(protocol or {})}
    )


def measurement_environment_id(environment):
    """Hash the machine and library fingerprint used for a measurement."""
    return stable_hash(
        {"schema": "measurement-environment-v1", "environment": dict(environment or {})}
    )
