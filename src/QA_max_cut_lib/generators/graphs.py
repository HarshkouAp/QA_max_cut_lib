"""Generate bounded, reproducible weighted graph tasks."""

from copy import deepcopy
from itertools import combinations
import numpy as np
from ..core.graph import WeightedGraph


class GraphFactory:
    """Create complete, G(n,m), or G(n,p) graphs from a JSON recipe."""

    @staticmethod
    def from_recipe(recipe):
        """Generate a graph with separate topology and weight seeds.

        Parameters
        ----------
        recipe : dict
            Keys type, parameters, seed, weights, and schema_version.
            Type is complete, gnm, or erdos_renyi. Weight type is
            constant, signed_discrete, or uniform.

        Returns
        -------
        WeightedGraph
            A graph with explicit weights, including for the unit case.
        """
        recipe = deepcopy(recipe)
        if set(recipe) - {"type", "parameters", "seed", "weights", "schema_version"}:
            raise ValueError("Unknown graph recipe field.")
        if recipe.get("schema_version", 1) != 1:
            raise ValueError("Unsupported recipe schema.")
        recipe.setdefault("schema_version", 1)
        recipe.setdefault("seed", None)
        kind = recipe.get("type")
        extra = {"complete": set(), "gnm": {"n_edges"}, "erdos_renyi": {"probability"}}
        if kind not in extra:
            raise ValueError(f"Unknown graph generator: {kind}")
        parameters = recipe.setdefault("parameters", {})
        if set(parameters) - ({"n_vertices", "nodes"} | extra[kind]):
            raise ValueError("Unknown graph generator parameter.")
        n = parameters["n_vertices"]
        if isinstance(n, bool) or not isinstance(n, int) or not 0 <= n <= 1000:
            raise ValueError("n_vertices must be an integer in [0, 1000].")
        labels = parameters.get("nodes", list(range(n)))
        parameters["nodes"] = list(labels)
        if len(labels) != n:
            raise ValueError("nodes must have n_vertices entries.")
        graph = WeightedGraph(nodes=labels)
        pairs = list(combinations(graph.nodes, 2))
        rng = np.random.default_rng(recipe.get("seed"))
        if kind == "complete":
            chosen = pairs
        elif kind == "gnm":
            m = parameters["n_edges"]
            if (
                isinstance(m, bool)
                or not isinstance(m, int)
                or not 0 <= m <= len(pairs)
            ):
                raise ValueError("Invalid n_edges.")
            chosen = [
                pairs[int(i)] for i in sorted(rng.choice(len(pairs), m, replace=False))
            ]
        elif kind == "erdos_renyi":
            p = float(parameters["probability"])
            if not 0 <= p <= 1:
                raise ValueError("probability must be in [0, 1].")
            chosen = [edge for edge in pairs if rng.random() < p]
        else:
            raise ValueError(f"Unknown graph generator: {kind}")
        weights = recipe.setdefault(
            "weights", {"type": "constant", "parameters": {"value": 1.0}}
        )
        if set(weights) - {"type", "parameters", "seed"}:
            raise ValueError("Unknown weight recipe field.")
        weights.setdefault("seed", recipe.get("seed"))
        wrng = np.random.default_rng(weights.get("seed", recipe.get("seed")))
        wp = weights.setdefault("parameters", {})
        weight_keys = {
            "constant": {"value"},
            "signed_discrete": {"values"},
            "uniform": {"low", "high"},
        }
        if (
            weights.get("type") not in weight_keys
            or set(wp) - weight_keys[weights["type"]]
        ):
            raise ValueError("Unknown weight generator or parameter.")
        if weights["type"] == "constant":
            wp.setdefault("value", 1.0)
            if (
                isinstance(wp["value"], bool)
                or not np.isscalar(wp["value"])
                or not np.isfinite(wp["value"])
            ):
                raise ValueError("The constant weight must be finite and real.")
        elif weights["type"] == "signed_discrete":
            values = np.asarray(wp["values"], dtype=float)
            if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
                raise ValueError("Weight values must be a nonempty finite vector.")
        elif (
            not np.isfinite([wp["low"], wp["high"]]).all() or not wp["low"] < wp["high"]
        ):
            raise ValueError("Uniform weight bounds must be finite, with low < high.")
        for u, v in chosen:
            if weights["type"] == "constant":
                w = wp.get("value", 1.0)
            elif weights["type"] == "signed_discrete":
                w = float(wrng.choice(wp["values"]))
            elif weights["type"] == "uniform":
                w = float(wrng.uniform(wp["low"], wp["high"]))
            else:
                raise ValueError(f"Unknown weight generator: {weights['type']}")
            graph.set_weight(u, v, w)
        graph._recipe = deepcopy(recipe)
        return graph


def partition(graph, max_vertices, *, seed=0):
    """Return disjoint induced subgraphs of at most max_vertices nodes.

    This is a random partition, not a graph-density optimization method.
    All original labels are retained, including labels of isolated nodes.
    """
    if (
        isinstance(max_vertices, bool)
        or not isinstance(max_vertices, int)
        or max_vertices < 1
    ):
        raise ValueError("max_vertices must be a positive integer.")
    order = np.random.default_rng(seed).permutation(len(graph.nodes))
    return [
        graph.subgraph(
            [graph.nodes[int(i)] for i in order[start : start + max_vertices]]
        )
        for start in range(0, len(order), max_vertices)
    ]
