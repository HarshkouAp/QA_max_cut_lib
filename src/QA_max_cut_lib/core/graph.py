"""Store a simple weighted graph with stable integer node labels."""

from copy import deepcopy
import hashlib
import json
import math
from numbers import Integral, Real


def _node(value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("Node labels must be integers, without boolean labels.")
    return int(value)


class WeightedGraph:
    """Store a mutable, undirected weighted graph.

    Parameters
    ----------
    edges : iterable of (int, int, float), optional
        Edges with explicit finite weights. Loops and duplicate edges are
        not allowed. Negative and zero weights are allowed.
    nodes : iterable of int, optional
        Node labels, including isolated nodes. Gaps and negative labels
        are allowed. The insertion order defines the state-vector order.

    Attributes
    ----------
    nodes : tuple of int
        Original node labels in their fixed insertion order.
    edges : tuple
        Triples containing the two labels and the explicit edge weight.
    version : int
        Number of changes to this graph instance.

    Notes
    -----
    The cut is C(s) = sum(w_uv * (1 - s_u*s_v) / 2).
    The signed energy is E(s) = -C(s). No absolute value is used.
    """

    def __init__(self, edges=(), nodes=()):
        self._nodes = {}
        self._edges = {}
        self._version = 0
        self._recipe = None
        for value in nodes:
            label = _node(value)
            if label in self._nodes:
                raise ValueError(f"Duplicate node label: {label}")
            self._nodes[label] = None
        for u, v, weight in edges:
            key = self._key(u, v)
            if key in self._edges:
                raise ValueError(f"Duplicate undirected edge: {key}")
            self.set_weight(u, v, weight)
        self._version = 0

    @staticmethod
    def _key(u, v):
        u, v = _node(u), _node(v)
        if u == v:
            raise ValueError("Self-loops are not supported.")
        return min(u, v), max(u, v)

    @property
    def nodes(self):
        return tuple(self._nodes)

    @property
    def edges(self):
        return tuple((u, v, w) for (u, v), w in self._edges.items())

    @property
    def version(self):
        return self._version

    def __len__(self):
        return len(self._nodes)

    def add_node(self, node):
        """Add one node. Keep all existing labels and edges."""
        node = _node(node)
        if node not in self._nodes:
            self._nodes[node] = None
            self._version += 1

    def set_weight(self, u, v, weight):
        """Set an edge weight and add missing endpoint nodes."""
        key = self._key(u, v)
        if isinstance(weight, bool) or not isinstance(weight, Real):
            raise TypeError("An edge weight must be a real number.")
        weight = float(weight)
        if not math.isfinite(weight):
            raise ValueError("Edge weights must be finite.")
        self._nodes.setdefault(_node(u), None)
        self._nodes.setdefault(_node(v), None)
        self._edges[key] = weight
        self._version += 1

    def remove_edge(self, u, v):
        """Remove an edge without removing its endpoint nodes."""
        del self._edges[self._key(u, v)]
        self._version += 1

    def weight(self, u, v):
        """Return the edge weight, or zero for an absent pair."""
        return self._edges.get(self._key(u, v), 0.0)

    def cut_value(self, solution):
        """Calculate the cut on this graph by direct edge summation.

        Parameters
        ----------
        solution : dict of int to int
            One spin, +1 or -1, for every original node.

        Returns
        -------
        float
            Signed weighted cut value. Larger values are better.
        """
        if set(solution) != set(self._nodes):
            raise ValueError("The solution must contain every node exactly once.")
        if any(
            isinstance(v, bool) or not isinstance(v, Integral) or v not in (-1, 1)
            for v in solution.values()
        ):
            raise ValueError("Every spin must be an integer +1 or -1.")
        result = math.fsum(
            w for (u, v), w in self._edges.items() if solution[u] != solution[v]
        )
        if not math.isfinite(result):
            raise ValueError("The sum of cut weights is not finite.")
        return result

    def subgraph(self, nodes):
        """Return an induced graph with the same external labels."""
        chosen = tuple(_node(v) for v in nodes)
        if len(set(chosen)) != len(chosen) or not set(chosen) <= set(self._nodes):
            raise ValueError("Subgraph nodes must be unique existing labels.")
        allowed = set(chosen)
        return WeightedGraph(
            (e for e in self.edges if e[0] in allowed and e[1] in allowed), chosen
        )

    def snapshot(self):
        """Return JSON-compatible graph data and its content fingerprint."""
        data = {"nodes": list(self.nodes), "edges": [list(e) for e in self.edges]}
        canonical = {"nodes": sorted(self.nodes), "edges": sorted(data["edges"])}
        encoded = json.dumps(canonical, sort_keys=True, allow_nan=False).encode()
        data.update(
            version=self.version,
            fingerprint=hashlib.sha256(encoded).hexdigest(),
            recipe=deepcopy(self._recipe),
        )
        return data

    @classmethod
    def from_snapshot(cls, snapshot):
        """Restore actual graph data, including isolated node labels."""
        graph = cls(snapshot["edges"], snapshot["nodes"])
        graph._recipe = deepcopy(snapshot.get("recipe"))
        if (
            "fingerprint" in snapshot
            and graph.snapshot()["fingerprint"] != snapshot["fingerprint"]
        ):
            raise ValueError("Graph fingerprint mismatch.")
        graph._version = int(snapshot.get("version", 0))
        return graph

    def copy(self):
        """Return an independent graph with the same content."""
        return self.from_snapshot(self.snapshot())
