"""Exact pair reductions of weighted Max-Cut and Ising problems.

The module separates three roles:

* ``IsingProblem`` stores the current polynomial
  ``E(s) = c + sum_i h_i s_i + sum_{i<j} J_ij s_i s_j``.
* ``ReductionState`` stores one working problem, the applied spin
  constraints, the accumulated Max-Cut offset, and a compact trace.
* ``GraphReducer`` starts a state and applies or reverses reductions.

A pair reduction removes one node ``u`` and keeps one node ``v``.  It
introduces the exact constraint ``s_u = sign * s_v``.  The transformation
keeps the energy equal for every assignment of the remaining spins.  It
does not claim that an arbitrary fixed pair preserves the global optimum.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Mapping

from .graph import WeightedGraph
from .records import to_json_data

__all__ = [
    "GraphReducer",
    "IsingProblem",
    "ReductionState",
    "SpinConstraint",
]

_ENERGY_CONVENTION = (
    "E(s) = c + sum_i h_i s_i + sum_{i<j} J_ij s_i s_j; "
    "for Max-Cut E = -C and C = sum_e w_e (1 - s_u s_v) / 2"
)


def _as_node(value, name="node"):
    """Return one integer node label.  Reject booleans and floats."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    return int(value)


def _as_finite(value, name):
    """Return one finite float.  Reject booleans and non-finite values."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _as_sign(value):
    """Return +1 or -1 as an integer.  A zero sign is not a spin."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("sign must be the integer +1 or -1.")
    result = int(value)
    if result not in (-1, 1):
        raise ValueError("sign must be the integer +1 or -1.")
    return result


@dataclass(frozen=True)
class SpinConstraint:
    """Store one exact spin constraint ``s_removed = sign * s_retained``.

    Parameters
    ----------
    removed, retained : int
        Original integer node labels.  They must differ.
    sign : int
        +1 or -1.

    Notes
    -----
    Constraints are stored in application order.  Restoration uses the
    reverse order, because a retained node can be removed by a later step.
    """

    removed: int
    retained: int
    sign: int

    def __post_init__(self):
        removed = _as_node(self.removed, "removed")
        retained = _as_node(self.retained, "retained")
        sign = _as_sign(self.sign)
        if removed == retained:
            raise ValueError("A node cannot be reduced with itself.")
        object.__setattr__(self, "removed", removed)
        object.__setattr__(self, "retained", retained)
        object.__setattr__(self, "sign", sign)

    def to_dict(self):
        """Return a JSON-compatible constraint record."""
        return {
            "removed_node": self.removed,
            "retained_node": self.retained,
            "sign": self.sign,
        }

    @classmethod
    def from_dict(cls, data):
        """Build a constraint from a snapshot record."""
        if not isinstance(data, Mapping):
            raise TypeError("A constraint record must be a mapping.")
        return cls(
            removed=data["removed_node"],
            retained=data["retained_node"],
            sign=data["sign"],
        )


class IsingProblem:
    """Store a finite Ising polynomial on integer node labels.

    Parameters
    ----------
    constant : float, default 0.0
        The constant term ``c``.
    fields : mapping of int to float, optional
        Local fields ``h_i``.  Missing fields are zero.
    couplings : mapping of (int, int) to float, optional
        Pair couplings ``J_ij``.  Each unordered pair occurs once.
        Missing pairs are zero.
    nodes : iterable of int, optional
        All node labels, including isolated nodes.  The order is kept.

    Attributes
    ----------
    constant : float
        The full constant term ``c``.  It is not a Max-Cut offset.
    nodes : tuple of int
        Node labels in insertion order.
    fields : dict of int to float
        Nonzero local fields.
    couplings : dict of (int, int) to float
        Nonzero pair couplings, keyed by the ordered pair.
    edge_count : int
        Number of nonzero couplings.
    sum_abs_couplings : float
        Sum of absolute nonzero couplings.  The value is maintained by
        local updates; a snapshot recomputes it.

    Notes
    -----
    The energy convention is
    ``E = c + sum_i h_i s_i + sum_{i<j} J_ij s_i s_j``.
    Zero fields and zero couplings are not stored.  The ``energy`` method
    costs O(N + E); one local coefficient update costs O(1), apart from
    normal dictionary growth.  A graph conversion uses ``J = w / 2`` and
    ``c = -sum(w) / 2``.
    """

    def __init__(self, constant=0.0, fields=None, couplings=None, nodes=()):
        self._constant = _as_finite(constant, "constant")
        self._nodes = {}
        self._fields = {}
        self._adj = {}
        self._sum_abs = 0.0
        self._edge_count = 0
        for node in nodes:
            self._add_node(node)
        if fields:
            if not isinstance(fields, Mapping):
                raise TypeError("fields must be a mapping.")
            for node, value in fields.items():
                node = _as_node(node, "field node")
                self._add_node(node)
                self.set_field(node, value)
        if couplings:
            if not isinstance(couplings, Mapping):
                raise TypeError("couplings must be a mapping.")
            for pair, value in couplings.items():
                if not isinstance(pair, tuple) or len(pair) != 2:
                    raise TypeError("A coupling key must be a pair of labels.")
                self.set_coupling(pair[0], pair[1], value)
        if self._edge_count:
            # Recompute once after construction to remove accumulation error.
            self._sum_abs = math.fsum(abs(value) for value in self.couplings.values())

    @classmethod
    def from_maxcut(cls, graph):
        """Build the Ising form of a weighted Max-Cut graph.

        Parameters
        ----------
        graph : WeightedGraph
            A graph with finite weights and original integer labels.

        Returns
        -------
        IsingProblem
            ``J_ij = w_ij / 2``, ``h_i = 0`` and
            ``c = -sum_e w_e / 2``.

        Notes
        -----
        Zero-weight edges have zero coupling.  They are not stored in the
        Ising adjacency.  The public graph still keeps them explicitly.
        """
        if not isinstance(graph, WeightedGraph):
            raise TypeError("graph must be a WeightedGraph.")
        weights = [weight for _, _, weight in graph.edges]
        constant = -0.5 * math.fsum(weights)
        couplings = {}
        for u, v, weight in graph.edges:
            if weight != 0.0:
                couplings[(u, v)] = 0.5 * weight
        return cls(constant=constant, couplings=couplings, nodes=graph.nodes)

    @property
    def constant(self):
        """Return the full constant term ``c``."""
        return self._constant

    @property
    def nodes(self):
        """Return node labels in insertion order."""
        return tuple(self._nodes)

    @property
    def fields(self):
        """Return a copy of the nonzero local fields."""
        return dict(self._fields)

    @property
    def couplings(self):
        """Return a copy of the nonzero pair couplings."""
        return {
            (u, v): self._adj[u][v]
            for u in self._nodes
            for v in self._adj[u]
            if u < v
        }

    @property
    def edge_count(self):
        """Return the number of nonzero couplings."""
        return self._edge_count

    @property
    def sum_abs_couplings(self):
        """Return the maintained sum of absolute nonzero couplings."""
        return self._sum_abs

    def __len__(self):
        return len(self._nodes)

    def __repr__(self):
        return (
            f"IsingProblem(nodes={len(self._nodes)}, edges={self._edge_count}, "
            f"constant={self._constant!r})"
        )

    def _add_node(self, node):
        node = _as_node(node)
        if node in self._nodes:
            return
        self._nodes[node] = None
        self._adj[node] = {}

    def _remove_coupling(self, u, v):
        """Remove one coupling and update its compact counters."""
        u = _as_node(u)
        v = _as_node(v)
        if u == v or u not in self._adj or v not in self._adj[u]:
            return
        value = self._adj[u].pop(v)
        del self._adj[v][u]
        self._edge_count -= 1
        self._sum_abs = max(0.0, self._sum_abs - abs(value))
        if self._edge_count == 0:
            self._sum_abs = 0.0

    def _remove_node(self, node):
        """Remove an isolated node.  All incident couplings must be gone."""
        node = _as_node(node)
        if self._adj[node]:
            raise RuntimeError("Remove all incident couplings before the node.")
        del self._adj[node]
        del self._nodes[node]
        self._fields.pop(node, None)

    def add_node(self, node):
        """Add one isolated node.  Existing nodes and couplings stay."""
        self._add_node(node)

    def field(self, node):
        """Return the local field, or zero for a missing field."""
        node = _as_node(node, "field node")
        return self._fields.get(node, 0.0)

    def coupling(self, u, v):
        """Return the coupling, or zero for a missing pair."""
        u = _as_node(u, "coupling node")
        v = _as_node(v, "coupling node")
        if u == v:
            raise ValueError("A self-coupling is not supported.")
        if u not in self._adj or v not in self._adj[u]:
            return 0.0
        return self._adj[u][v]

    def neighbors(self, node):
        """Return a copy of the coupling dictionary of one node."""
        node = _as_node(node, "node")
        if node not in self._adj:
            raise KeyError(node)
        return dict(self._adj[node])

    def set_field(self, node, value):
        """Set one local field.  An exact zero removes the stored field."""
        node = _as_node(node, "field node")
        value = _as_finite(value, "field")
        self._add_node(node)
        if value == 0.0:
            self._fields.pop(node, None)
        else:
            self._fields[node] = value

    def set_coupling(self, u, v, value):
        """Set one pair coupling.  An exact zero removes the stored pair."""
        u = _as_node(u, "coupling node")
        v = _as_node(v, "coupling node")
        if u == v:
            raise ValueError("A self-coupling is not supported.")
        value = _as_finite(value, "coupling")
        self._add_node(u)
        self._add_node(v)
        old = self._adj[u].get(v, 0.0)
        if old == 0.0 and value == 0.0:
            return
        if old == 0.0:
            self._edge_count += 1
            self._sum_abs += abs(value)
        elif value == 0.0:
            self._edge_count -= 1
            self._sum_abs = max(0.0, self._sum_abs - abs(old))
        else:
            self._sum_abs += abs(value) - abs(old)
        self._sum_abs = max(0.0, self._sum_abs)
        if self._edge_count == 0:
            self._sum_abs = 0.0
        if value == 0.0:
            del self._adj[u][v]
            del self._adj[v][u]
        else:
            self._adj[u][v] = value
            self._adj[v][u] = value

    def energy(self, solution):
        """Calculate the polynomial energy of one complete assignment.

        Parameters
        ----------
        solution : mapping of int to int
            One integer spin +1 or -1 for every node.

        Returns
        -------
        float
            ``c + sum h_i s_i + sum J_ij s_i s_j``.

        Raises
        ------
        TypeError
            The solution is not a mapping.
        ValueError
            A node is missing, duplicated, or a spin is not integer ±1.
        """
        if not isinstance(solution, Mapping):
            raise TypeError("solution must be a mapping.")
        if set(solution) != set(self._nodes):
            raise ValueError("The solution must contain every node exactly once.")
        spins = {}
        for node, spin in solution.items():
            node = _as_node(node, "solution node")
            if isinstance(spin, bool) or not isinstance(spin, Integral):
                raise ValueError("Every spin must be an integer +1 or -1.")
            spin = int(spin)
            if spin not in (-1, 1):
                raise ValueError("Every spin must be an integer +1 or -1.")
            spins[node] = spin
        terms = [self._constant]
        terms.extend(
            value * spins[node] for node, value in self._fields.items()
        )
        terms.extend(
            value * spins[u] * spins[v]
            for u in self._nodes
            for v, value in self._adj[u].items()
            if u < v
        )
        return float(math.fsum(terms))

    def copy(self):
        """Return an independent problem with the same coefficients."""
        return IsingProblem.from_snapshot(self.snapshot())

    def to_weighted_graph(self):
        """Return the equivalent Max-Cut graph if all fields are zero.

        Raises
        ------
        ValueError
            A nonzero local field is present.  Such a problem is not a
            public Max-Cut graph, and its constant needs separate handling.
        """
        if any(value != 0.0 for value in self._fields.values()):
            raise ValueError("A graph view is not defined for nonzero fields.")
        edges = [
            (u, v, 2.0 * self._adj[u][v])
            for u in self._nodes
            for v in self._adj[u]
            if u < v
        ]
        return WeightedGraph(edges=edges, nodes=self.nodes)

    def snapshot(self):
        """Return a JSON-compatible problem record."""
        data = {
            "schema_version": 1,
            "problem_type": "ising",
            "constant": self._constant,
            "nodes": list(self._nodes),
            "fields": [
                {"node": node, "value": value}
                for node, value in self._fields.items()
            ],
            "couplings": [
                {"u": u, "v": v, "value": self._adj[u][v]}
                for u in self._nodes
                for v in self._adj[u]
                if u < v
            ],
        }
        return to_json_data(data)

    @classmethod
    def from_snapshot(cls, data):
        """Restore a problem from a snapshot record."""
        if not isinstance(data, Mapping):
            raise TypeError("An Ising snapshot must be a mapping.")
        if data.get("schema_version") != 1 or data.get("problem_type") != "ising":
            raise ValueError("Unsupported Ising snapshot.")
        fields = {
            _as_node(item["node"], "field node"): item["value"]
            for item in data.get("fields", [])
        }
        couplings = {
            (_as_node(item["u"], "coupling node"), _as_node(item["v"], "coupling node")): item["value"]
            for item in data.get("couplings", [])
        }
        problem = cls(
            constant=data["constant"],
            fields=fields,
            couplings=couplings,
            nodes=data.get("nodes", ()),
        )
        # The explicit sum is authoritative after a round trip.
        problem._sum_abs = math.fsum(abs(value) for value in problem.couplings.values())
        return problem


class ReductionState:
    """Store one mutable reduction process for a graph or Ising problem.

    Parameters
    ----------
    The constructor is internal.  Use ``GraphReducer.start`` or
    ``GraphReducer.start_from_ising``.

    Attributes
    ----------
    original_nodes : tuple of int
        All original labels in their insertion order.
    constraints : tuple of SpinConstraint
        Constraints in application order.
    trace : tuple of dict
        Compact event records from applied reductions.
    cut_offset : float or None
        Accumulated Max-Cut offset ``b_t``.  It is None for a general
        Ising origin or a problem with nonzero fields.
    version : int
        Number of applied reductions.

    Notes
    -----
    The original graph and Ising problem are copied at start.  A reduction
    changes only the working Ising problem and the state records.
    """

    def __init__(
        self,
        *,
        original_nodes,
        ising,
        original_graph=None,
        original_ising=None,
        maxcut_origin,
        cut_offset,
        constraints=(),
        trace=(),
        version=0,
    ):
        self._original_nodes = tuple(_as_node(node) for node in original_nodes)
        if len(set(self._original_nodes)) != len(self._original_nodes):
            raise ValueError("Original node labels must be unique.")
        if not isinstance(ising, IsingProblem):
            raise TypeError("ising must be an IsingProblem.")
        self._ising = ising
        self._maxcut_origin = bool(maxcut_origin)
        self._original_graph = original_graph
        self._original_ising = original_ising
        self._constraints = list(constraints)
        self._trace = [deepcopy(event) for event in trace]
        self._version = int(version)
        if self._version < 0:
            raise ValueError("version must be nonnegative.")
        if self._maxcut_origin:
            if not isinstance(self._original_graph, WeightedGraph):
                raise ValueError("A Max-Cut state needs an original graph.")
            if tuple(self._original_graph.nodes) != self._original_nodes:
                raise ValueError("Original graph labels changed.")
            if self._original_ising is not None:
                raise ValueError("A Max-Cut state does not need an Ising origin.")
            self._cut_offset = _as_finite(cut_offset, "cut_offset")
        else:
            if not isinstance(self._original_ising, IsingProblem):
                raise ValueError("A general Ising state needs an original problem.")
            if tuple(self._original_ising.nodes) != self._original_nodes:
                raise ValueError("Original Ising labels changed.")
            if self._original_graph is not None:
                raise ValueError("A general Ising state does not need a graph.")
            if cut_offset is not None:
                raise ValueError("cut_offset is not defined for a general Ising problem.")
            self._cut_offset = None
        if set(self._ising.nodes) - set(self._original_nodes):
            raise ValueError("A working node is absent from the original task.")
        if len(self._trace) != len(self._constraints):
            raise ValueError("Trace and constraint counts differ.")

    @classmethod
    def from_graph(cls, graph):
        """Start a Max-Cut reduction state from an independent graph copy."""
        if not isinstance(graph, WeightedGraph):
            raise TypeError("graph must be a WeightedGraph.")
        original = graph.copy()
        return cls(
            original_nodes=original.nodes,
            ising=IsingProblem.from_maxcut(original),
            original_graph=original,
            maxcut_origin=True,
            cut_offset=0.0,
        )

    @classmethod
    def from_ising(cls, problem):
        """Start a general Ising reduction state from an independent copy."""
        if not isinstance(problem, IsingProblem):
            raise TypeError("problem must be an IsingProblem.")
        original = problem.copy()
        working = problem.copy()
        return cls(
            original_nodes=original.nodes,
            ising=working,
            original_ising=original,
            maxcut_origin=False,
            cut_offset=None,
        )

    @property
    def original_nodes(self):
        """Return all original labels in insertion order."""
        return self._original_nodes

    @property
    def original_graph(self):
        """Return an independent copy of the original graph."""
        if self._original_graph is None:
            raise ValueError("This state has no original graph.")
        return self._original_graph.copy()

    @property
    def original_ising(self):
        """Return an independent copy of the original Ising problem."""
        if self._original_ising is None:
            raise ValueError("This state has no original Ising problem.")
        return self._original_ising.copy()

    @property
    def ising(self):
        """Return the working Ising problem.  Treat it as read-only."""
        return self._ising

    @property
    def constraints(self):
        """Return constraints in application order."""
        return tuple(self._constraints)

    @property
    def trace(self):
        """Return independent trace events in application order."""
        return tuple(deepcopy(event) for event in self._trace)

    @property
    def cut_offset(self):
        """Return the accumulated Max-Cut offset, or None."""
        return self._cut_offset

    @property
    def version(self):
        """Return the number of applied reductions."""
        return self._version

    @property
    def current_nodes(self):
        """Return the current working labels in insertion order."""
        return self._ising.nodes

    @property
    def maxcut_metrics_applicable(self):
        """Return True when graph cut metrics are mathematically defined."""
        return self._maxcut_reason() is None

    def _maxcut_reason(self):
        if not self._maxcut_origin:
            return "original_problem_is_general_ising"
        if any(value != 0.0 for value in self._ising.fields.values()):
            return "working_problem_has_fields"
        return None

    def current_graph(self):
        """Return the current working graph for a Max-Cut state.

        Raises
        ------
        ValueError
            The state started from a general Ising problem or has fields.
        """
        reason = self._maxcut_reason()
        if reason is not None:
            raise ValueError(f"A Max-Cut graph view is not available: {reason}.")
        return self._ising.to_weighted_graph()

    def current_cut_value(self, solution):
        """Calculate the original cut for a residual solution.

        The value is ``b_t + C_current(solution)``.  The solution must
        contain every current node exactly once.
        """
        if self._maxcut_reason() is not None:
            raise ValueError("Cut metrics are not defined for this state.")
        offset = self._cut_offset
        if offset is None:
            raise RuntimeError("The Max-Cut offset is missing.")
        return offset + self.current_graph().cut_value(solution)

    def original_cut_value(self, solution):
        """Calculate C on the independent original graph."""
        if self._original_graph is None:
            raise ValueError("This state has no original graph.")
        return self._original_graph.cut_value(solution)

    def original_energy(self, solution):
        """Calculate E on the original task.

        For Max-Cut, ``E = -C``.  For a general Ising origin, the original
        polynomial is evaluated directly.
        """
        if self._original_graph is not None:
            return -self.original_cut_value(solution)
        original = self._original_ising
        if original is None:
            raise RuntimeError("The original Ising problem is missing.")
        return original.energy(solution)

    def current_energy(self, solution):
        """Calculate the working Ising energy on current nodes."""
        return self._ising.energy(solution)

    def apply_pair(
        self,
        removed,
        retained,
        sign,
        *,
        correlation=None,
        selection_score=None,
    ):
        """Apply one exact pair reduction and append one trace event.

        Parameters
        ----------
        removed, retained : int
            Current working labels.  They must differ.
        sign : int
            +1 or -1.  The constraint is ``s_removed = sign * s_retained``.
        correlation : float, optional
            Physical ``<Z_removed Z_retained>`` in [-1, 1].  It is not a
            selection score and is not used to choose the pair.
        selection_score : float, optional
            Separate finite score for a future selector.  This method only
            records it.

        Returns
        -------
        dict
            The trace event of this reduction.

        Notes
        -----
        All validation and all new coefficients are checked before the
        working problem changes.  A failed request leaves the state intact.
        The algebraic update touches the neighbors of ``removed`` only.
        Its cost is O(deg(u)) for the coefficients and the offset.  Building
        one trace event uses the maintained compact counters; it does not
        copy the graph or recompute a graph fingerprint.

        Raises
        ------
        TypeError
            A label is not an integer or a value is not a real number.
        ValueError
            A node is absent, the pair is a self-pair, the sign is not
            integer ±1, the correlation is outside [-1, 1], or a selected
            value is not finite.
        ArithmeticError
            A new constant, field, or coupling is not finite.
        """
        u = _as_node(removed, "removed")
        v = _as_node(retained, "retained")
        if u == v:
            raise ValueError("A node cannot be reduced with itself.")
        if u not in self._ising._adj:
            raise ValueError(f"Removed node is not in the working task: {u}")
        if v not in self._ising._adj:
            raise ValueError(f"Retained node is not in the working task: {v}")
        sigma = _as_sign(sign)

        corr = None
        corr_reason = "physical_correlation_not_provided"
        if correlation is not None:
            corr = _as_finite(correlation, "physical_correlation")
            if not -1.0 <= corr <= 1.0:
                raise ValueError("physical_correlation must be in [-1, 1].")
            corr_reason = None
        score = None
        score_reason = "selection_score_not_provided"
        if selection_score is not None:
            score = _as_finite(selection_score, "selection_score")
            score_reason = None

        before_version = self._version
        before_constant = self._ising.constant
        selected_coupling = self._ising.coupling(u, v)
        field_u = self._ising.field(u)
        field_v = self._ising.field(v)

        # s_u = sigma * s_v.  The edge term J_uv*s_u*s_v becomes
        # sigma*J_uv because s_v*s_v = 1.  This is the only term that
        # leaves the working polynomial as a constant.
        after_constant = before_constant + sigma * selected_coupling
        # The field term h_u*s_u becomes sigma*h_u*s_v.
        after_field_v = field_v + sigma * field_u
        if not math.isfinite(after_constant):
            raise ArithmeticError("The reduced constant is not finite.")
        if not math.isfinite(after_field_v):
            raise ArithmeticError("The reduced field is not finite.")

        incident = list(self._ising.neighbors(u).items())
        updates = {}
        for k, coupling_uk in incident:
            if k == v:
                continue
            # The term J_uk*s_u*s_k becomes sigma*J_uk*s_v*s_k.  Add it
            # to the existing J_vk term.  A missing J_vk is zero.
            value = self._ising.coupling(v, k) + sigma * coupling_uk
            if not math.isfinite(value):
                raise ArithmeticError("A reduced coupling is not finite.")
            updates[k] = value

        maxcut_reason = self._maxcut_reason()
        metrics_ok = maxcut_reason is None
        if metrics_ok:
            incident_weight = math.fsum(2.0 * value for _, value in incident)
            offset_delta = 0.5 * (1.0 - sigma) * incident_weight
            current_offset = self._cut_offset
            if current_offset is None:
                raise RuntimeError("The Max-Cut offset is missing.")
            after_offset = current_offset + offset_delta
            if not math.isfinite(after_offset):
                raise ArithmeticError("The accumulated cut offset is not finite.")
        else:
            incident_weight = None
            offset_delta = None
            after_offset = None

        if metrics_ok:
            selected_weight = 2.0 * selected_coupling
            fixed_energy = -0.5 * selected_weight * (1.0 - sigma)
            if corr is not None:
                expected_fixed_energy = -0.5 * selected_weight * (1.0 - corr)
            else:
                expected_fixed_energy = None
            for value in (selected_weight, fixed_energy, expected_fixed_energy):
                if value is not None and not math.isfinite(value):
                    raise ArithmeticError("A reduction metric is not finite.")
        else:
            selected_weight = None
            fixed_energy = None
            expected_fixed_energy = None
        constraint_probability = (
            None if corr is None else 0.5 * (1.0 + sigma * corr)
        )

        # All checks passed.  Apply local updates.  No operation below
        # depends on user input, so the state cannot be half changed by a
        # rejected request.
        for k, _ in incident:
            self._ising._remove_coupling(u, k)
        self._ising._remove_node(u)
        self._ising.set_field(v, after_field_v)
        for k, value in updates.items():
            self._ising.set_coupling(v, k, value)
        self._ising._constant = after_constant
        self._version += 1
        if metrics_ok:
            self._cut_offset = after_offset

        constraint = SpinConstraint(u, v, sigma)
        self._constraints.append(constraint)
        sum_abs_coupling = self._ising.sum_abs_couplings
        if self._ising.edge_count:
            mean_abs_coupling = sum_abs_coupling / self._ising.edge_count
        else:
            mean_abs_coupling = None
        if metrics_ok:
            sum_abs_weight = 2.0 * sum_abs_coupling
            mean_abs_weight = (
                2.0 * mean_abs_coupling
                if mean_abs_coupling is not None
                else None
            )
        else:
            sum_abs_weight = None
            mean_abs_weight = None

        event = {
            "step": len(self._constraints),
            "task_version_before": before_version,
            "task_version_after": self._version,
            "removed_node": u,
            "retained_node": v,
            "sign": sigma,
            "selected_coupling": selected_coupling,
            "selected_weight": selected_weight,
            "physical_correlation": corr,
            "correlation_reason": corr_reason,
            "selection_score": score,
            "selection_score_reason": score_reason,
            "fixed_energy": fixed_energy,
            "fixed_energy_reason": (
                None if metrics_ok else maxcut_reason
            ),
            "expected_fixed_energy": expected_fixed_energy,
            "expected_fixed_energy_reason": (
                maxcut_reason
                if not metrics_ok
                else (
                    "physical_correlation_not_provided"
                    if corr is None
                    else None
                )
            ),
            "constraint_probability": constraint_probability,
            "cut_offset": after_offset,
            "cut_offset_delta": offset_delta,
            "ising_constant": after_constant,
            "ising_constant_before": before_constant,
            "ising_constant_delta": after_constant - before_constant,
            "vertices_after": len(self._ising),
            "edges_after": self._ising.edge_count,
            "sum_abs_weight": sum_abs_weight,
            "mean_abs_weight": mean_abs_weight,
            "sum_abs_coupling": sum_abs_coupling,
            "mean_abs_coupling": mean_abs_coupling,
            "mean_abs_weight_domain": "nonzero_edges_after",
            "mean_abs_weight_reason": (
                maxcut_reason
                if not metrics_ok
                else (
                    "empty_nonzero_edge_set"
                    if mean_abs_weight is None
                    else None
                )
            ),
            "maxcut_metrics_applicable": metrics_ok,
            "maxcut_metrics_reason": maxcut_reason,
        }
        self._trace.append(event)
        return deepcopy(event)

    def restore(self, residual_solution):
        """Restore all original spins from one residual assignment.

        Parameters
        ----------
        residual_solution : mapping of int to int
            One integer spin for every current node.

        Returns
        -------
        dict of int to int
            One spin for every original node.

        Notes
        -----
        Constraints are reversed.  A retained node may have been removed
        later; the reverse order restores its spin first.  The cost is
        O(N + number of constraints).  The state is not changed.

        Raises
        ------
        TypeError
            The residual is not a mapping.
        ValueError
            A node is missing, extra, duplicated, or a spin is not ±1.
        RuntimeError
            The stored constraint chain is incomplete.
        """
        if not isinstance(residual_solution, Mapping):
            raise TypeError("residual_solution must be a mapping.")
        current = set(self._ising.nodes)
        found = set()
        solution = {}
        for node, spin in residual_solution.items():
            node = _as_node(node, "solution node")
            if isinstance(spin, bool) or not isinstance(spin, Integral):
                raise ValueError("Every spin must be an integer +1 or -1.")
            spin = int(spin)
            if spin not in (-1, 1):
                raise ValueError("Every spin must be an integer +1 or -1.")
            if node not in current:
                raise ValueError(f"Node is not in the current task: {node}")
            if node in found:
                raise ValueError(f"Duplicate solution node: {node}")
            found.add(node)
            solution[node] = spin
        if found != current:
            raise ValueError("The residual solution must contain every current node.")
        # A retained node can be removed by a later constraint.  The
        # reverse order restores its spin before an earlier constraint uses it.
        for constraint in reversed(self._constraints):
            if constraint.retained not in solution:
                raise RuntimeError("The constraint chain is incomplete.")
            solution[constraint.removed] = (
                constraint.sign * solution[constraint.retained]
            )
        if set(solution) != set(self._original_nodes):
            raise RuntimeError("Restoration did not recover every original node.")
        return {int(node): int(spin) for node, spin in solution.items()}

    def snapshot(self):
        """Return a versioned JSON-compatible mathematical state.

        The snapshot is not an optimizer checkpoint.  It stores the working
        polynomial, the original task, the constraints, the offset and the
        compact trace.
        """
        data = {
            "schema_version": 1,
            "state_type": "reduction_state",
            "energy_convention": _ENERGY_CONVENTION,
            "version": self._version,
            "maxcut_origin": self._maxcut_origin,
            "original_nodes": list(self._original_nodes),
            "original_graph": (
                None
                if self._original_graph is None
                else self._original_graph.snapshot()
            ),
            "original_ising": (
                None
                if self._original_ising is None
                else self._original_ising.snapshot()
            ),
            "ising": self._ising.snapshot(),
            "cut_offset": self._cut_offset,
            "constraints": [
                constraint.to_dict() for constraint in self._constraints
            ],
            "trace": deepcopy(self._trace),
        }
        return to_json_data(data)

    @classmethod
    def from_snapshot(cls, data):
        """Restore a reduction state from a snapshot record."""
        if not isinstance(data, Mapping):
            raise TypeError("A reduction snapshot must be a mapping.")
        if data.get("schema_version") != 1 or data.get("state_type") != "reduction_state":
            raise ValueError("Unsupported reduction snapshot.")
        original_nodes = tuple(
            _as_node(node) for node in data.get("original_nodes", ())
        )
        ising = IsingProblem.from_snapshot(data["ising"])
        maxcut_origin = bool(data["maxcut_origin"])
        original_graph = (
            None
            if data.get("original_graph") is None
            else WeightedGraph.from_snapshot(data["original_graph"])
        )
        original_ising = (
            None
            if data.get("original_ising") is None
            else IsingProblem.from_snapshot(data["original_ising"])
        )
        constraints = [
            SpinConstraint.from_dict(item)
            for item in data.get("constraints", [])
        ]
        trace = [dict(item) for item in data.get("trace", [])]
        state = cls(
            original_nodes=original_nodes,
            ising=ising,
            original_graph=original_graph,
            original_ising=original_ising,
            maxcut_origin=maxcut_origin,
            cut_offset=data.get("cut_offset"),
            constraints=constraints,
            trace=trace,
            version=data.get("version", 0),
        )
        state._validate_chain()
        return state

    def _validate_chain(self):
        """Check that constraints describe one valid reduction chain."""
        remaining = set(self._original_nodes)
        for constraint in self._constraints:
            if constraint.removed not in remaining:
                raise ValueError("A snapshot removes a node twice.")
            if constraint.retained not in remaining:
                raise ValueError("A snapshot retains a missing node.")
            remaining.remove(constraint.removed)
        if remaining != set(self._ising.nodes):
            raise ValueError("Snapshot constraints and working nodes disagree.")


class GraphReducer:
    """Start, apply, restore, and save exact pair reductions.

    The component is stateless.  All mutable data lives in a
    ``ReductionState``.  This keeps one solver or experiment free to hold
    several independent reduction processes.
    """

    def start(self, graph):
        """Start a Max-Cut reduction process from a graph.

        Parameters
        ----------
        graph : WeightedGraph
            The graph is copied.  The caller's object is not changed.

        Returns
        -------
        ReductionState
            A new state with ``b_0 = 0`` and an Ising working problem.

        Raises
        ------
        TypeError
            ``graph`` is not a WeightedGraph.
        """
        return ReductionState.from_graph(graph)

    def start_from_ising(self, problem):
        """Start a general Ising reduction process.

        Parameters
        ----------
        problem : IsingProblem
            The problem is copied into an original task and a working task.

        Returns
        -------
        ReductionState
            A state without Max-Cut cut-offset metrics.
        """
        return ReductionState.from_ising(problem)

    def reduce(
        self,
        state,
        removed,
        retained,
        sign,
        *,
        correlation=None,
        selection_score=None,
    ):
        """Apply one reduction to a state and return its trace event.

        Parameters
        ----------
        state : ReductionState
            The state changes in place only after all validation passes.
        removed, retained : int
            Current working labels.
        sign : int
            +1 or -1.
        correlation : float, optional
            Physical correlation in [-1, 1].
        selection_score : float, optional
            Separate finite selection score.

        Returns
        -------
        dict
            One compact trace event.

        Notes
        -----
        This method delegates to ``state.apply_pair``.  See that method for
        the exact algebra and error contract.
        """
        if not isinstance(state, ReductionState):
            raise TypeError("state must be a ReductionState.")
        return state.apply_pair(
            removed,
            retained,
            sign,
            correlation=correlation,
            selection_score=selection_score,
        )

    def restore(self, state, residual_solution):
        """Restore original spins from a residual assignment.

        Parameters
        ----------
        state : ReductionState
            A state with one or more applied constraints.
        residual_solution : mapping of int to int
            One spin for every current node.

        Returns
        -------
        dict of int to int
            One spin for every original node.

        Notes
        -----
        This method delegates to ``state.restore``.
        """
        if not isinstance(state, ReductionState):
            raise TypeError("state must be a ReductionState.")
        return state.restore(residual_solution)

    def snapshot(self, state):
        """Return the mathematical state snapshot.

        Parameters
        ----------
        state : ReductionState
            The state to serialize.

        Returns
        -------
        dict
            A JSON-compatible ``reduction_state`` record.
        """
        if not isinstance(state, ReductionState):
            raise TypeError("state must be a ReductionState.")
        return state.snapshot()

    def from_snapshot(self, data):
        """Restore a mathematical state snapshot.

        Parameters
        ----------
        data : mapping
            A record returned by ``snapshot``.

        Returns
        -------
        ReductionState
            A state that can accept further reductions and restore spins.
        """
        return ReductionState.from_snapshot(data)
