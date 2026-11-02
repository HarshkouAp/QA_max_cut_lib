"""Local-search components layered on the shared correlation machinery."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
import math
from numbers import Real
import numpy as np

from ..core.reduction import ReductionState
from .analytic_p1 import AnalyticP1CorrelationSource, validate_analytic_p1_source_config
from .components import (
    CorrelationResult,
    MaxAbsCorrelationSelector,
    SelectionResult,
    validate_correlation_source_config,
)

__all__ = [
    "build_correlation_source",
    "MeanCorrelationAggregator",
    "ObservedMaxAbsCorrelationSelector",
]

_TOLERANCE = 1e-9


def build_correlation_source(config, *, seed=None):
    """Construct one supported source from a validated section."""
    kind = config.get("type")
    if kind == "qaoa_statevector_cpu":
        from .components import QAOAStatevectorCorrelationSource

        return QAOAStatevectorCorrelationSource(config, seed=seed)
    if kind == "qaoa_analytic_p1_cpu":
        return AnalyticP1CorrelationSource(config, seed=seed)
    raise ValueError(f"Unknown correlation source: {kind!r}.")


def _allowed_pairs(state, pairs):
    if pairs == "edges":
        return [
            (min(u, v), max(u, v))
            for u, v, weight in state.current_graph().edges
            if float(weight) != 0.0
        ]
    return list(combinations(sorted(state.current_nodes), 2))


class MeanCorrelationAggregator:
    """Average local estimates per pair and retain their provenance.

    Parameters
    ----------
    config : dict, optional
        ``{"type": "mean", "parameters": {}}``.
    """

    type = "mean"

    def __init__(self, config=None):
        config = deepcopy(config or {})
        if set(config) - {"type", "parameters"} or config.get(
            "type", self.type
        ) != self.type:
            raise ValueError("aggregation type must be mean.")
        if set(config.setdefault("parameters", {})):
            raise ValueError("The mean aggregator has no parameters.")

    @property
    def config(self):
        """Return the canonical aggregation section."""
        return {"type": self.type, "parameters": {}}

    def aggregate(self, state, task_results, *, pairs):
        """Aggregate complete local tables in canonical task order.

        Parameters
        ----------
        state : ReductionState
            Parent state that all tasks evaluated.
        task_results : sequence of dict
            Records with ``task_id``, ``values``, ``pairs``, and
            ``version``.  Only pairs inside the parent are retained.
        pairs : {'edges', 'all_pairs'}
            Allowed global pair domain.

        Returns
        -------
        dict
            Canonical mean values, observation counts, task IDs, and
            membership provenance.  Real zeros are retained.
        """
        if not isinstance(state, ReductionState):
            raise TypeError("state must be a ReductionState.")
        allowed = set(_allowed_pairs(state, pairs))
        sums = {}
        counts = {}
        origins = {}
        for result in sorted(task_results, key=lambda item: str(item["task_id"])):
            result_version = result.get("parent_version", result.get("version", -1))
            if int(result_version) != int(state.version):
                raise ValueError("A local result belongs to an older task version.")
            for raw_pair, raw_value in result["values"].items():
                pair = (min(int(raw_pair[0]), int(raw_pair[1])), max(int(raw_pair[0]), int(raw_pair[1])))
                if pair not in allowed:
                    continue
                value = float(raw_value)
                if not math.isfinite(value):
                    raise ValueError("A local correlation is not finite.")
                if abs(value) > 1.0 + _TOLERANCE:
                    raise ValueError("A local correlation is outside [-1, 1].")
                value = min(1.0, max(-1.0, value))
                sums[pair] = sums.get(pair, 0.0) + value
                counts[pair] = counts.get(pair, 0) + 1
                origins.setdefault(pair, []).append(str(result["task_id"]))
        values = {pair: sums[pair] / counts[pair] for pair in sorted(sums)}
        return {
            "values": values,
            "counts": {pair: counts[pair] for pair in sorted(counts)},
            "task_ids": {pair: tuple(origins[pair]) for pair in sorted(origins)},
            "allowed_pairs": tuple(sorted(allowed)),
            "observed_pairs": tuple(sorted(values)),
            "pairs": tuple(sorted(values)),
        }

    def make_table(self, state, aggregated):
        """Wrap an aggregation result in the shared correlation record."""
        return CorrelationResult(
            node_order=tuple(state.current_nodes),
            task_version=int(state.version),
            values=dict(aggregated["values"]),
            pairs=tuple(aggregated["pairs"]),
            mode="exact",
            angles=np.empty(0, dtype=float),
            layers=0,
            training={},
            resources={},
            objective_reference="working_graph",
        )


class ObservedMaxAbsCorrelationSelector(MaxAbsCorrelationSelector):
    """Select the largest absolute observed correlation.

    Unlike :class:`MaxAbsCorrelationSelector`, a missing allowed pair is not
    an error.  The parent coverage policy decides whether the observed set is
    sufficient before this selector runs.
    """

    def select(self, state, correlations):
        """Select from the observed subset of the allowed pairs."""
        if not isinstance(state, ReductionState):
            raise TypeError("state must be a ReductionState.")
        if not isinstance(correlations, CorrelationResult):
            raise TypeError("correlations must be a CorrelationResult.")
        if correlations.task_version != state.version:
            raise ValueError("The correlation table belongs to an older task.")
        if tuple(correlations.node_order) != tuple(state.current_nodes):
            raise ValueError("The correlation node order does not match the task.")
        if self._pairs == "edges":
            allowed = {
                (min(u, v), max(u, v))
                for u, v, weight in state.current_graph().edges
                if float(weight) != 0.0
            }
        else:
            allowed = set(combinations(sorted(state.current_nodes), 2))
        candidates = []
        for pair, raw_value in correlations.values.items():
            pair = (min(pair[0], pair[1]), max(pair[0], pair[1]))
            if pair not in allowed:
                continue
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError("A selected correlation is not finite.")
            if abs(value) > 1.0 + _TOLERANCE:
                raise ValueError("A selected correlation is outside [-1, 1].")
            value = min(1.0, max(-1.0, value))
            candidates.append((abs(value), pair, value))
        if not candidates:
            raise ValueError("No observed allowed pair exists for selection.")
        _, pair, correlation = min(
            candidates, key=lambda item: (-item[0], item[1][0], item[1][1])
        )
        sign = 1 if correlation > 0 else -1 if correlation < 0 else self._zero_sign
        return SelectionResult(
            removed=max(pair),
            retained=min(pair),
            sign=sign,
            score=abs(correlation),
            correlation=correlation,
        )
