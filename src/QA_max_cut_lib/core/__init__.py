"""Common graph, result, configuration, and reduction contracts."""

from .reduction import GraphReducer, IsingProblem, ReductionState, SpinConstraint

__all__ = [
    "GraphReducer",
    "IsingProblem",
    "ReductionState",
    "SpinConstraint",
]
