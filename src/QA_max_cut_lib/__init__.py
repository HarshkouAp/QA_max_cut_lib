"""Solve, reduce, and inspect weighted Max-Cut tasks.

The package contains two CPU solvers and the exact pair-reduction core.
Import optional plotting libraries only when a plotting function needs
them.  Import CuPy only when a GPU implementation needs it.
"""

from importlib import import_module

from ._version import __version__
from .core.graph import WeightedGraph
from .core.records import RunRecord

__all__ = [
    "WeightedGraph",
    "RunRecord",
    "GraphFactory",
    "SolverFactory",
    "RandomPartitionBuilder",
    "GreedyAbsWeightPartitionBuilder",
    "ExactCPU",
    "QAOACPU",
    "RQAOACPU",
    "LSRQAOACPU",
    "AnalyticP1Evaluator",
    "AnalyticP1CorrelationSource",
    "QAOAStatevectorCorrelationSource",
    "MaxAbsCorrelationSelector",
    "MeanCorrelationAggregator",
    "ObservedMaxAbsCorrelationSelector",
    "ExactCPUFinisher",
    "CorrelationResult",
    "SelectionResult",
    "GraphReducer",
    "ReductionState",
    "IsingProblem",
    "SpinConstraint",
]

_EXPORTS = {
    "GraphFactory": (".generators.graphs", "GraphFactory"),
    "RandomPartitionBuilder": (".generators.subgraphs", "RandomPartitionBuilder"),
    "GreedyAbsWeightPartitionBuilder": (
        ".generators.subgraphs",
        "GreedyAbsWeightPartitionBuilder",
    ),
    "SolverFactory": (".core.factory", "SolverFactory"),
    "ExactCPU": (".solvers.exact_cpu", "ExactCPU"),
    "QAOACPU": (".solvers.qaoa_cpu", "QAOACPU"),
    "RQAOACPU": (".solvers.rqaoa_cpu", "RQAOACPU"),
    "LSRQAOACPU": (".solvers.ls_rqaoa_cpu", "LSRQAOACPU"),
    "AnalyticP1Evaluator": (".solvers.analytic_p1", "AnalyticP1Evaluator"),
    "AnalyticP1CorrelationSource": (
        ".solvers.analytic_p1",
        "AnalyticP1CorrelationSource",
    ),
    "QAOAStatevectorCorrelationSource": (
        ".solvers.components",
        "QAOAStatevectorCorrelationSource",
    ),
    "MaxAbsCorrelationSelector": (
        ".solvers.components",
        "MaxAbsCorrelationSelector",
    ),
    "MeanCorrelationAggregator": (
        ".solvers.ls_components",
        "MeanCorrelationAggregator",
    ),
    "ObservedMaxAbsCorrelationSelector": (
        ".solvers.ls_components",
        "ObservedMaxAbsCorrelationSelector",
    ),
    "ExactCPUFinisher": (".solvers.components", "ExactCPUFinisher"),
    "CorrelationResult": (".solvers.components", "CorrelationResult"),
    "SelectionResult": (".solvers.components", "SelectionResult"),
    "GraphReducer": (".core.reduction", "GraphReducer"),
    "ReductionState": (".core.reduction", "ReductionState"),
    "IsingProblem": (".core.reduction", "IsingProblem"),
    "SpinConstraint": (".core.reduction", "SpinConstraint"),
}


def __getattr__(name):
    """Load a public class when the caller first requests it."""
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value
