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
    "ExactCPU",
    "QAOACPU",
    "GraphReducer",
    "ReductionState",
    "IsingProblem",
    "SpinConstraint",
]

_EXPORTS = {
    "GraphFactory": (".generators.graphs", "GraphFactory"),
    "SolverFactory": (".core.factory", "SolverFactory"),
    "ExactCPU": (".solvers.exact_cpu", "ExactCPU"),
    "QAOACPU": (".solvers.qaoa_cpu", "QAOACPU"),
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
