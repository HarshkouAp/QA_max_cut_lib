"""Solve and inspect weighted Max-Cut tasks.

The first milestone contains two CPU solvers. Import optional plotting
libraries only when a plotting function needs them.
"""

from importlib import import_module
from .core.graph import WeightedGraph
from .core.records import RunRecord

__version__ = "0.1.0a1"
__all__ = [
    "WeightedGraph",
    "RunRecord",
    "GraphFactory",
    "SolverFactory",
    "ExactCPU",
    "QAOACPU",
]

_EXPORTS = {
    "GraphFactory": (".generators.graphs", "GraphFactory"),
    "SolverFactory": (".core.factory", "SolverFactory"),
    "ExactCPU": (".solvers.exact_cpu", "ExactCPU"),
    "QAOACPU": (".solvers.qaoa_cpu", "QAOACPU"),
}


def __getattr__(name):
    """Load a public class when the caller first requests it."""
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value
