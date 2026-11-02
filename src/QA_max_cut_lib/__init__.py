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
    "WarmStartQAOACPU",
    "WarmStartRQAOACPU",
    "WarmStartEnsembleCorrelationSource",
    "RQAOACPU",
    "LSRQAOACPU",
    "QAOAGPU",
    "QAOAGPUCorrelationSource",
    "RQAOAGPU",
    "LSRQAOAGPU",
    "QAOAInQAOACPU",
    "QAOAInQAOAGPU",
    "GoemansWilliamsonCPU",
    "SimulatedAnnealingCPU",
    "ExactBlockRefiner",
    "RefinedSolver",
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
    "BenchmarkSpec",
    "ExperimentRunner",
    "ExperimentSeries",
    "ParallelismProfiler",
    "ProfileSpec",
    "ReferencePolicy",
    "RunControl",
    "restore",
    "inspect_checkpoint",
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
    "WarmStartQAOACPU": (".solvers.warm_start_qaoa_cpu", "WarmStartQAOACPU"),
    "WarmStartRQAOACPU": (".solvers.warm_start_rqaoa_cpu", "WarmStartRQAOACPU"),
    "WarmStartEnsembleCorrelationSource": (".solvers.warm_start_ensemble", "WarmStartEnsembleCorrelationSource"),
    "RQAOACPU": (".solvers.rqaoa_cpu", "RQAOACPU"),
    "LSRQAOACPU": (".solvers.ls_rqaoa_cpu", "LSRQAOACPU"),
    "QAOAGPU": (".solvers.qaoa_gpu", "QAOAGPU"),
    "QAOAGPUCorrelationSource": (".solvers.qaoa_gpu", "QAOAGPUCorrelationSource"),
    "RQAOAGPU": (".solvers.rqaoa_gpu", "RQAOAGPU"),
    "LSRQAOAGPU": (".solvers.ls_rqaoa_gpu", "LSRQAOAGPU"),
    "QAOAInQAOACPU": (".solvers.qaoa_in_qaoa_cpu", "QAOAInQAOACPU"),
    "QAOAInQAOAGPU": (".solvers.qaoa_in_qaoa_gpu", "QAOAInQAOAGPU"),
    "GoemansWilliamsonCPU": (".solvers.goemans_williamson_cpu", "GoemansWilliamsonCPU"),
    "SimulatedAnnealingCPU": (".solvers.simulated_annealing_cpu", "SimulatedAnnealingCPU"),
    "ExactBlockRefiner": (".solvers.exact_block_refiner", "ExactBlockRefiner"),
    "RefinedSolver": (".solvers.refined", "RefinedSolver"),
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
    "BenchmarkSpec": (".core.experiment_spec", "BenchmarkSpec"),
    "ExperimentRunner": (".core.runner", "ExperimentRunner"),
    "ExperimentSeries": (".core.runner", "ExperimentSeries"),
    "ParallelismProfiler": (".core.profile", "ParallelismProfiler"),
    "ProfileSpec": (".core.profile", "ProfileSpec"),
    "ReferencePolicy": (".core.references", "ReferencePolicy"),
    "RunControl": (".core.checkpoint", "RunControl"),
    "restore": (".core.checkpoint", "restore"),
    "inspect_checkpoint": (".core.checkpoint", "inspect_checkpoint"),
}


def __getattr__(name):
    """Load a public class when the caller first requests it."""
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value
