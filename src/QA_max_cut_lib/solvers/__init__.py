"""Independent CPU solver implementations."""

from importlib import import_module


def restore(path, **kwargs):
    """Restore a solver object from a checkpoint without running it."""
    from ..core.checkpoint import restore as _restore

    return _restore(path, **kwargs)


def from_config(config):
    """Construct a solver through the package factory."""
    from ..core.factory import SolverFactory

    return SolverFactory.from_config(config)


def __getattr__(name):
    modules = {
        "ExactCPU": ".exact_cpu",
        "QAOACPU": ".qaoa_cpu",
        "RQAOACPU": ".rqaoa_cpu",
        "LSRQAOACPU": ".ls_rqaoa_cpu",
        "QAOAGPU": ".qaoa_gpu",
        "QAOAGPUCorrelationSource": ".qaoa_gpu",
        "RQAOAGPU": ".rqaoa_gpu",
        "LSRQAOAGPU": ".ls_rqaoa_gpu",
        "QAOAInQAOACPU": ".qaoa_in_qaoa_cpu",
        "QAOAInQAOAGPU": ".qaoa_in_qaoa_gpu",
        "GoemansWilliamsonCPU": ".goemans_williamson_cpu",
        "SimulatedAnnealingCPU": ".simulated_annealing_cpu",
        "ExactBlockRefiner": ".exact_block_refiner",
        "RefinedSolver": ".refined",
        "AnalyticP1Evaluator": ".analytic_p1",
        "AnalyticP1CorrelationSource": ".analytic_p1",
        "WarmStartQAOACPU": ".warm_start_qaoa_cpu",
        "WarmStartRQAOACPU": ".warm_start_rqaoa_cpu",
        "WarmStartEnsembleCorrelationSource": ".warm_start_ensemble",
        "ConditionalQAOACPU": ".conditional_qaoa_cpu",
        "BlockMeanFieldQAOACPU": ".block_mean_field_qaoa_cpu",
        "DistributedLSRQAOACPU": ".distributed_ls",
        "IncrementalLSRQAOACPU": ".incremental_ls",
        "QAOAWithMIObserverCPU": ".mi_observer",
        "RQAOAWithMIObserverCPU": ".mi_observer",
        "EnvironmentLSRQAOACPU": ".environment_ls_rqaoa_cpu",
    }
    if name not in modules:
        raise AttributeError(name)
    return getattr(import_module(modules[name], __name__), name)
