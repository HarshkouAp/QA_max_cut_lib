"""Independent CPU solver implementations."""

from importlib import import_module


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
        "AnalyticP1Evaluator": ".analytic_p1",
        "AnalyticP1CorrelationSource": ".analytic_p1",
    }
    if name not in modules:
        raise AttributeError(name)
    return getattr(import_module(modules[name], __name__), name)
