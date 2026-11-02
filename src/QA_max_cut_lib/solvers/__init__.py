"""Independent CPU solver implementations."""

from importlib import import_module


def __getattr__(name):
    modules = {
        "ExactCPU": ".exact_cpu",
        "QAOACPU": ".qaoa_cpu",
        "RQAOACPU": ".rqaoa_cpu",
        "AnalyticP1Evaluator": ".analytic_p1",
        "AnalyticP1CorrelationSource": ".analytic_p1",
    }
    if name not in modules:
        raise AttributeError(name)
    return getattr(import_module(modules[name], __name__), name)
