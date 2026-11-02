"""Public Laboratory route for RQAOA with passive MI diagnostics."""

from ..solvers.mi_observer import RQAOAWithMIObserverCPU


def main(argv=None):
    from ..tools.demo import run_solver_demo
    return run_solver_demo(RQAOAWithMIObserverCPU, argv)


__all__ = ["RQAOAWithMIObserverCPU", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
