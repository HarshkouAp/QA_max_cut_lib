"""Public Laboratory route for QAOA with passive MI diagnostics."""

from ..solvers.mi_observer import QAOAWithMIObserverCPU


def main(argv=None):
    from ..tools.demo import run_solver_demo
    return run_solver_demo(QAOAWithMIObserverCPU, argv)


__all__ = ["QAOAWithMIObserverCPU", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
