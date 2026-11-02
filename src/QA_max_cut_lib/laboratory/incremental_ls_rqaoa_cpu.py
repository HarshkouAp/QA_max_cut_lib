"""Public route for incremental LS-RQAOA (CPU)."""

from ..solvers.incremental_ls import IncrementalLSRQAOACPU


def main(argv=None):
    from ..tools.demo import run_solver_demo
    return run_solver_demo(IncrementalLSRQAOACPU, argv)


__all__ = ["IncrementalLSRQAOACPU", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
