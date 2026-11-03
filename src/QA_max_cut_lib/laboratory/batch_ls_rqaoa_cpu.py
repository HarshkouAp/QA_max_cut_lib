"""Public route for batch LS-RQAOA (CPU)."""
from ..solvers.batch_ls import BatchLSRQAOACPU

def main(argv=None):
    from ..tools.demo import run_solver_demo
    return run_solver_demo(BatchLSRQAOACPU, argv)

__all__ = ["BatchLSRQAOACPU", "main"]
if __name__ == "__main__":
    raise SystemExit(main())
