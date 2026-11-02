"""QAOA-in-QAOA with the NVIDIA GPU local readout owner."""

from __future__ import annotations

from .qaoa_in_qaoa_cpu import QAOAInQAOACPU

__all__ = ["QAOAInQAOAGPU"]


class QAOAInQAOAGPU(QAOAInQAOACPU):
    """Run the shared hierarchy with one long-lived GPU local owner.

    The CPU coordinator owns the hierarchy, budgets, projection, and tree.
    This class requires ``local_solver.type='qaoa_gpu'`` and
    ``execution.backend='gpu_owner'``.  Local blocks are served sequentially
    so independent worker processes never create competing CUDA contexts.
    """

    algorithm_type = "qaoa_in_qaoa_gpu"


def main(argv=None):
    """Run a small GPU QAOA-in-QAOA example through the common interface."""
    from ..tools.demo import run_solver_demo

    return run_solver_demo(QAOAInQAOAGPU, argv)


if __name__ == "__main__":
    raise SystemExit(main())
