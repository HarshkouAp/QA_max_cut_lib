"""LS-RQAOA with GPU local correlation tasks and one device owner."""

from __future__ import annotations

from .ls_rqaoa_cpu import LSRQAOACPU

__all__ = ["LSRQAOAGPU"]


class LSRQAOAGPU(LSRQAOACPU):
    """Serve all GPU local tasks from one long-lived device context.

    The plan, aggregation, reduction, and budgets are inherited from the
    Stage 05 CPU coordinator.  Set ``execution.backend="gpu_owner"`` in the
    configuration; the coordinator keeps the task loop sequential so that
    independent worker processes never create competing CUDA contexts.
    """

    algorithm_type = "ls_rqaoa_gpu"

    def _validate(self, config, graph):
        super()._validate(config, graph)
        if config["correlation_source"].get("type") != "qaoa_statevector_gpu":
            raise ValueError(
                "LSRQAOAGPU requires correlation_source.type='qaoa_statevector_gpu'."
            )
        if config["execution"]["backend"] != "gpu_owner":
            raise ValueError(
                "LSRQAOAGPU requires execution.backend='gpu_owner'."
            )


def main(argv=None):
    """Run a small GPU LS-RQAOA example through the common demo interface."""
    from ..tools.demo import run_solver_demo

    return run_solver_demo(LSRQAOAGPU, argv)


if __name__ == "__main__":
    raise SystemExit(main())
