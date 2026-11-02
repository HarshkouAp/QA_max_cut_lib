"""RQAOA with the NVIDIA GPU correlation source."""

from __future__ import annotations

from .rqaoa_cpu import RQAOACPU

__all__ = ["RQAOAGPU"]


class RQAOAGPU(RQAOACPU):
    """Run the shared RQAOA coordinator with a GPU state-vector source.

    All reduction math, selection, budgets, and restoration are inherited
    from :class:`RQAOACPU`.  The device is used only by the nested correlation
    source, so the public backend choice remains explicit.
    """

    algorithm_type = "rqaoa_gpu"

    def _validate(self, config, graph):
        super()._validate(config, graph)
        source_type = config["correlation_source"].get("type")
        if source_type != "qaoa_statevector_gpu":
            raise ValueError(
                "RQAOAGPU requires correlation_source.type='qaoa_statevector_gpu'."
            )


def main(argv=None):
    """Run a small GPU RQAOA example through the common demo interface."""
    from ..tools.demo import run_solver_demo

    return run_solver_demo(RQAOAGPU, argv)


if __name__ == "__main__":
    raise SystemExit(main())
