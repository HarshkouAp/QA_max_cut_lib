"""Laboratory assembly for multi-start warm-start RQAOA on CPU."""

from __future__ import annotations

from copy import deepcopy

from .rqaoa_cpu import RQAOACPU

__all__ = ["WarmStartRQAOACPU"]


class WarmStartRQAOACPU(RQAOACPU):
    """Run the verified recursive RQAOA loop with a warm-start ensemble source.

    The source is ``warm_start_ensemble_cpu``.  Each current graph gets its
    own bounded candidate batch, unique starts, child training tasks, and
    equal-weight correlation mixture.  The class adds only the Laboratory
    identity and result provenance; the reducer, selector, finisher, budget,
    and RQAOA loop remain shared with :class:`RQAOACPU`.

    Notes
    -----
    CPU full statevector only.  The source must use ``restart`` angle
    initialization; a previous-step single-source policy is ambiguous for
    an ensemble and is rejected before training.
    """

    algorithm_type = "laboratory.warm_start_rqaoa_cpu"
    optional_fields = RQAOACPU.optional_fields + (
        "variant_id",
        "warm_start_history",
        "aggregate_statevector",
        "aggregate_statevector_status",
    )

    def _validate(self, config, graph):
        super()._validate(config, graph)
        source = config.get("correlation_source", {})
        if source.get("type") != "warm_start_ensemble_cpu":
            raise ValueError(
                "WarmStartRQAOACPU requires correlation_source.type="
                "'warm_start_ensemble_cpu'."
            )
        if source.get("step_initialization", "restart") != "restart":
            raise ValueError("WarmStartRQAOACPU requires step_initialization=restart.")
        config["correlation_source"] = source

    def _run(self, graph, config, rng):
        result = super()._run(graph, config, rng)
        history = result.get("source_records") or []
        result.update(
            {
                "variant_id": "warm_start_ensemble_rqaoa_cpu_v1",
                "warm_start_history": deepcopy(result.get("source_records") or []),
                "aggregate_statevector": None,
                "aggregate_statevector_status": "not_applicable_statistical_mixture",
            }
        )
        result.setdefault("resources", {})["source_cost"] = (
            None if not history else history[0].get("source", {}).get("source_cost")
        )
        result["resources"]["source_cost_status"] = (
            None if not history else history[0].get("source", {}).get("source_cost_status")
        )
        return result


def main(argv=None):
    """Run a small warm-start RQAOA demonstration."""
    from ..tools.demo import run_solver_demo

    return run_solver_demo(WarmStartRQAOACPU, argv)


if __name__ == "__main__":
    raise SystemExit(main())
