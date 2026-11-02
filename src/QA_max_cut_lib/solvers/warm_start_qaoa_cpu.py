"""First Laboratory transfer: CPU warm-start product QAOA."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from .qaoa_cpu import QAOACPU
from .warm_start import (
    WarmStartError,
    prepare_warm_start,
    state_at_warm_angles,
    validate_warm_start_section,
)

__all__ = ["WarmStartQAOACPU"]


class WarmStartQAOACPU(QAOACPU):
    """Run full-state QAOA from an explicit clipped product state.

    Parameters
    ----------
    config : dict, optional
        A normal QAOA CPU configuration plus a required ``warm_start``
        section.  The section contains ``epsilon``, ``source``, and
        ``mixer``.  Source data are validated against the current graph
        before any expensive work.

    Attributes
    ----------
    warm_start_data : dict or None
        The frozen source, clipped probabilities, preparation angles, and
        initial statistics used by this run.
    variant_id : str
        Stable Laboratory variant ID.

    Notes
    -----
    This is a CPU full-state implementation.  It does not use the
    symmetric chi GPU path or the analytic p=1 evaluator.  The preparation
    angles ``alpha`` are not trainable parameters.  A normal ``solve()``
    is always a new run; only ``restore(...).resume()`` continues a saved
    run.
    """

    algorithm_type = "laboratory.warm_start_qaoa_cpu"
    checkpoint_component_id = "laboratory.warm_start_qaoa_cpu"
    extra_config_fields = QAOACPU.extra_config_fields + ("warm_start",)
    optional_fields = QAOACPU.optional_fields + (
        "variant_id",
        "warm_start_data",
        "mixer_id",
        "source_type",
        "source_cut",
        "initial_expected_cut",
        "initial_expected_energy",
        "source_run",
        "source_cost",
        "source_cost_status",
    )

    def __init__(self, config=None):
        self._warm_data = None
        super().__init__(config)

    def _validate(self, config, graph):
        super()._validate(config, graph)
        if "warm_start" not in config:
            raise WarmStartError("Laboratory warm-start requires a warm_start section.")
        config["warm_start"] = validate_warm_start_section(config["warm_start"], graph)

    def _run(self, graph, config, rng):
        resume_state = deepcopy(self._resume_state) if self._resume_mode else None
        if resume_state is not None:
            saved = resume_state.get("warm_start_data")
            if not isinstance(saved, dict):
                raise WarmStartError(
                    "A warm-start checkpoint has no frozen warm_start_data."
                )
            self._warm_data = deepcopy(saved)
        else:
            self._warm_data = prepare_warm_start(
                graph, config["warm_start"], seed=config.get("seed")
            )
        result = super()._run(graph, config, rng)
        data = self._warm_data
        result.update(
            {
                "variant_id": "warm_start_product_qaoa_cpu_v1",
                "warm_start_data": deepcopy(data),
                "mixer_id": data["mixer"],
                "source_type": data["source_type"],
                "source_cut": data["source_cut"],
                "initial_expected_cut": data["initial_expected_cut"],
                "initial_expected_energy": data["initial_expected_energy"],
                "source_run": deepcopy(data.get("source_run")),
                "source_cost": deepcopy(data.get("source_cost")),
                "source_cost_status": data.get("source_cost_status"),
            }
        )
        result.setdefault("resources", {})
        result["resources"]["source_cost"] = deepcopy(data.get("source_cost"))
        result["resources"]["warm_state_build"] = 0
        return result

    def _state_at_angles(self, cuts, angles, layers, dtype):
        if self._warm_data is None:
            return super()._state_at_angles(cuts, angles, layers, dtype)
        return state_at_warm_angles(
            cuts,
            angles,
            layers,
            self._warm_data["q_clipped"],
            self._warm_data["mixer"],
            dtype,
        )

    def _checkpoint_extra(self):
        if self._warm_data is None:
            return {"warm_start_data": None, "variant_id": "warm_start_product_qaoa_cpu_v1"}
        return {
            "warm_start_data": deepcopy(self._warm_data),
            "variant_id": "warm_start_product_qaoa_cpu_v1",
        }

    def resume(self, *, control=None):
        """Continue a restored warm-start run at a declared QAOA boundary."""
        return super().resume(control=control)


def main(argv=None):
    """Run a small warm-start demonstration through the common CLI."""
    from ..tools.demo import run_solver_demo

    return run_solver_demo(WarmStartQAOACPU, argv)


if __name__ == "__main__":
    raise SystemExit(main())
