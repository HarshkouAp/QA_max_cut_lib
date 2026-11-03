"""Quantized-angle QAOA wrapper (post-training discrete-angle recipe)."""

from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from .qaoa_cpu import QAOACPU
from .qaoa_state import cut_diagonal, state_at_angles
from .strategies import quantize_angles

__all__ = ["QuantizedQAOACPU"]


class QuantizedQAOACPU(QAOACPU):
    """Train continuous QAOA, then evaluate a quantized-angle candidate.

    The quantized state is arithmetic from the stored angles.  No extra
    measurement batch is created.  The continuous optimizer result remains
    available separately from the quantized candidate.
    """

    algorithm_type = "laboratory.quantized_qaoa_cpu"
    extra_config_fields = QAOACPU.extra_config_fields + ("angle_quantization",)
    optional_fields = QAOACPU.optional_fields + (
        "angle_quantization", "continuous_angles", "quantized_angles",
    )

    def _validate(self, config, graph):
        quantization = config.setdefault(
            "angle_quantization", {"enabled": True, "delta": math.pi / 8, "tie_rule": "nearest_ties_to_even"}
        )
        if set(quantization) - {"enabled", "delta", "tie_rule"}:
            raise ValueError("Unknown angle_quantization field.")
        if type(quantization.get("enabled", True)) is not bool:
            raise TypeError("angle_quantization.enabled must be boolean.")
        delta = float(quantization.get("delta", math.pi / 8))
        if not math.isfinite(delta) or delta <= 0:
            raise ValueError("angle_quantization.delta must be positive.")
        if quantization.get("tie_rule", "nearest_ties_to_even") != "nearest_ties_to_even":
            raise ValueError("Only ties-to-even quantization is supported.")
        super()._validate(config, graph)

    def _run(self, graph, config, rng):
        fields = super()._run(graph, config, rng)
        if fields.get("status") != "completed" or fields.get("optimal_angles") is None:
            return fields
        continuous = np.asarray(fields["optimal_angles"], dtype=float)
        quantized = quantize_angles(
            continuous,
            delta=float(config["angle_quantization"]["delta"]),
        )
        qtheta = np.asarray(quantized["quantized"], dtype=float)
        cuts = cut_diagonal(graph)
        state, probabilities = state_at_angles(cuts, qtheta, int(config["parameters"]["layers"]), dtype=config["parameters"]["dtype"])
        chosen = int(np.argmax(probabilities))
        solution = {int(node): 1 - 2 * ((chosen >> index) & 1) for index, node in enumerate(graph.nodes)}
        fields["continuous_angles"] = [float(value) for value in continuous]
        fields["quantized_angles"] = [float(value) for value in qtheta]
        fields["angle_quantization"] = quantized
        fields["solution"] = solution
        fields["solution_source"] = "quantized_argmax"
        return fields


def main(argv=None):
    from ..tools.demo import run_solver_demo
    return run_solver_demo(QuantizedQAOACPU, argv)


if __name__ == "__main__":
    raise SystemExit(main())
