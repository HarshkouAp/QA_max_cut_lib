# Laboratory decision components (release candidate)

This is a release-candidate slice for the remaining Laboratory strategies.
It implements tested formula components, not every historical controller.

## Implemented

- `ParityForest`: parity union-find with accepted/conflict/redundant
  proposal events and canonical roots.
- `odds_fusion`: `q=tanh(sum atanh(c))`, stable conflict handling, physical
  mean kept separately.
- `selected_observation`: weighted-magnitude round-max decision.
- `quantize_angles`: `Δ*rint(theta/Δ)`, ties-to-even, continuous and
  quantized angles retained.
- `restriction_cut_proxy`: read-only survivor-moment proxy.
- `legacy_history_mix` / `positive_history_mix`: explicitly separate
  signed-affine statistic and corrected positive-weight rule.
- `hierarchical_proxy`: covariance transfer with Fréchet validity mask,
  raw and unavailable projected values.
- `repeat_while_improving`: finite replacement for source repeat-until.

## Not implemented end-to-end

`BatchLSRQAOACPU`, `ConfidenceBacktrackingRQAOACPU`, hierarchical
covariance source assembly, quantized solver wrappers and full odds/
history adapters remain release blockers. The components above are not
claimed as complete legacy controllers.
