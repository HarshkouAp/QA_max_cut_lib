# Laboratory decision components: reuse the tested building blocks

The Laboratory separates small decision rules from the controllers that call
them. The components below are public through
`QA_max_cut_lib.solvers.strategies`; they are tested formulas with explicit
inputs and outputs, not complete historical controllers.

Use this page when you are assembling a controller or reviewing whether a
Laboratory claim is a real implementation or a deferred backlog item.

## What the components guarantee

| Component | Input | Output and guarantee |
|---|---|---|
| `ParityForest(labels)` | active vertex labels | union-find with `parity[x]` such that `s_x = parity[x] * s_parent[x]` |
| `ParityForest.add(u, v, sign)` | edge and `sign in {-1, +1}` | `accepted`, `redundant`, or `conflict`; canonical root is the larger label |
| `odds_fusion(correlations)` | iterable of correlations or dicts with `physical_correlation`/`c` | `q = tanh(sum atanh(c))`; saturated conflicts are explicit, not NaN |
| `selected_observation(observations, weight_power=1)` | observations with `physical_correlation` | one observation by `abs(weight)**power * abs(c)`, stable ties |
| `quantize_angles(theta, delta=pi/8)` | finite angle vector | `delta * rint(theta / delta)`, ties-to-even, continuous and quantized values kept |
| `restriction_cut_proxy(graph, correlations, pair, sign)` | graph, correlations, pair, sign | read-only survivor-moment proxy; a proxy, not a certified cut |
| `legacy_history_mix(old, new, w_old, w_new)` | cuts and literal weights | signed affine statistic; `None` for a zero denominator |
| `positive_history_mix(old, new, w_old, w_new)` | cuts and weights | corrected nonnegative rule; all-zero weights give `0.0` |
| `hierarchical_proxy(...)` | parent and leaf means, parent correlation | covariance-transfer proxy with a Fréchet validity mask |
| `repeat_while_improving(callback, max_calls, ...)` | callback and a finite cap | finite replacement for a source repeat-until loop |

The components keep physical observations, decision evidence, and heuristic
proxies in separate fields. A `DecisionEvidence` value has `kind`, `value`,
`domain`, `origin`, `valid`, `reason`, and the observations that produced it.

## Parity reconciliation

```python
from QA_max_cut_lib.solvers.strategies import ParityForest

forest = ParityForest([0, 1, 2])
print(forest.add(0, 1, 1)["status"])    # accepted
print(forest.add(1, 2, -1)["status"])   # accepted
print(forest.add(0, 2, -1)["status"])   # redundant
```

A repeated pair that agrees with the path is `redundant`; the same pair with the
opposite sign is `conflict`. A conflict does not silently overwrite the forest.
The root is canonical, so proposal order does not change the component identity.

## Odds fusion and observation choice

```python
from QA_max_cut_lib.solvers.strategies import odds_fusion, selected_observation

evidence = odds_fusion([0.5, -0.25])
print(round(evidence.value, 6))          # 0.285714
print(evidence.observations["physical_mean"])  # 0.125
```

The fused value is a probability-domain decision, `(q+1)/2` is reported as
`p_plus`, and the physical mean is kept separately. A set containing both `+1`
and `-1` is a saturated conflict: `valid=False`, `reason="saturated_conflict"`,
and `value=None`. It is not reported as a numerical winner.

`selected_observation` ranks observations by weighted magnitude and breaks ties
by `round` and `task_id`. The selected entry is copied into the evidence; the
input list is not modified.

## Angle quantization

```python
from math import pi
from QA_max_cut_lib.solvers.strategies import quantize_angles

result = quantize_angles([0.1, 0.8], delta=pi / 4)
print(result["quantized"])        # [0.0, 0.7853981633974483]
print(result["tie_rule"])         # nearest_ties_to_even
```

`QuantizedQAOACPU` uses this rule after training: it keeps the continuous angles,
computes the quantized angles, evaluates the quantized state arithmetically, and
reports `solution_source="quantized_argmax"`. No extra measurement batch is
created for the quantized candidate.

## History mixing

`legacy_history_mix` reproduces the historical signed-affine statistic literally:
negative weights are used as negative numbers, and a zero denominator returns
`None` with `reason="zero_denominator"`.

`positive_history_mix` is the corrected rule: weights are clamped at zero, and a
pair of zero weights returns `0.0` with `rule="uniform_all_zero"`. The two
functions are deliberately separate so a caller cannot mistake one for the other.

## Hierarchical proxy

`hierarchical_proxy` transfers a parent correlation coefficient to leaf means.
When the parent variance is degenerate, it returns
`reason="zero_variance"`. When the raw value falls outside the Fréchet interval
for the leaf marginals, `valid=False`, `value=None`, and
`reason="frechet_violation"`; the raw value is retained for inspection.

A valid pair marginal is not a global state. The proxy is not a substitute for a
full joint distribution.

## What is wired end-to-end

- `BatchLSRQAOACPU` (`laboratory.batch_ls_rqaoa_cpu`) uses `ParityForest` and
  implements the batch contract: all local observations finish before the graph
  changes, proposals are frozen, and one of two policies is applied:
  `parity_forest` or `active_endpoints_only`.
- `QuantizedQAOACPU` (`laboratory.quantized_qaoa_cpu`) uses `quantize_angles`
  and the post-training recipe above.
- `BlockMeanFieldQAOACPU`, `DistributedLSRQAOACPU`, `IncrementalLSRQAOACPU`, and
  the MI observer classes have their own chapters and are not claims about the
  historical controllers.

`BatchLSRQAOACPU` is a compact CPU implementation, not a bitwise reproduction of
the historical `MK_3` controller.

## What is not implemented end-to-end

The following remain release blockers and are not claimed as complete legacy
controllers:

- `ConfidenceBacktrackingRQAOACPU` and a confidence-backtracking controller;
- hierarchical covariance source assembly on top of `hierarchical_proxy`;
- full odds/history adapters that consume every historical observation shape;
- the legacy fast LS-RQAOA alias family;
- a single controller that composes all of the components above.

A deferred item is visible in `catalog.json` with a backlog and is not runnable
under a historical name.

## Limitations

- The components are CPU-side and make no GPU claim.
- `ParityForest` reconciles parity, not weighted cut values; it does not rank two
  accepted proposals.
- `restriction_cut_proxy` is a survivor-moment estimate; it is not a bound.
- `quantize_angles` supports ties-to-even only.
- No component allocates a shot budget or performs a measurement.
