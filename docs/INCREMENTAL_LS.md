# Incremental LS-RQAOA

`IncrementalLSRQAOACPU` (`type="laboratory.incremental_ls_rqaoa_cpu"`) is a CPU
local-search assembly that reuses correlation observations across epochs instead of
re-measuring every pair at every step. Given a graph and a compact recipe, it returns
the original-label assignment plus the epoch, bank, and selection records that
explain every reduction. This page keeps three questions apart, because the
configuration answers them separately:

- **where new observations come from** — `refresh_all` or
  `survivor_neighbor_cover_v1`;
- **which old observations may still be used** — bank policy `current_epoch_only`
  or `keep_unobserved(max_age_steps)`;
- **how the constraint is chosen** — one of four score rules.

Reuse is the whole point of the assembly, and it is also its main risk: an old
correlation describes an older graph. The bank rules below exist to make that risk
explicit.

## One epoch

For the current graph `G_t` the assembly builds a plan of tasks, runs a full barrier,
and averages over every completed task:

\[
\widehat c^{new}_{t,e}
=\frac{\sum_a c^{(a)}_{t,e}}{d_{t,e}}.
\]

The sums are cleared once before the epoch, and `d_{t,e}` counts the tasks that
completed in it; rounds do not divide by their own denominators. A genuine zero is
kept as zero, while a pair that was never observed carries `mask=false` — the two
cases are not merged. Exactly one constraint is applied, and only after the full
barrier, so a reduction never sees a half-finished epoch.

## Bank and age

`age_steps = t - observed_epoch`. Under `keep_unobserved`, a pair survives if it was
not observed in the new epoch and `age_steps <= max_age_steps`; a new observation
replaces the old one, including when the new value is zero. A pair with a removed
endpoint is dropped.

An old `c` is never rewritten and never rescaled by the new weight. The score is
computed from the current weight and the stored correlation, so a pair whose edge
weight changed is judged by what the graph is now, not by what it was when the
observation was made.

## Score

| rule | score |
|---|---|
| `max_abs_correlation` | `abs(c)` |
| `max_abs_weighted_correlation`, `weight_power=1` | `abs(w) * abs(c)` |
| `max_abs_weighted_correlation`, `weight_power=2` | `abs(w)**2 * abs(c)` |
| `min_signed_edge_product` | `w * c`, minimized |

The constraint sign is `sign(c)`, and `zero_sign` when `c = 0`; the sign of the
weight does not multiply `sigma`. Pairs are stored in ascending label order, so the
default direction removes the smaller label and keeps the larger. A score is neither
a correlation nor a probability: it is a ranking key, and two rules can rank the same
pair list differently.

## Partial update

`survivor_neighbor_cover_v1` builds blocks around a retained vertex. Each block
contains the centre, all of its direct neighbours are covered, and the last block may
be extended by a 2-hop vertex only when the connection exists. This is a heuristic
cover. It is not a claim that quantum correlations are local, and a pair outside the
cover is simply not measured in that epoch.

## Budget

All mandatory correlation shots are reserved first; complete training iterations are
then distributed round-robin. If the mandatory part does not fit, the source is not
run: the epoch stops with `budget_exhausted` and the run returns the declared
fallback (`all_plus_reduced_fallback`) instead of an over-budget result. An exact or
analytic source has zero quantum shots but still counts CPU cost.

## Running example

The checked run uses the complete six-vertex graph from
`configs/graphs/incremental_complete_6.json`, where the 15 edges carry weights
`(2i+1)/8`, and the recipe `configs/solvers/incremental_ls_keep.json`
(`seed=17`, `max_total_shots=224`, `max_reductions=2`, `finish_at=4`).

```bash
python -m QA_max_cut_lib.laboratory.incremental_ls_rqaoa_cpu \
  --graph configs/graphs/incremental_complete_6.json \
  --config configs/solvers/incremental_ls_keep.json
```

The run reports `Cut: 18`, `Energy: -18`, `reductions_completed=2`, `epochs=2`,
three completed local tasks, and `shots_used=168` of 224 (120 training, 48
correlation). `cut_offset` is `0.0` and `ising_constant` is `-8.0`.

The result is checkable because the exact optimum of that graph is also `18`: this
instance ends at the optimum. One instance is not a guarantee — the assertion in
`tests/test_incremental_ls.py` is the one-sided
`solver.cut_value <= exact.cut_value`, which every correct heuristic satisfies.

## Limits

- No checkpoint/resume of the bank. `supports_checkpoint` is not declared for
  `laboratory.incremental_ls_rqaoa_cpu`, and a configuration that enables
  checkpointing is rejected rather than silently ignored.
- No `MK_3` batch reductions, no odds fusion, no round-max or energy-prediction
  selection, no MI-guided selection, no GPU path, no angle transfer, and no quantum
  lift.
- Old, incorrect counters from earlier versions are not reproduced; `shots_used` and
  `shots_upper_bound` describe this implementation.

## Where next

- [API reference](API.md) — recipe fields, bank policies, and result keys.
- [Distributed LS-RQAOA](DISTRIBUTED_LS.md) — the same episode idea without bank
  reuse.
- [Choosing an algorithm](CHOOSE_ALGORITHM.md) — cost and guarantees of local-search
  methods.
