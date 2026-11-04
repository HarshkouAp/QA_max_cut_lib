# Verification 0.1.0a15

Date: 2 November 2026.

## Base

- `QA_max_cut_lib 0.1.0a15`, Python 3.12.14, NumPy 2.5.4.
- CPU: `189 passed, 1 skipped, 1 warning`.
- Sphinx `-W`: success.
- Wheel installed into `.venv` and `.venv-wheelcheck`; examples/CLI run
  outside the `src` tree.
- Notebook 16 was executed by the installed kernel.

## C — fresh final correlation

Two nodes 10,30, edge of weight 2; one block; `max_sweeps=0`, exact modes,
θ=(-π/4, π/8). Fresh extraction gives `c=-1.0`, the reducer selects
`sigma=-1`; after the lift the full cut equals 2. The damped/model q was not
used for the selection.

## Main six-node run

Greedy partition max_size=3, one episode per parent step, `finish_at=4`,
`max_reductions=2`:

```text
reductions=2
solution_source=exact_finisher
status=completed
final full cut recomputed on the source graph
```

The fresh tables contain episode/block/pairs/mode/shots and real c values.

## Transport / reducer

For fixture A after removing 80 and retaining 30 with `sigma=-1`:

```text
new edge (10,30) = -1
b = 8
carried mean retained labels = (0.2,-0.4,0.6)
product q(10,30) = -0.08
```

`needs_repartition=True`; the old block q/θ are not transported.

## Checkpoint

The outer codec stores the reduction state, parent means, budget, completed
episodes/history, `logical_updates` and phase. `restore()` + `resume()` were
verified on C and on the six-node run. A mid-episode nested codec, fault
injection and a replay receipt are **not claimed** in this version.

## Not run

- the 16-trial series, real parallel episode execution, the P=1/2/4 profile;
- multi-round / all_pairs production cases;
- strict require_full/anchor production sizes;
- crash injection I and cross-process restore;
- GPU/chi/analytic/warm-start/overlap/nonlinear fields.
