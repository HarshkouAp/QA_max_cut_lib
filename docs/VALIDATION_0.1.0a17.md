# Verification 0.1.0a17

Date: 2 November 2026.

## Base

- `QA_max_cut_lib 0.1.0a17`, Python 3.12.14, NumPy 2.5.4.
- CPU: `203 passed, 1 skipped, 1 warning`.
- Sphinx `-W`: success.
- Wheel installed into `.venv` and `.venv-wheelcheck`; example/CLI outside
  `src`.

## Real incremental run

Graph: complete N=6, labels `{-7,10,30,80,101,205}`, weights
`(2i+1)/8` over sorted pairs. Recipe:
`greedy_abs_weight_partition` + `survivor_neighbor_cover_v1` +
`keep_unobserved(age=2)` + `max_abs_weighted_correlation(q=1)`.

- status `completed`, reductions `2`, solution restored on the source graph;
- `4` query-bearing local tasks: 2 in epoch 0 and 2 in epoch 1;
- `224` shots: `160` training + `64` correlation;
- selected pairs of the epochs: `(10,80)` and `(30,80)`;
- final cut `18.0`, exact N=6 optimum `18.0`;
- P=1 and P=2 on the saved plan give the same solution, selection history and
  final bank; `num_treads_applied=2` at P=2.

Analytic p=1 exact Mean Cut: `0` shots, `4` source calls, `2` reductions, cut
`17.5`, without CuPy/CVXPY.

## Builders and selectors

- Star cover center 80, k=3: three memberships, all direct neighbors covered,
  filler `-7` in the last block; `k=6` gives one full task.
- Densest objective on N=4: membership does not change when all weights are
  replaced by their opposites; internal edge count `3`.
- Fixture B: max-abs → `(10,30)`, q=1 → `(10,80)`, q=2 → `(30,80)`,
  min-wc → `(80,101)`; signs `-,+,+,-`.
- Weighted+all_pairs and q outside 1/2 are rejected by validation.

## Budget

For the deterministic N=6 recipe:
- at `max_total_shots=47` and a plan with a mandatory minimum of 32 the solver
  completes the first epoch, then stops with `budget_exhausted`;
- at `max_total_shots=224` it passes two epochs in exactly 224 shots.
The exact minimum H on three tasks is verified by the arithmetic of
`_round_robin_quotas`: 48 mandatory, 40 for a full iteration.

## Not done

- checkpoint/resume of the bank, pending plan, child states and exact restore
  of a single observation;
- fault injection and unknown-charge replay;
- the 48-trial series;
- cache/performance result cache;
- MK_3 batch, odds-fusion, round-max, energy prediction, Mk_8 and GPU.

## Artifacts

- wheel: `qa_max_cut_lib-0.1.0a17-py3-none-any.whl`, SHA-256
  `2939482686e0096780d44ab8656c5cc588a165b2089cf4c2f8109bf9a1a7fc2f`;
- sdist: `qa_max_cut_lib-0.1.0a17.tar.gz`, SHA-256
  `45f54d106c4887b6e3e2d41bcbbb244153b7e34f53b0ece811516dfc827d5554`.
