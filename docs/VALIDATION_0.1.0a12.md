# Verification 0.1.0a12

Date: 2 November 2026.

## Base

- `QA_max_cut_lib 0.1.0a12`, Python 3.12.14, NumPy 2.5.4.
- CPU: `166 passed, 1 skipped, 1 warning`.
- GPU: `10 passed` (old GPU profile; GPU warm start is not claimed).
- Sphinx `-W`: success after the documentation update.
- Wheel/sdist `0.1.0a12` installed into `.venv` and `.venv-wheelcheck`.

## Catalog

```text
archive SHA (assignment): b4c684a5d4da469adc25bd99a405e402892d1a2524465e4c97ab6229c71a4aed
Python members: 89
variant cards: 7
new card: warm_start_ensemble_rqaoa_cpu_v1
relation: component_recomposition
```

The RAR container is absent from the working tree; the inventory was built
from the extracted `reference/RQAOA_main`. All 89 member hashes are recorded.

## A — two starts and a negative edge

```text
nodes 10,30,80,101; edges 2,3,5
A=(+,+,−,+), B=(+,−,+,+)
raw batch: A, −A, B, A2
U=2; multiplicities A=3, B=1
C(A)=8, C(B)=7
selected c: (10,30)=0, (10,80)=0, (30,80)=-1
selected pair (30,80), sigma=-1
C_final=8, b=8, E_Ising=-7.5
reductions=1, finisher=exact_finisher
```

The check was performed on a real `WarmStartRQAOACPU` with a provided batch,
without CVXPY.

## B/C/D/E

- ranking on the current graph uses the recomputed `C_t`, not the old weight;
- global-flip dedup: A/−A/A2 give one key, multiplicity does not change the
  weight;
- mixture A: `P=(1/4,1/4,1/4,1/4)`, `c=0`; the coherent sum gives
  `P=(1/2,0,0,1/2)`, `c=1` — erroneous coherent aggregation is caught;
- nonlinear comparison: mean cut and upper-tail CVaR of the components are not
  replaced by the mixture criterion;
- the endpoint K=1 is verified as a special case.

## GW source

A real `gw_rounds_cpu` was run on a 4-vertex graph:

```text
sdp_calls=1
rounds_requested=4
rounds_completed=4
source_cost.hyperplanes=4
sdp_status=optimal
```

This is one SDP, not K independent calls. The series used 8 roundings.

## G/H — budget and series

```text
K=3, B=368, p=1, M_train=32, M_corr=16:
iterations=[1,1,0], reserve=48, quotas=[176,176,16]
K=2: iterations=[1,1], reserve=32
B=47: quantum child does not start
B=48: iterations=[0,0,0], correlations only
```

Small series `experiments/benchmark_warm_start_rqaoa.json`:

```text
2 graphs × 2 seeds × 3 recipes = 12 trials
rqaoa_cpu, ws_rqaoa_k1, ws_rqaoa_k2
finish_at=2, p=1, max_steps=3, M_train=M_corr=32
planned=12 valid=12 failed=0
```

Actual `wall=0.384 s` in the current environment; `rqaoa_cpu` and K=1/K=2
spend the budget differently, so this is not a fixed-total-budget comparison.

## Integration

- `WarmStartRQAOACPU` uses the existing `GraphReducer`,
  `MaxAbsCorrelationSelector`, `ExactCPUFinisher`, `ShotBudget` and QAOA
  trainer; there is no second ADAM.
- `provided_by_graph` works without CVXPY/CUDA.
- The GW source requires the classical profile.
- `aggregate_statevector=null`; a physical mixture is not passed off as a pure
  statevector.
- `source_cost` of classical preprocessing is kept separate from quantum
  shots.

## Limitations

- Checkpoint/resume for `WarmStartRQAOACPU` is **not implemented** in a12;
  `supports_checkpoint` is not claimed, a config with checkpoint is rejected.
- No adaptive weights, no transfer of starts between reductions, no SA pools,
  no fractional ensemble members, no GPU warm start.
- No `provided_by_graph` after an arbitrary reduction without an explicit
  record of the new snapshot; a missing record is an error.
- An SDP primal value is not a certificate of optimality.
