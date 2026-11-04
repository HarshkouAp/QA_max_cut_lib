# Git Stage 12

Main commit: `29224e8` — `Stage 12: add warm-start ensemble RQAOA (0.1.0a12)`.

## Contents

- `solvers/warm_start_ensemble.py`: bounded `gw_rounds_cpu`/`provided_by_graph`,
  global-flip dedup, current-cut ranking, child quotas, equal-weight
  physical correlation mixture;
- `solvers/warm_start_rqaoa_cpu.py`: Laboratory assembly on the existing
  RQAOA loop;
- catalog card `warm_start_ensemble_rqaoa_cpu_v1`;
- public Laboratory module/CLI and factory route;
- GW `candidate_roundings` collection (one SDP, L logical rounds);
- provided/GW examples, benchmark spec and notebook 13;
- `WARM_START_RQAOA.md`, validation report and tests.

## Checks

```text
inventory: 89/89 parsed
CPU: 166 passed, 1 skipped, 1 warning
GPU: 10 passed (no warm GPU claim)
Sphinx -W: success
A: U=2, multiplicities 3/1, C=8, b=8, E_Ising=-7.5
G: K=3/B=368 -> [1,1,0], reserve=48
GW: one SDP, 4/4 rounds
series: 12 valid, 0 failed
```

Checkpoint/resume for `WarmStartRQAOACPU` is not claimed in a12; its
checkpoint config is rejected explicitly. Legacy WS_RQAOA is not marked
resumed or fully reproduced.
