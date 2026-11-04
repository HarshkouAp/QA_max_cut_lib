# Git stage 17

Version: `0.1.0a17`.

Added:

- `generators.incremental`: `single_densest_edge_count_v1` and
  `survivor_neighbor_cover_v1`;
- `solvers.incremental_ls`: `IncrementalLSRQAOACPU`, the epoch bank,
  freshness policies, four scores and a strict budget barrier;
- `laboratory.incremental_ls_rqaoa_cpu` / CLI;
- recipes `incremental_ls_keep/current/dense_* /analytic`;
- tests, notebook 18, docs and validation records.

Boundaries: checkpoint/resume of the bank, fault injection, the 48-trial
series, MK_3 batch, odds-fusion, round-max, energy prediction, Mk_8 and GPU are
not claimed. The new solver is a CPU assembly, not a literal reproduction of
all historical MK/legacy trajectories.
