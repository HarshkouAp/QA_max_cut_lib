# Git stage 17

Версия: `0.1.0a17`.

Добавлены:

- `generators.incremental`: `single_densest_edge_count_v1` и
  `survivor_neighbor_cover_v1`;
- `solvers.incremental_ls`: `IncrementalLSRQAOACPU`, epoch bank,
  freshness policies, четыре score и строгий budget barrier;
- `laboratory.incremental_ls_rqaoa_cpu` / CLI;
- recipes `incremental_ls_keep/current/dense_* /analytic`;
- tests, notebook 18, docs и validation records.

Границы: checkpoint/resume банка, fault-injection, 48-trial серия,
MK_3 batch, odds-fusion, round-max, energy prediction, Mk_8 и GPU не
заявлены. Новый solver — CPU assembly, не буквальное воспроизведение
всех исторических MK/legacy траекторий.
