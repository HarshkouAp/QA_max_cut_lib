# Distributed LS-RQAOA на CPU

Версия `0.1.0a15` добавляет `DistributedLSRQAOACPU`
(`laboratory.distributed_ls_rqaoa_cpu`). Один внешний шаг:

1. frozen parent graph `G_t`, offset `b_t`, messages `μ_t`;
2. несколько независимых mean-field episodes из одного `μ_t`;
3. fresh final correlations на frozen final means и final angles;
4. mean по реально наблюдавшим пару records;
5. один reducer commit полного `G_t`;
6. transport policy и reset/repartition.

## Episodes

Partition builder создаёт complete disjoint partitions текущего графа.
Каждый обычный round — отдельный episode. Anchor repair добавляет
**отдельную полную** partition, если edges coverage пусто; он не вставляет
перекрывающийся block в существующую модель.

`carry_survivor_means` переносит average episode means на оставшиеся
метки. `cold_restart` заново применяет исходный initializer. Оба варианта
сбрасывают `q=m_i m_j` и angles; это heuristic reinitialization, а не
точная conditional projection.

## Fresh extraction

После episode engine (a14) новые поля строятся по final committed means,
затем при final angles выполняется отдельная exact/shots операция.
Damped model `q` и последний update receipt не подменяют fresh `c`.

## Ограничения

- Episodes исполняются последовательно; episodes-parallelism не заявлен.
- Mid-episode nested checkpoint и fault-injection replay не заявлены.
- Нет overlapping partitions, nonlinear/adaptive полей, GPU/chi/analytic
  или global readout.
- Fresh extraction — статистическая local ZZ-оценка, не matrix одного
  общего чистого состояния.
