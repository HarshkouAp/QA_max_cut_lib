# Проверка 0.1.0a9

Дата: 2 ноября 2026 года.

## Что проверено

- `BenchmarkSpec` и полный product: 2 графа, 28 cases, 56 scientific trials,
  2 exact references; `alpha` не появляется в mean/SA; topology/weight seeds
  сохраняются отдельно.
- `ExperimentRunner`: малая серия `small_comparison` — 56 valid, 0 failed;
  exact maximum для графов 8/10; C/energy пересчитаны по полному назначению.
- `ReferencePolicy`: exact max/min, GW heuristic, best-known source и
  отдельный `ReferenceRecord`.
- `ParallelismProfiler`: полный grid 84 case/candidate points, 84 warmups и
  252 measured calls; 1/2/4; work counters, Karp–Flatt, Amdahl fit,
  paired-block bootstrap и глобальная рекомендация.
- `concurrent_batch`: 4 независимых trial, 0 failed, process backend.
- mixed CPU/GPU: реальный `qaoa_gpu` на RTX 3050, `gpu_synchronized=true`.
- save/write-once, summary versioning, completed reuse, no-solver plot.

## Числа

```text
CPU tests: 141 passed, 1 skipped, 1 warning
GPU tests: 10 passed
small series: 56 valid / 56 planned; 2 exact references
profile: 84 points; 336 calls; recommended_num_treads=1
```

Профиль small включает QAOACPU (последовательный effective=1) и SA; поэтому
общая кривая плоская/замедляется на 2/4. Это не заявление об ускорении
последовательной реализации. SA-профиль с 1/2/4 измерен отдельно.

## Ограничения

- Concurrent profiler на одном GPU не выполнен.
- RQAOA/LS/hierarchy/GPU checkpoint в a9 не входили.
- GPU-профиль не является частью CPU-ноутбука.
