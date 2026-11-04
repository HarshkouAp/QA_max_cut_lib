# Release readiness — 0.1.0rc2

Дата: 3 ноября 2026 года. Статус: **release candidate 2, не 0.1.0**.

## Что проверено

- CPU core/reduction/QAOA/RQAOA/LS/warm-start/fields/mean-field/distributed/
  incremental/MI/plans/batch/quantized тесты: `218 passed, 1 skipped` (см. validation).
- Wheel import без CuPy/CUDA/CVXPY/Qiskit; CLI и examples вне `src`.
- Typed plans: vertex/induced/edge domains и coverage audit.
- Benchmark297: 297/297 IDs materialized без solver calls.
- 48-case generation/plan/analysis series: 0 quantum shots.
- Sphinx `-W`.

## Матрица R01–R65

`REQUIREMENTS_TRACEABILITY.json`: **64 passed, 1 blocked**.

Единственный оставшийся blocker — **R19**: полный перенос всех
содержательно различающихся ветвей 89-source архива. `SOURCE_CLOSURE.json`
содержит все 89 записей; 18 записей имеют `reviewed_deferred`
(GraphDecomposer_light truss/repair/sampling, часть MK_6/7/8,
CorrelationExtractor и связанные active branches). Поэтому
`full_archive_migration_complete=false`.

## Release decision

`0.1.0rc2` — честный reviewable CPU/GPU candidate. Финальный `0.1.0`
требует закрыть или явно исключить оставшиеся 18 source records и
повторить release gates. Маркировка `0.1.0` без этого была бы ложной.
