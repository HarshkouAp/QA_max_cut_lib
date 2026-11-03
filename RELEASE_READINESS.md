# Release readiness — 0.1.0rc2

Дата: 3 ноября 2026 года. Статус: **release candidate 2, не 0.1.0**.

## Что проверено

- CPU core/reduction/QAOA/RQAOA/LS/warm-start/fields/mean-field/distributed/
  incremental/MI/plans/batch/quantized тесты: `217 passed, 1 skipped` (см. validation).
- Wheel import без CuPy/CUDA/CVXPY/Qiskit; CLI и examples вне `src`.
- Typed plans: vertex/induced/edge domains и coverage audit.
- Benchmark297: 297/297 IDs materialized без solver calls.
- 48-case generation/plan/analysis series: 0 quantum shots.
- Sphinx `-W`.

## Blockers для 0.1.0

1. **Stage 19 controllers:** `BatchLSRQAOACPU` и
   `QuantizedQAOACPU` добавлены как реальные CPU assemblies;
   `ConfidenceBacktrackingRQAOACPU`, hierarchical covariance source и
   полный odds/history end-to-end adapter всё ещё не закрыты.
   `DecisionEvidence`, parity forest, odds, selected-observation,
   restriction proxy, quantization и non-mutating history formulas
   реализованы как проверяемые компоненты.
2. **89-source closure:** в registry сохранены исходные записи, но
   содержательно отличающиеся GraphDecomposer_light/legacy branches
   остаются unresolved; aggregate `full_archive_migration_complete`
   не заявлен.
3. **G8 NVIDIA:** текущий source GPU test set `10 passed` на локальном
   RTX 3050; полный installed-wheel GPU profile ещё не является
   отдельным release gate evidence.
4. **R01–R65:** полная матрица evidence заполнена только для
   реализованных срезов; несвязанные записи имеют blocker, а не
   фиктивный pass.
5. **Два главных notebooks:** 00/01 исполнены для CPU-среза; полный
   controller coverage остаётся ограниченным.

## Release decision

Версия **0.1.0rc1** может использоваться для проверки CPU-среза и
подготовки reviewable RC. Она не должна маркироваться как финальный
`0.1.0`, пока blockers выше не закрыты или явно не исключены владельцем
проекта.
