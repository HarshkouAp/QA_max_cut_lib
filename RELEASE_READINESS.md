# Release readiness — 0.1.0rc2

Date: 3 November 2026. Status: **release candidate 2, not 0.1.0**.

## Verified

- CPU core/reduction/QAOA/RQAOA/LS/warm-start/fields/mean-field/distributed/
  incremental/MI/plans/batch/quantized tests: `218 passed, 1 skipped`.
- Wheel import without CuPy/CUDA/CVXPY/Qiskit; CLI and examples outside `src`.
- Typed plans: vertex/induced/edge domains and coverage audit.
- Benchmark297: 297/297 IDs materialized without solver calls.
- 48-case generation/plan/analysis series: 0 quantum shots.
- Sphinx `-W`.

## R01–R65 matrix

`REQUIREMENTS_TRACEABILITY.json`: **64 passed, 1 blocked**.

The only remaining blocker is **R19**: the complete transfer of every materially
different branch of the 89-source archive. `SOURCE_CLOSURE.json` contains all 89
records; 18 records are `reviewed_deferred` (GraphDecomposer_light
truss/repair/sampling, parts of MK_6/7/8, CorrelationExtractor, and related active
branches). Therefore `full_archive_migration_complete=false`.

## Release decision

`0.1.0rc2` is an honest, reviewable CPU/GPU candidate. A final `0.1.0` requires
closing or explicitly excluding the remaining 18 source records and repeating
the release gates. Labelling it `0.1.0` before that would be false.
