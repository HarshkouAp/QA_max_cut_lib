# Release readiness — 0.1.0rc2

Date: 4 November 2026. Status: **release candidate with all 65 requirements passed**.

## Verified

- CPU core/reduction/QAOA/RQAOA/LS/warm-start/fields/mean-field/distributed/
  incremental/MI/plans/batch/quantized/legacy tests: `233 passed, 1 skipped`.
- Wheel import without CuPy/CUDA/CVXPY/Qiskit; CLI and examples outside `src`.
- Typed plans: vertex/induced/edge domains and coverage audit.
- Benchmark297: 297/297 IDs materialized without solver calls.
- 48-case generation/plan/analysis series: 0 quantum shots.
- Sphinx `-W`; documentation, notebooks, and previews are English-only.

## R01–R65 matrix

`REQUIREMENTS_TRACEABILITY.json`: **65 passed, 0 blocked**.

**R19 is closed.** `SOURCE_CLOSURE.json` records all 89 archive members with
terminal dispositions. The 18 formerly deferred branches are reachable through
`QA_max_cut_lib.laboratory.legacy` compatibility adapters:

- `MK6V1`, `MK6V2`, `MK7V1`, `MK7V2` — batch LS variants with parity and
  active-endpoint reconciliation;
- `LSRQAOAExp`, `LSRQAOAFast1`–`Fast4` — LS recipes with distinct builders and
  block schedules;
- `QAOAAdam`, `QAOAF2`, `QAOAFM` — QAOA objective/optimizer recipes;
- `CorrectedMeanMap`, `CorrelationExtractor`, `EdgeExtender`,
  `GraphDecomposerLight`, `GraphScanner`, `partition_smoke_test` — diagnostic
  and decomposition components.

The adapters delegate to the shared `LSRQAOACPU`, `BatchLSRQAOACPU`, and
`QAOACPU` controllers; they do not import or execute the legacy archive at
runtime. `tests/test_legacy.py` exercises every adapter. Exact source names and
source-to-component mappings are in `SOURCE_CLOSURE.json`.

## Release decision

All scientific and documentation requirements are closed. `0.1.0rc2` can be
promoted to a stable `0.1.0` release after the packaging choices (license, CI,
and optionally PyPI) are confirmed. The code no longer has an R19 blocker.
