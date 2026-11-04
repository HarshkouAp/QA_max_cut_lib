# Acceptance evidence archive

This page indexes the historical acceptance records of the project. Each record is
a snapshot of one build on one machine; none of them describes the current release
by itself. The current position is in `RELEASE_READINESS.md`, the current counts
are in `VALIDATION_0.1.0rc2.md`, and the per-file source ledger is
`SOURCE_CLOSURE.json`.

## How to read the archive

- A record states what was run, on which build, with which result.
- A test count is valid for that build only.
- A missing check is listed as missing; it is not implied to pass.
- A later build does not retroactively validate an earlier record, and an earlier
  record does not certify a later build.

## Language

All public pages, stage records, git summaries, and the architecture reference are
in English. Any historical command, hash, or count remains verbatim; translation
did not change the evidence.

## First build and release candidates

| Record | Date | Build | CPU result | GPU result | Scope |
|---|---|---|---|---|---|
| `VALIDATION.md` (historical section) | 1 November 2026 | `0.1.0a1` | 42 checks | not run | first working CPU stage, installed wheel, two tutorial notebooks |
| `VALIDATION_0.1.0rc1.md` | 3 November 2026 | `0.1.0rc1` | 216 passed, 1 skipped, 1 warning | not rechecked | first release candidate; artifacts and known gaps |
| `VALIDATION_0.1.0rc2.md` | 3 November 2026 | `0.1.0rc2` | 218 passed, 1 skipped, 1 warning | 10 passed | current candidate; batch LS, quantized QAOA, R19 open |

## Stage records `0.1.0a2`–`0.1.0a18`

| Record | Build | CPU result | GPU result | Main addition |
|---|---|---|---|---|
| `VALIDATION_0.1.0a2.md` | `0.1.0a2` | 56 passed, 0 failed | not run | exact pair reduction Max-Cut to Ising |
| `VALIDATION_0.1.0a3.md` | `0.1.0a3` | 80 passed, 1 warning | not run | base RQAOA on CPU |
| `VALIDATION_0.1.0a4.md` | `0.1.0a4` | 92 passed, 1 warning | not run | analytic `p=1` source and RQAOA connection |
| `VALIDATION_0.1.0a5.md` | `0.1.0a5` | 109 passed, 1 warning | not run | LS-RQAOA CPU with process execution |
| `VALIDATION_0.1.0a6.md` | `0.1.0a6` | 109 passed, 1 skipped, 1 warning | 9 passed | NVIDIA GPU QAOA, sampler, five objectives, GPU source |
| `VALIDATION_0.1.0a7.md` | `0.1.0a7` | 124 passed, 1 skipped, 1 warning | 10 passed | hierarchical QAOA-in-QAOA |
| `VALIDATION_0.1.0a8.md` | `0.1.0a8` | 134 passed, 1 skipped, 1 warning | 10 passed | GW, SA, exact block refinement |
| `VALIDATION_0.1.0a9.md` | `0.1.0a9` | 141 passed, 1 skipped, 1 warning | 10 passed | series, references, measurement, parallelism profiler |
| `VALIDATION_0.1.0a10.md` | `0.1.0a10` | 151 passed, 1 skipped, 1 warning | 10 passed | `QAOACPU` checkpoints and resume |
| `VALIDATION_0.1.0a11.md` | `0.1.0a11` | 161 passed, 1 skipped, 1 warning | 10 passed | Laboratory catalog and warm-start CPU |
| `VALIDATION_0.1.0a12.md` | `0.1.0a12` | 166 passed, 1 skipped, 1 warning | 10 passed | warm-start RQAOA ensemble |
| `VALIDATION_0.1.0a13.md` | `0.1.0a13` | 173 passed, 1 skipped, 1 warning | not rechecked | conditional fields and environment LS |
| `VALIDATION_0.1.0a14.md` | `0.1.0a14` | 184 passed, 1 skipped, 1 warning | not rechecked | block mean-field exchange |
| `VALIDATION_0.1.0a15.md` | `0.1.0a15` | 189 passed, 1 skipped, 1 warning | not rechecked | distributed LS-RQAOA |
| `VALIDATION_0.1.0a16.md` | `0.1.0a16` | 198 passed, 1 skipped, 1 warning | not rechecked | mutual-information diagnostics |
| `VALIDATION_0.1.0a17.md` | `0.1.0a17` | 203 passed, 1 skipped, 1 warning | not rechecked | incremental LS and MK components |
| `VALIDATION_0.1.0a18.md` | `0.1.0a18` | 211 passed, 1 skipped, 1 warning | not rechecked | generators, structured weights, typed plans |

`VALIDATION_LOCAL.md` is a separate Windows 10 check of `0.1.0a2`: 56 CPU tests,
both notebook runs, wheel build, clean install, Sphinx `-W`, and the 89/89 source
hash comparison. It is not a duplicate of the stage record; it verifies the local
environment.

## Git summaries

| Record | Commit or base | Scope |
|---|---|---|
| `GIT_STAGE02.md` | base `8dc86ed`; commits `5d7cd11`, `457f20e` | environment, documentation, stage 02 report and patch |
| `GIT_STAGE04.md` | `bd62460bd17e0d99ffd1a08907a9e928dfec2616` | analytic `p=1` correlations and RQAOA source (`0.1.0a4`) |
| `GIT_STAGE05.md` | `ec4d0973312a1204c02dd1192a8d44a0d01a9618` | LS-RQAOA CPU with process execution (`0.1.0a5`) |
| `GIT_STAGE07.md` | `1ebfd75` | hierarchical QAOA-in-QAOA (`0.1.0a7`), including uncommitted stage 06 files |
| `GIT_STAGE08.md` | `d31cf13` | classical baselines and exact block refinement (`0.1.0a8`) |
| `GIT_STAGE09.md` | stage 09 and 10 combined | series and measurement layer (`0.1.0a9`) |
| `GIT_STAGE10.md` | `eef2fa4` | checkpoints and resume (`0.1.0a10`) |
| `GIT_STAGE11.md` | `e08a343` | Laboratory catalog and warm start (`0.1.0a11`) |
| `GIT_STAGE12.md` | `29224e8` | warm-start RQAOA ensemble (`0.1.0a12`) |
| `GIT_STAGE13.md` | `e1958a2` | conditional fields and environment LS |
| `GIT_STAGE14.md` | `368b6bb` | block mean-field QAOA |
| `GIT_STAGE15.md` | `c048016` | distributed LS-RQAOA |
| `GIT_STAGE16.md` | not recorded in the file | mutual-information diagnostics |
| `GIT_STAGE17.md` | not recorded in the file | incremental LS and MK components |
| `GIT_STAGE18.md` | not recorded in the file | generators, structured weights, typed plans |

## Measurement and environment files

| File | Content |
|---|---|
| `PERFORMANCE_ANALYTIC_P1_0.1.0a4.json`, `.txt` | analytic `p=1` cases and timings |
| `PERFORMANCE_LS_0.1.0a5.json` | LS-RQAOA graph, builder, rounds, and gamma points |
| `SCALE_LS_0.1.0a5.json` | LS scaling case: vertices, edges, timings, reductions |
| `PERFORMANCE_GPU_0.1.0a6.json` | GPU device, driver, memory, and timings |
| `PERFORMANCE_HIERARCHY_0.1.0a7.json` | hierarchy runs at 1, 2, and 4 threads |
| `PERFORMANCE_CLASSICAL_0.1.0a8.json` | exact, GW, SA, and refiner cases |
| `validation_environment*.json` | machine-readable package versions for the recorded builds |
| `STAGE02_DIFF.patch` | stage 02 source patch |
| `01_quickstart_execution.json`, `02_current_api_guide_execution.json` | notebook cell counts, images, errors, and runtimes |

## Rules for a new record

A new acceptance record states the date, the build, the exact command, the observed
count, the environment, and the limits of the check. Every number comes from a real
run; an analytical value is labelled as expected. A record that cannot reproduce a
number says so instead of copying the previous record.
