# Verification 0.1.0a7

Date: 2 November 2026. Base: `0.1.0a6`; the CPU/GPU checks of stage 06 are
preserved. The new object is `QAOAInQAOACPU` / `QAOAInQAOAGPU`.

## 1. What was verified

| Area | Check | Result |
|---|---|---|
| CPU regression | `.venv-stage02` without CuPy, `pytest -m "not gpu"` | **124 passed, 1 skipped, 1 warning** |
| Real GPU | `.venv` with CuPy 14.2.0, `pytest -m gpu` | **10 passed** |
| One level | `C_original(r*tau)=Delta b+C_meta(tau)` for all 4 tau | Matches |
| W/offset control | blocks `[10,30]`, `[80,101]` | `W=-2`, `Delta b=10`, `b=10`, `c_Ising=-9` |
| Core vs formula | N-K constraints through `GraphReducer`; weights, b, c | Matches direct computation |
| Several levels | N=8, k=2, finish_at=2 | 2 levels, 6 local calls, 6 merges |
| Restoration | all source indices, spins ±1, `C` of the source graph | Complete |
| Parallelism | num_treads=1/2, same plan and result | Matches; different worker PIDs in the parallel level |
| Readout | `best_of_shots` with real shots | 32 shots against a limit of 32; M is not multiplied by vertices |
| Budget | insufficient readout reserve | Level does not start, full fallback, 0 shots |
| Limits | `max_levels`, `max_local_tasks_total` | Full fallback with status stopped |
| Errors | invalid partitioner and backend | Error before projection |
| Tools | overview, parameters, 3D, tree | Built from the saved result |
| Notebook | `08_qaoa_in_qaoa.ipynb` with a GPU kernel | Executed top to bottom |
| Scale | N=64 and N=128, k=8, 1/2/4 threads | Complete results |

Control projection from the assignment:

```text
W(10,80) = 5 - 1 - 4 - 2 = -2
Delta b = 5 (internal) + 5 (boundary) = 10
C = 10 for equal tau; C = 8 for opposite
```

The maximum arithmetic error of the control check is at the float64 rounding
level (`< 1e-12`). This checks the formulas and the Core, not the hardware
part.

## 2. Scale

Full CPU runs, local QAOA p=1, 3 Adam steps, `most_probable`:

| N | k | threads | levels | local calls | merges | time, s | cut |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 8 | 1 | 2 | 9 | 63 | 0.12 | 55.148331 |
| 64 | 8 | 2 | 2 | 9 | 63 | 0.93 | 55.148331 |
| 64 | 8 | 4 | 2 | 9 | 63 | 0.88 | 55.148331 |
| 128 | 8 | 1 | 2 | 18 | 126 | 0.30 | 91.329203 |
| 128 | 8 | 2 | 2 | 18 | 126 | 1.78 | 91.329203 |
| 128 | 8 | 4 | 2 | 18 | 126 | 1.81 | 91.329203 |

On such small local problems the process pool on Windows can be slower than
sequential mode because of process startup. This is not declared a speedup.
The larger times are not hidden; parallel mode was verified for the same
plan/result and for real worker PIDs.

For the GPU variant on the RTX 3050 Laptop, N=8, 2 levels, local `qaoa_gpu`,
one owner were verified: `num_treads_applied=1`, `backend=cupy`, complete
restoration. N=128 on GPU is not declared verified.

## 3. Out of scope

- An updatable environment/fields, several consecutive reconciliation passes,
  overlapping blocks, CE-fixed and a shared quantum state are not
  implemented.
- `finish_at=0` does not produce a small exact remainder; an explicit finisher
  is used.
- No checkpoint/resume and no shared Benchmark/ParallelismProfiler.
- The moment-based analytic source is not a local readout.
- The GPU part does not replace the CPU check, and a CPU run does not confirm
  CUDA.
