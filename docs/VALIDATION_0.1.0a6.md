# Verification 0.1.0a6

Date: 2 November 2026. Base: `0.1.0a5` (stage 05 completed and verified
separately). Stage goal: NVIDIA GPU QAOA, the sampler, five criteria and
connection of the GPU source to RQAOA/LS.

## 1. What was verified

| Area | Check | Result |
|---|---|---|
| CPU regression | `.venv-stage02` without CuPy, `pytest -m "not gpu"` | **109 passed, 1 skipped, 1 warning** |
| Real GPU | `.venv` with CuPy 14.2.0, `pytest -m gpu` | **9 passed** |
| Full GPU | `psi`, normalisation, probabilities and ZZ correlations at p=1,2,3 | Matches CPU |
| Pair GPU | `chi`, restoration of `psi`, `q`, N=1 and the last mixer | Matches CPU |
| Five criteria | Mean Cut, CVaR with a partial tail, Gibbs, Best-of-K, Threshold | Matches the float64 reference |
| Sampler | 20 000 physical shots, both orientations of a pair | Frequencies converge to `q/2` |
| Adjoint | exact Mean Cut against central differences, p=2 | Matches |
| Readout | `most_probable`, `best_of_shots`, budget | Real shots; complete solution |
| RQAOA/LS | GPU source, shared reducer, `gpu_owner` | Small complete solutions |
| Memory | preflight of an impossible full state | `MemoryError` before a heavy allocation |
| Notebook | `07_nvidia_gpu_qaoa.ipynb` from a clean GPU kernel | Executed top to bottom |

Maximum deviations on small problems (`complex128`, p=1,2,3):

| Quantity | Maximum |
|---|---:|
| full state, abs | `2.01e-16` |
| pair restored state, abs | `1.67e-16` |
| five criteria, abs | `4.44e-16` |
| ZZ correlation matrix, abs | `2.78e-16` |
| adjoint against central difference, abs | `1.64e-10` |

Sampler error at 20 000 shots: maximum `4.21e-3` in physical mass (a
statistical check, not bitwise RNG agreement).

## 2. Device and environments

| Field | Actual value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| Compute capability | 8.6 |
| VRAM | 4 294 508 544 bytes (about 3.46 GB free at the start of the tests) |
| Process CUDA runtime | 13020 |
| Driver | 13010 (591.74) |
| CuPy | 14.2.0 |
| Python | 3.12.14 |
| CPU environment | `.venv-stage02`, CuPy absent |

The RTX 5060 Laptop from the assignment reference was not available. V100 and
the CUDA 12 profile were not verified. Those rows are not declared as
verified on hardware.

## 3. Acceptance commands

```bash
# CPU regression without CuPy
.venv-stage02/Scripts/python.exe -m pytest -q -m "not gpu"

# Explicit GPU acceptance: without CuPy/device the test fails
# rather than succeeding with zero GPU checks executed
QA_MAX_CUT_REQUIRE_GPU=1 .venv/Scripts/python.exe -m pytest -q -m gpu

# Notebook
.venv-stage02/Scripts/jupyter nbconvert --to notebook --execute \
  notebooks/07_nvidia_gpu_qaoa.ipynb \
  --ExecutePreprocessor.kernel_name=qa-max-cut-lib-gpu
```

`-m gpu` uses `pytest.importorskip` in normal mode for CPU CI. The variable
`QA_MAX_CUT_REQUIRE_GPU=1` disables the skip: a missing CuPy or device is an
error.

## 4. Composite examples

- `QAOAGPU`: pair, complex64, exact Mean Cut, 2 ADAM steps, correlations;
  `shots_used=0`; complete solution.
- `RQAOAGPU`: 2 reductions, shared `GraphReducer`, exact finisher, complete
  solution.
- `LSRQAOAGPU`: `backend="gpu_owner"`, 2 threads requested, in fact 1 service
  sequence; complete solution.
- `best_of_shots`: 576 actual shots against a limit of 2000; correlations and
  training do not charge the final readout twice.

## 5. Performance

`complex64` measurements, p=2, full evolution after warm-up:

| N | full | pairs |
|---:|---:|---:|
| 16 | 8.7 ms | 6.8 ms |
| 20 | 50.6 ms | 25.6 ms |
| 24 | 906 ms | 448 ms |

The first allocation and kernel compilation can take hundreds of
milliseconds and are not mixed with warmed time. This is a kernel-launch
profile, not a promise of end-to-end speedup. N=24 is the largest stably
verified size for both representations in `complex64` on 4 GiB. N=26 is not
declared a supported profile.

## 6. Limitations and honest status

- CUDA Graphs, Walsh tiles, batch>1, multiple GPUs and distributed execution
  are not implemented.
- Compute Sanitizer was not run; the absence of races is supported only by
  mathematical/numerical checks and the absence of test failures.
- Pair mode does not apply to problems with fields, another initial state or
  another mixer.
- The full `psi`/`probabilities` is not assembled automatically in pair mode;
  `chi` is not passed off as a physical `psi`.
- The GPU path does not fall back to CPU on OOM or when CuPy is missing.
