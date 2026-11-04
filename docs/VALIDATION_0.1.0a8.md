# Verification 0.1.0a8

Date: 2 November 2026. Base: `0.1.0a7`; previous CPU/GPU checks are
preserved. New dependencies were verified in the real environment:
CVXPY `1.9.3`, SCS `3.3.1`, dimod `0.12.23`, `dwave-samplers 1.8.0`.

## 1. Tests

| Area | Command | Result |
|---|---|---|
| CPU regression | `.venv-stage02/Scripts/python.exe -m pytest -q -m "not gpu"` | **134 passed, 1 skipped, 1 warning** |
| GPU regression | `.venv/Scripts/python.exe -m pytest -q -m gpu` | **10 passed** |
| GW | real CVXPY/SCS, triangle and signed/small graphs | OK |
| SA | real `dwave.samplers.SimulatedAnnealingSampler` | OK |
| Refiner | D/E/F/G and the state limit | OK |
| RefinedSolver | ExactCPU, QAOACPU, RQAOACPU base | OK |
| Import without extras | `.venv-stage02` without CuPy, classical dependencies not imported at import time | OK |
| Documentation | `sphinx -W` | OK |
| Notebook | `09_classical_baselines_and_refinement.ipynb` | Executed |

## 2. Control examples

### A/B: sign of the BQM

For edge `(10,30,2)`:
`h=0`, `J=+1`, `offset=-1`; opposite spins give `C=2`, `E=-2`.
For `(10,30,-2)`: `J=-1`, `offset=+1`; equal spins give `C=0`, `E=0`.
The isolated node `80` is present in every complete dictionary.

### C: GW triangle

Three edges of weight 1. Actual SCS result:
`sdp_objective_raw=2.25000000088`, `sdp_status=optimal`, minimum eigenvalue
`-1.17e-09`, diagonal residual `2.09e-12`.
The rounded C need not be 2 for every seed; the check does not require it.
`sdp_upper_bound=null`; the reason is the absence of a strict dual certificate.

### D: conditional block

For `S={10,30}`, `t_80=+1`:
`h_10=2`, `h_30=1.5`, `c_S=-4`, conditional optimum `x=(-1,-1)`, final
`C=7`. Fixing one internal spin is not used.

### E: no internal edges

`S={10}`, external edge of weight 3, `t_30=+1`: field `h_10=1.5`, the flip
gives `C=3`, and the block is not skipped as "a block without internal edges".

### F: sequential environment

Blocks `[10]` and `[30]` under the previous environment each give `gain=1`;
the sequential refiner accepts the first correction, recomputes the field and
rejects the second: `C=1`. Simultaneous application would give `C=-4`; that
mode is not implemented.

### G/H: non-degradation and limits

- A refiner with an already optimal input returns an equal complete
  dictionary.
- `refinement_states_total=4` with a first block of `2^3` does not start the
  block, returns the original assignment and
  `stop_reason=refinement_states_total`.
- `refinement_states_total=10` executes the first block of 8 states, does not
  start the second block of 4 and honestly reports `states_evaluated=8`.
- SA with `num_reads=10`, `num_sweeps=64`, `anneal_sweeps_total=192`
  performs exactly 3 complete reads; there is no fractional anneal.

## 3. Real measurements

Full data: `docs/PERFORMANCE_CLASSICAL_0.1.0a8.json`.

| Method | N | Time, s | C | Note |
|---|---:|---:|---:|---|
| ExactCPU | 7 | — | 6 | control graph |
| GW | 32 | 1.22 | 24.886145 | SCS |
| GW | 64 | 0.47 | 84.962514 | SCS |
| SA, threads=1 | 128 | 0.447 | 167.125300 | 8 reads × 32 sweeps |
| SA, threads=2 | 128 | 0.025 | 167.125300 | same seed plan |
| SA, threads=4 | 128 | 0.028 | 167.125300 | same seed plan |
| SA | 512 | 0.132 | 795.746057 | 4 reads × 16 sweeps |
| Refiner | 128 | 0.027 | 48.734279 | all-plus input, 4 blocks |
| Refiner | 512 | 0.169 | 39.280090 | all-plus input, 4 blocks |

Times include dependency start-up and a full public solve. The difference
between 1/2/4 threads for SA is not declared a speedup: small reads and pool
start-up may dominate. A GW cut is not a proven optimum. For N=7 ExactCPU,
GW and SA on one control graph gave C=6; this is a small check of the
convention, not a quality guarantee on arbitrary signed graphs.

## 4. Limitations

- No strict SDP dual certificate is built; `sdp_upper_bound=null`.
- `optimal_inaccurate` is rejected by default; allowing it requires an
  explicit `allow_inaccurate=true`.
- CUDA/QPU are not used in the new methods. RefinedSolver can accept a GPU
  base, but that is a separate composition and does not confirm classical GPU
  work.
- Full Benchmark/ParallelismProfiler and checkpoint/resume are not
  implemented.
- The refiner enumerates blocks sequentially; simultaneous addition of local
  gains is forbidden.
