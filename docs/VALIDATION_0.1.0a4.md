# Build verification 0.1.0a4

**Version:** `QA_max_cut_lib 0.1.0a4`, 2 November 2026.  
**Base:** local build after stage 03 (`0.1.0a3`).  
**Purpose:** analytic moments of weighted QAOA p=1, exact beta minimisation
and integration with the existing `RQAOACPU`.

## 1. What was added

- `AnalyticP1Evaluator`: mean cut, mean energy, coefficients `E0,A,B`,
  physical ZZ correlations and the full correlation matrix.
- Source `qaoa_analytic_p1_cpu` with `edges`/`all_pairs` coverage, a finite
  gamma grid, exact beta and a `min_beta E(gamma,beta)` profile.
- Integration into `RQAOA` through the existing `type="rqaoa_cpu"`; the
  recursive coordinator, exact reducer and exact finisher were not replaced.
- The profile is stored in `local_optimizer_history` with the explicit type
  `analytic_gamma_grid`; `shots_used=0` and `limits.shots_total=0` are
  allowed.
- Configuration `configs/solvers/rqaoa_analytic_p1_cpu.json`.
- Notebook `notebooks/05_analytic_p1_rqaoa.ipynb`; notebooks 01 and 02 were
  updated.
- Time/memory profile `docs/PERFORMANCE_ANALYTIC_P1_0.1.0a4.json`.

## 2. Mathematical checks

The checks use independent matrix evolution / dense statevector, not the
analytic functions. Summary from `scripts/verify_analytic_p1.py`:

| Check | Result |
|---|---:|
| "Graph × angles" cases for N=2…8 | 100 |
| Maximum correlation error | `8.88e-16` |
| Maximum mean-energy error | `1.78e-15` |
| Maximum excess of the dense beta grid over the exact minimum | `1.15e-05` (2001-point grid) |

Control examples:

- one edge at `gamma=pi/(2|w|)`, `beta=-pi/8`: `c=-sign(w)`; for a positive
  weight `E=-w`, for a negative one `E=0`;
- path `u-k-v` with two unit weights at `gamma=pi/2`, `beta=pi/4`:
  `c_uv=1` for the non-adjacent pair;
- at `gamma=0` or `beta=0` all distinct correlations are zero;
- an empty graph gives energy 0, a single vertex gives no correlation pairs;
- the reduction shift `b_t` changes neither correlations nor angles; with
  `objective_reference="original_graph"` the energy profile shifts by `-b_t`.

## 3. Full checks

Command from the repository root:

```bash
.venv-stage02/Scripts/python.exe -m pytest -q
```

Result for this build: **92 passed, 1 warning**. The warning concerns
`tight_layout` with 15 panels at p=3 in the visualisation test; it is not a
mathematics or API error.

Verified in particular:

- correlations and energy at fixed angles for `p=1`, including zero and
  negative weights, non-adjacent pairs, negative/missing indices;
- exact beta minimisation and two independent recoveries of `F0,Fs,Fc`;
- the minimum point of an explicit gamma grid and deterministic tie-breaking;
- `edges`/`all_pairs`, with no substitution of full coverage by edges;
- rejection of `p>1`, fields, shots, CVaR/Gibbs/Best-of-K/Threshold
  Probability, `initialization`/ADAM and unknown fields;
- pair and gamma batch sizes, absence of `2**N`;
- RQAOA with `shots_total=0`, complete solution and `C=b+C_reduced`;
- the earlier ExactCPU, QAOACPU, statevector RQAOA, saving and plots.

## 4. A problem larger than the statevector limit

Notebook 05 runs a sparse graph `N=30` (the statevector source limit is 24)
with the analytic source and a short full RQAOA:

- vertices: 30, edges: 95;
- status: `completed`, reductions: 27;
- cut: 23.25;
- `shots_used=0`;
- time: about `0.13 s` in the notebook.

A full chain on 1000 vertices is not claimed. For `N=1000`, weight-matrix
preparation and a single pass of coefficients/correlations were verified;
that is not a full RQAOA.

## 5. Time and memory profile

Data source: `docs/PERFORMANCE_ANALYTIC_P1_0.1.0a4.json`. Measurements were
taken on CPU, `num_treads=1`, `float64`, `pair_batch_size=2048`, 41 gamma
points for N≤256. Time is wall-clock, memory is peak `tracemalloc`. The
numbers are a local reference, not a performance promise.

| N | Edges | Pairs | 41-gamma profile, s | Profile memory, MiB | Correlations, s | Correlation memory, MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 110 | 496 | 0.020 | 0.21 | 0.014 | 0.94 |
| 64 | 314 | 2016 | 0.163 | 1.11 | 0.058 | 7.32 |
| 128 | 828 | 8128 | 0.803 | 5.80 | 0.316 | 17.57 |
| 256 | 1673 | 1673 (edges) | 2.858 | 23.35 | 0.089 | 23.51 |
| 1000 | 9838 | 9838 (edges) | not measured | — | 1.66 | 128.51 |

For `N=32`, a full RQAOA with 41 gamma points: `0.265 s`, 28 reductions,
`completed`, `shots_used=0`. Memory does not contain an object of size
`2**N`.

## 6. Build and installation

Executed:

```bash
.venv-stage02/Scripts/python.exe -m build
.venv-stage02/Scripts/python.exe -m sphinx -W -b html docs docs/_build/html
.venv-stage02/Scripts/python.exe -m QA_max_cut_lib.solvers.rqaoa_cpu \
  --graph configs/graphs/complete_signed_7.json \
  --config configs/solvers/rqaoa_analytic_p1_cpu.json --report 1 --logs 0
```

The CLI returned `completed`, `Cut=6`, `shots_used=0`, `reductions=3`. The
wheel was installed into `.venv-wheelcheck` without CuPy/CUDA; the import
came from `site-packages`, and the analytic example and notebook 05 executed.

## 7. Limitations and open items

- Only `p=1`, the standard `|+>`, an X mixer and Mean Cut are supported.
- There is no full distribution, no amplitudes, no sampling, no discrete
  readout, CVaR, Gibbs, Best-of-K or Threshold Probability.
- There are no local fields, no warm start with another state/mixer, no GPU,
  LS, QAOA-in-QAOA or parallel `num_treads>1`.
- The gamma minimum is known only on the scanned grid; a continuous global
  optimum is not certified.
- A full RQAOA chain on 1000 vertices was not run; preparation and batch
  passes were verified.
- The standalone 3D HTML was generated and contains no external `script src`;
  WebGL is unavailable in this environment's headless browser, so rotation
  and scaling of the interactive scene are not confirmed. This is an
  environment limitation, not a claim of a successful browser check.
- One `tight_layout` test warning with 15 panels at p=3 remains cosmetic.

## 8. Artifacts

- `dist/qa_max_cut_lib-0.1.0a4-py3-none-any.whl`;
- `dist/qa_max_cut_lib-0.1.0a4.tar.gz`;
- `notebooks/05_analytic_p1_rqaoa.ipynb`;
- `docs/PERFORMANCE_ANALYTIC_P1_0.1.0a4.json` and `.txt`;
- `docs/notebook_previews/05_analytic_p1_rqaoa.html`;
- `tests/test_analytic_p1.py`.
