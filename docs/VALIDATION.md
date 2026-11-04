# Validation overview

This page answers three questions: what was actually run for the current release
candidate, what the result was, and where the detailed evidence is stored. It is an
evidence page, not a tutorial. The current build is `0.1.0rc2`; the historical
first-build record is kept at the end of this page for provenance.

## Current result

| Check | Command or scope | Result |
|---|---|---|
| CPU test suite | `.venv-stage02/Scripts/python.exe -m pytest -q` | `233 passed, 1 skipped, 1 warning` |
| GPU test suite | `.venv/Scripts/python.exe -m pytest -q -m gpu` | `10 passed, 218 deselected` |
| Documentation | `.venv-stage02/Scripts/sphinx-build.exe -W -b html docs docs/_build/html` | success, no warnings |
| Wheel import | install into a separate environment, import outside `src` | no CuPy, CUDA, CVXPY, or Qiskit required for the CPU import |
| Benchmark materialization | Benchmark297 resolver | 297/297 IDs materialized without solver calls |
| Generator/plan series | 48-case series | 0 quantum shots |
| Public quickstart | `examples/public_quickstart.py` from the installed wheel | optimum `3.5`, expected cut `3.044365`, best of 128 `3.5`, optimal mass `0.805755` |

The single warning is `UserWarning: Tight layout not applied` from
`tools/plotting.py` in a parameter-plot test. The layout is still produced; the
warning is reported rather than suppressed.

## What the suite covers

The 21 test modules are grouped by the property they defend, not by release stage:

| Area | Modules | What is checked |
|---|---|---|
| Scientific core | `test_science.py`, `test_reduction.py` | energy sign, cut reconstruction, exact enumeration against brute force, bit order, pair-reduction identity, offset `b_t`, Ising constant `c_t`, label restoration |
| QAOA and objectives | `test_science.py`, `test_analytic_p1.py` | statevector against dense matrix products, `p=1,2`, five objectives, exact and shot estimation, readout and budget reserve |
| RQAOA and LS | `test_rqaoa.py`, `test_ls_rqaoa.py`, `test_qaoa_in_qaoa.py` | correlation signs, pair selection, fallback, coverage, shared budget, hierarchy projection, block constraints |
| Advanced controllers | `test_conditional.py`, `test_block_mean_field.py`, `test_distributed_ls.py`, `test_incremental_ls.py`, `test_final_strategies.py` | conditional fields, Gauss-Seidel/Jacobi schedules, episode transport, epoch bank, batch reconciliation, quantized angles |
| Classical | `test_classical.py` | GW rounding and SDP objective, SA reads and seeds, exact block refinement, refined wrapper |
| I/O and records | `test_io_and_tools.py`, `test_checkpoint.py` | JSON/NPY export, `atts`, overwrite guard, concurrent exports, checkpoint generations, journal, reject by capability matrix |
| Series and diagnostics | `test_experiments.py`, `test_mutual_information.py` | stable IDs, rejected records, reference policy, profiler, MI identities for product, Bell, and CZ-plus states |
| Generators and Laboratory | `test_generators_and_plans.py`, `test_laboratory.py`, `test_warm_start_rqaoa.py` | schema1/schema2 recipes, typed plans, coverage, catalog counts, warm-start ensembles |
| Public surface | `test_public_quickstart.py`, `test_gpu.py` | the installed-wheel example, GPU paths on a real device |

The exact assertions and fixtures are in `tests/`; this page does not duplicate
them.

## How to reproduce

```bash
cd C:/Users/Alexey/PycharmProjects/QA_max_cut_lib
.venv-stage02/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m pytest -q -m gpu
.venv-stage02/Scripts/sphinx-build.exe -W -b html docs docs/_build/html
.venv-stage02/Scripts/python.exe examples/public_quickstart.py
```

One environment observation belongs here: on a cold Matplotlib font cache the first
full run failed in `tests/test_experiments.py::test_plot_reads_profile_without_solver`.
The same test passed alone and the repeated full run returned the documented result.
The failure is not reproduced once the font cache exists.

## What is not covered

- Hardware noise: all quantum execution is simulated; shots model ideal sampling.
- Graphs large enough to exceed the full statevector; the CPU QAOA path allocates
  `2^n` amplitudes.
- GPUs other than the local RTX 3050 Laptop; no multi-GPU configuration.
- Operating systems other than this Windows 10 machine for the local records. The
  first-build record below was produced on Linux and says so.
- Checkpoint/resume for RQAOA, LS, hierarchy, conditional, environment, incremental,
  and GPU runs; those codecs do not exist and are rejected explicitly.
- Statistical optimality of heuristic methods: a feasible cut is not a certificate.

## Where the evidence is

| Record | Scope |
|---|---|
| [`VALIDATION_0.1.0rc2.md`](VALIDATION_0.1.0rc2.md) | current release candidate: counts, artifacts, quickstart |
| [`VALIDATION_LOCAL.md`](VALIDATION_LOCAL.md) | Windows 10 / Python 3.12 local check of `0.1.0a2` (historical) |
| [`ARCHIVE.md`](ARCHIVE.md) | index of all historical acceptance records |
| `RELEASE_READINESS.md` (repository root) | release position and R19 closure evidence |
| `SOURCE_CLOSURE.json` (repository root) | per-file source closure ledger |
| `REQUIREMENTS_TRACEABILITY.json` (repository root) | R01–R65 matrix, 65 passed and 0 blocked |

## Historical record: first build `0.1.0a1`

> This section is the translated record of the first working CPU stage, dated
> 1 November 2026. It is kept for provenance and does not describe the current
> build.

The record verifies the first working CPU stage, not all capabilities of the target
architecture.

### Installation and automated checks

The wheel `qa_max_cut_lib-0.1.0a1-py3-none-any.whl` was built and installed into a
separate directory with no reference to `src`. The automated checks and both
tutorial scenarios used that installed wheel. **42 checks** were performed. The
Sphinx/numpydoc reference was built with `-W` and no warnings; the HTML pages are in
`docs/_build/html`.

Checked:

- `ExactCPU` against independent enumeration of all spin assignments: signed
  weights, original labels with gaps, isolated vertices, maximum, minimum, and
  degeneracy.
- QAOA against independent dense matrix exponentiation for `p=1` and `p=2`: the
  sign of `H_E=-C`, layer order, and bit order.
- An analytic one-edge graph: expected energy `-1` at the check angles and
  correlation `-1`.
- The five objectives on known distributions, fractional CVaR boundary mass,
  independent Best-of-K computation, and a stable Gibbs evaluation.
- Training, both state precisions, a strict shared shot budget, RNG recreation from
  the seed, independence of a saved record from later runs and graph changes.
- JSON/complex/ndarray/NPY, relative links, both `savedir` modes, an extra-file
  conflict, and simultaneous exports to one name.
- All nine combinations of `report`/`logs` levels, no execution at import, the CLI
  of both modules, and explicit rejection of modes not yet implemented.
- Static plots and restoration of the global theme, metadata caption, array
  orientation of the `p=1` map, and map saving.

A numerical check does not prove QAOA quality for an arbitrary problem or parameter
set. Exact enumeration uses `float64`. Size limits are guard limits, not a measured
performance guarantee.

### Tutorial scenarios

`01_quickstart.ipynb`: 8 code cells and 5 real images. `02_current_api_guide.ipynb`:
14 code cells and 6 real images. Both were executed top to bottom in separate clean
IPython processes, with saved outputs, assertions, and nbformat structure checked. An
HTML view was produced with nbconvert.

In that environment local TCP/IPC sockets were forbidden, so a normal Jupyter kernel
could not be started through nbclient. The check therefore does not confirm a Jupyter
server in that environment; it confirms sequential execution of the cells in IPython
and the scientific scenario on the installed wheel. For a local installation the user
starts the notebook with standard Jupyter.

The HTML views were produced by nbconvert. Browser rendering of their layout was not
verified: Chromium also could not start because local sockets were forbidden. The PNG
images embedded in the notebooks were inspected manually. Mathematical formulas in
the HTML view may require network MathJax resources; a standalone interactive HTML
belongs to later stages.

The plots were inspected separately after execution. Visual inspection found an
overlap between the title and the graph statistics; the padding was fixed and the
images were regenerated. Long user captions and sizes should be checked separately.

### Environment and limits

Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, Matplotlib 3.10.8, Seaborn 0.13.2; Linux.
The full machine-readable list of principal versions is in
`validation_environment.json`.

The verified text engine was MathText with Computer Modern. The system TeX had no
working `latex.fmt`; a successful LaTeX render was not confirmed. NVIDIA, GPU
installation profiles, Windows/macOS, older and newer claimed Python versions, and
file systems without hard links were not checked.

RQAOA, LS, QAOA-in-QAOA, GW, Laboratory, parallel pools, the profiler, shared series
execution, checkpoints, and interactive HTML were not implemented at that point.
Consequently that build does not confirm their correctness or performance. Formal
ASD-STE100 certification was not performed.
