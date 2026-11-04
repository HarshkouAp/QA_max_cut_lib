# Project environment

**Version:** `QA_max_cut_lib 0.1.0rc2`.
**Project directory:** `C:\Users\Alexey\PycharmProjects\QA_max_cut_lib`.

This page describes the working environment, the package layout, and the commands
that reproduce the reported checks. It is written for someone who has to rebuild
the environment or continue the migration on this machine; installation details for
other systems are in [`INSTALLATION.md`](INSTALLATION.md).

## What is prepared

1. A project directory with the `src` layout of the `QA_max_cut_lib` package.
2. A CPU virtual environment `.venv-stage02` on Python 3.12.14 without CuPy.
3. A GPU virtual environment `.venv` with CuPy 14.2.0 for the NVIDIA checks.
4. Full CPU dependency profiles and separate NVIDIA CUDA 12/13 profiles.
5. Notebook, build, and Sphinx documentation tools.
6. The source research archive `RQAOA_main` in `reference/RQAOA_main`, read-only and
   used only for comparison during the migration.
7. The agreed architecture in `docs/reference/MaxCut_Library_Architecture.md`.
8. The exact core of pair reductions Max-Cut to Ising in `core/reduction.py`.
9. A git repository with the full history.

The current build is a release candidate: CPU and GPU solver families, reductions,
Laboratory controllers, series, and checkpoints are implemented, while the source
closure R19 is still open. `0.1.0rc2` is not the final `0.1.0`.

## Project structure

```text
QA_max_cut_lib/
├── src/QA_max_cut_lib/
│   ├── core/           # graph, records, saving, factory, budget, reduction,
│   │                   # experiments, checkpoint
│   ├── solvers/        # base contract, exact, QAOA, RQAOA, LS, hierarchy,
│   │                   # classical, warm start, conditional, mean field
│   ├── generators/     # graph recipes, structured weights, plans, benchmarks
│   ├── tools/          # numerics, plots, reports, experiments, parallelism
│   └── laboratory/     # catalog and CPU assemblies of historical variants
├── tests/              # 21 test modules
├── configs/            # JSON examples for graphs, solvers, and plots
├── examples/           # public quickstart and module demonstrations
├── notebooks/          # 00–19 and saved runs in notebooks/_outputs
├── docs/               # API, mathematics, status, validation, Sphinx sources
├── reference/RQAOA_main/  # source archive (89 files), read-only
├── requirements/       # CPU/GPU profiles, notebook and docs profiles, locks
├── results/            # numerical results and figures (not committed)
├── .venv/              # GPU environment with CuPy (not committed)
├── .venv-stage02/      # CPU environment without CuPy (not committed)
└── pyproject.toml
```

The reference archive contains 89 Python files and 25 548 lines. `core`, `solvers`,
`generators`, and `tools` are the public layers; `laboratory` holds versioned
controllers that may use every layer, while the base layers never import it.

## Environment

| Parameter | Value |
|---|---|
| OS | Windows 10, 10.0.19043 |
| Python | 3.12.14 (uv-managed CPython) |
| CPU environment | `C:\Users\Alexey\PycharmProjects\QA_max_cut_lib\.venv-stage02` |
| GPU environment | `C:\Users\Alexey\PycharmProjects\QA_max_cut_lib\.venv` |
| NumPy / SciPy | 2.5.4 / 1.18.1 |
| pandas / NetworkX | 3.1.0 / 3.7 |
| Matplotlib / Seaborn / Plotly | 3.11.2 / 0.13.2 / 7.1.0 |
| CVXPY / SCS | 1.9.3 / 3.3.1 |
| dimod / dwave-samplers | 0.12.23 / 1.8.0 |
| CPU | 4 physical cores, 8 logical |
| RAM | 15.7 GiB |
| GPU | NVIDIA GeForce RTX 3050 Laptop, 4 GiB |
| NVIDIA driver | 591.74 |
| CUDA Toolkit | 13.0 (V13.0.48) |
| CuPy | 14.2.0 (`cupy-cuda13x`, system CUDA, no `ctk` extra) |
| TeX | MiKTeX 25.12, pdfTeX 4.23 |

The CPU environment has no CuPy by design: importing the package and running the
CPU test suite must work without CUDA. The GPU environment is used only for the
NVIDIA checks. The exact installed versions are listed in
`requirements/freeze-local-py312-win.txt`; the allowed profile versions are in
`requirements/lock-cpu-py312-win.txt` and
`requirements/lock-gpu-cu13-py312-win.txt`.

### Activation

PowerShell:

```powershell
cd C:\Users\Alexey\PycharmProjects\QA_max_cut_lib
.\.venv-stage02\Scripts\Activate.ps1
```

Git Bash:

```bash
cd /c/Users/Alexey/PycharmProjects/QA_max_cut_lib
source .venv-stage02/Scripts/activate
```

In PyCharm: **File → Open** → the project directory, then select
`.venv-stage02\Scripts\python.exe` as the interpreter. Use `.venv` only for GPU
work.

A user Jupyter kernel `Python 3.12 (QA_max_cut_lib)` is registered for JupyterLab:

```bash
.venv-stage02/Scripts/jupyter.exe lab notebooks/00_quickstart.ipynb
```

The kernel points at this project's environment. If the project moves, register
the kernel again.

### Clean installation

```bash
uv venv --python 3.12 .venv-stage02
uv pip install --python .venv-stage02/Scripts/python.exe -e ".[cpu,notebooks,dev,docs]"
```

The GPU profile is installed separately, into `.venv`, and exactly one CUDA
variant is chosen:

```bash
uv pip install --python .venv/Scripts/python.exe -e ".[cpu,gpu-cu13]"
```

For CUDA 12 use `gpu-cu12` instead of `gpu-cu13`.

## Verification

```bash
cd C:/Users/Alexey/PycharmProjects/QA_max_cut_lib
.venv-stage02/Scripts/python.exe -m pytest -q
.venv-stage02/Scripts/python.exe scripts/check_environment.py
.venv-stage02/Scripts/sphinx-build.exe -W -b html docs docs/_build/html
```

Expected results on this machine:

| Check | Command | Result |
|---|---|---|
| CPU suite | `.venv-stage02/Scripts/python.exe -m pytest -q` | `218 passed, 1 skipped, 1 warning` |
| GPU suite | `.venv/Scripts/python.exe -m pytest -q -m gpu` | `10 passed, 218 deselected` |
| Sphinx | `.venv-stage02/Scripts/sphinx-build.exe -W -b html docs docs/_build/html` | success, no warnings |

The single warning is a Matplotlib `tight_layout` warning from a parameter-plot
test; it is reported, not suppressed. A first full run on a cold Matplotlib font
cache can fail in `tests/test_experiments.py::test_plot_reads_profile_without_solver`;
the test passes in isolation and on the next full run.

Module demonstrations:

```bash
.venv-stage02/Scripts/python.exe -m QA_max_cut_lib.solvers.exact_cpu --report 1 --logs 0
.venv-stage02/Scripts/python.exe -m QA_max_cut_lib.solvers.qaoa_cpu --report 1 --logs 0
.venv-stage02/Scripts/python.exe examples/public_quickstart.py
```

## GPU profile

CuPy is installed in `.venv`: `cupy-cuda13x 14.2.0`, using the system CUDA Toolkit
13.0. The `ctk` variant was not used because the system CUDA is installed and is
compatible with driver 591.74.

Verified earlier on this machine:

- `cupy.cuda.runtime.runtimeGetVersion()` returns `13020`;
- `driverGetVersion()` returns `13010`;
- the device is available with about 3.3 GB free of 4 GB;
- elementwise operations and `RawKernel` compilation through NVRTC work.

Limit: 4 GB of video memory. Choose the local problem size from the actual free
memory and run the built-in memory preflight before a large allocation. A
preflight is a guard, not a performance guarantee. The current GPU tests cover
`QAOAGPU`, the GPU correlation source, `RQAOAGPU`, `LSRQAOAGPU`, and the GPU
QAOA-in-QAOA path on this one device; other cards and multi-GPU are not covered.

`requirements/gpu-nvidia-cu12.in` and `requirements/gpu-nvidia-cu13.in` are
prepared for clean installations with the `ctk` extra. To repeat the local
installation without `ctk`, a compatible `cupy-cuda13x` version is sufficient.

## Working rules

1. Work inside this project; the source archive is never modified.
2. Before a stage, the working tree is clean and the tests are green.
3. New capabilities are not replaced by stubs with a successful status.
4. After a stage: tests, a demonstration run, documentation updates, and a separate
   git commit.
5. Numerical results and figures are written to `results/` and are not committed
   unless they are a reference example.
6. Names and contracts are checked against the architecture; deviations are
   recorded explicitly.

## Known limitations

They are not hidden and are not presented as finished capabilities:

- R19 is open: 18 of the 89 source records are reviewed and deferred, so
  `full_archive_migration_complete=false`. See `RELEASE_READINESS.md` and
  `SOURCE_CLOSURE.json`.
- No D-Wave quantum annealer; the classical SA is implemented.
- The public `solve(graph)` does not accept fields; fields are available in
  `IsingProblem` and in the conditional local problem.
- No multi-GPU execution, CUDA Graphs, Walsh tiles, or `batch>1` on the GPU path.
- Checkpoint/resume exists for `QAOACPU` and four Laboratory controllers; RQAOA, LS,
  hierarchy, conditional, environment, incremental, and GPU runs have no codec and
  are rejected explicitly by the capability matrix.
- `num_treads>1` is implemented in `LSRQAOACPU`, `IncrementalLSRQAOACPU`,
  `QAOAInQAOACPU`, and `SimulatedAnnealingCPU`; the other solvers are sequential.
- Large-distribution histograms as columns on a plane and a resumable landscape
  scanner are not implemented.
- No public distribution license; the package is not on PyPI.
- `ASD-STE100` conformance is not certified.
