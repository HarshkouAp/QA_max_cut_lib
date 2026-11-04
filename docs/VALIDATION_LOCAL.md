# Local environment check

> **Historical record.** This is the Windows 10 check of `0.1.0a2`, dated
> 2 November 2026. It is kept as acceptance evidence and does not describe the
> current build; current counts are in
> [`VALIDATION_0.1.0rc2.md`](VALIDATION_0.1.0rc2.md).

**Machine:** Windows 10 (10.0.19043), 4 physical / 8 logical cores, 15.7 GB RAM.
**Python:** 3.12.14 in `.venv-stage02`.
**Build under test:** `QA_max_cut_lib 0.1.0a2`.
**Detailed stage report:** `VALIDATION_0.1.0a2.md`.

## Checks performed

| Check | Command | Result |
|---|---|---|
| Automated tests | `.venv-stage02/Scripts/python.exe -m pytest -q` | **56 passed** |
| `ExactCPU` CLI | `python -m QA_max_cut_lib.solvers.exact_cpu --report 1 --logs 0` | `completed`, Cut = 5 |
| `QAOACPU` CLI | `python -m QA_max_cut_lib.solvers.qaoa_cpu --report 1 --logs 0` | `completed`, 20 ADAM steps |
| Quickstart | `python examples/quickstart.py` | Exact and QAOA both gave energy -6, quality = 1.000 |
| Notebook 01 | `jupyter nbconvert --execute notebooks/01_quickstart.ipynb` | executed completely, 5 images, JSON/NPY saved |
| Notebook 02 | `jupyter nbconvert --execute notebooks/02_current_api_guide.ipynb` | executed completely, 6 images, map and plots |
| Notebook 03 | `jupyter nbconvert --execute notebooks/03_reduction_core.ipynb` | 8 code cells, 0 errors |
| Wheel build | `python -m build` | wheel and sdist `0.1.0a2` created |
| Clean wheel install | `.venv-wheelcheck` outside the project | import from `site-packages`, reduction and solvers work |
| Sphinx build | `python -m sphinx -W -b html docs docs/_build/html` | success, no warnings |
| GPU environment `.venv` (separate) | CuPy 14.2.0, CUDA 13.0, `RawKernel` | checked earlier; not used by stage 02 |
| Source integrity | SHA-256 of 89 Python files | 89/89 match the `RQAOA_main.rar` extraction |
| Profile import | `import QA_max_cut_lib, cvxpy, plotly, dimod, dwave.samplers, community` | success |

## Principal package versions

| Package | Version |
|---|---|
| NumPy | 2.5.4 |
| SciPy | 1.18.1 |
| pandas | 3.1.0 |
| Matplotlib | 3.11.2 |
| Seaborn | 0.13.2 |
| Plotly | 7.1.0 |
| CVXPY | 1.9.3 |
| SCS | 3.3.1 |
| NetworkX | 3.7 |
| dimod | 0.12.23 |
| dwave-samplers | 1.8.0 |
| pytest | 9.2.0 |
| Sphinx | 9.1.1 |
| CuPy | 14.2.0 (`cupy-cuda13x`) |

Full list: `requirements/freeze-local-py312-win.txt`; allowed profile versions:
`requirements/lock-cpu-py312-win.txt` and
`requirements/lock-gpu-cu13-py312-win.txt`.

## What is confirmed

- The package installs into a separate environment in editable mode.
- The earlier 42 checks and 14 new reduction-core checks pass on Windows and
  Python 3.12.
- Importing the package does not load Matplotlib or CuPy.
- The module CLI examples run without starting a computation at import.
- Plots are built in a local theme and do not change global `matplotlib.rcParams`.
- JSON/NPY saving, the overwrite guard, and simultaneous exports pass the tests.

## What this check did not cover

- The full GPU solver: only CuPy, the device, and kernel compilation were checked.
- The Sphinx and wheel builds were verified, but their contents were not compared
  with the earlier `ARTIFACTS.json`.
- Numerical QAOA quality on large graphs: the starting build is limited to the full
  statevector and small problems.
- `style.text_engine="latex"`: the stage 02 notebooks use MathText; real LaTeX is
  outside that stage's acceptance.
