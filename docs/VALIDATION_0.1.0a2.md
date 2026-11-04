# Build verification 0.1.0a2

**Date:** 2 November 2026.
**Stage:** 02 — exact core of pairwise Max-Cut → Ising reductions.
**Platform:** Windows 10 (10.0.19043), 4 physical / 8 logical cores,
15.7 GB RAM.
**Python:** 3.12.14 in a separate environment `.venv-stage02`.

**Input artifacts:** SHA-256 of `QA_max_cut_lib_0.1.0a1.zip` —
`c9b4f7ed86ecf39371957f93ba9c9e8521577c920ac8942f6da5e9ce10b8ba64`;
SHA-256 of the architecture — `3252449e29026503de050852486b4ccb15578d5e3d4d7602e57f1aa32db39ee7`.
Both values matched those stated in the assignment.

## Environment

| Package | Version |
|---|---|
| NumPy | 2.5.4 |
| SciPy | 1.18.1 |
| pandas | 3.1.0 |
| Matplotlib | 3.11.2 |
| Seaborn | 0.13.2 |
| Plotly | 7.1.0 |
| NetworkX | 3.7 |
| CVXPY | 1.9.3 |
| pytest | 9.2.0 |
| Sphinx | 9.1.1 |
| numpydoc | 1.11.0 |
| nbformat / nbclient | 5.11.1 / 0.11.0 |
| ipykernel | 7.4.0 |

CuPy is absent from `.venv-stage02`: stage 02 is a CPU stage. The working
environment `.venv` with CuPy 14.2.0 exists separately and was not used for
these checks.

## Commands executed

```bash
uv venv --python 3.12 .venv-stage02
uv pip install --python .venv-stage02/Scripts/python.exe -e ".[cpu,notebooks,dev,docs]"
.venv-stage02/Scripts/python.exe -m pytest -q
.venv-stage02/Scripts/python.exe -m build
.venv-stage02/Scripts/python.exe -m sphinx -W -b html docs docs/_build/html
```

## Automated checks

**56 passed, 0 failed.** The previous 42 checks are preserved; checks for the
reduction core were added: graph-to-Ising conversion, both signs of the
constraint, a general Ising with fields, non-adjacent pairs, edge
compensation, chains, negative indices and isolates, limit cases,
JSON round-trip, invalid input and event metrics. Reference values are
computed by direct summation of the cut or of the polynomial, not by calling
the reducer under test.

## Notebooks

| Notebook | Code cells | Images | Errors |
|---|---:|---:|---:|
| `01_quickstart.ipynb` | 8 | 5 | 0 |
| `02_current_api_guide.ipynb` | 14 | 6 | 0 |
| `03_reduction_core.ipynb` | 8 | 0 | 0 |

All three notebooks ran top to bottom in the clean kernel `qa-stage02`, which
points to `.venv-stage02`. Notebook 03 uses only classes that are actually
implemented and saves a state snapshot to `notebooks/_outputs/`.

## Wheel build and installation

Built:

- `dist/qa_max_cut_lib-0.1.0a2-py3-none-any.whl`;
- `dist/qa_max_cut_lib-0.1.0a2.tar.gz`.

The wheel was installed into a separate clean environment `.venv-wheelcheck`
without a link to `src`. The check was run from a directory outside the
project:

- `QA_max_cut_lib.__file__` resolves inside `site-packages` of that
  environment;
- `__version__ == "0.1.0a2"`;
- the reference reduction example yields `fixed_energy=-2`, `cut_offset=5`,
  `ising_constant=-6`, restored energy `-7`;
- `ExactCPU` and `QAOACPU` work, `result["package_version"]="0.1.0a2"`;
- JSON/NPY export and reading of the selected result work.

## Cross-check against the research archive

The formulas `w'_{vk}=w_{vk}+σw_{uk}` and `Δb=(1-σ)/2·Σw_{uk}` agree with
`Solvers/AnalyticalRQAOA.py::_reduce_graph` (lines 215–237): there the weight
of the retained vertex is also updated and `offset` is accumulated over all
incident edges. The legacy code was not run numerically; this check is
structural, not a claim of numerical equivalence of all historical
implementations.

## Main control example

Source graph: vertices `[10, 30, 80, 101]`, isolate `101`, edges
`(10,30,2)`, `(10,80,3)`, `(30,80,5)`. Vertex `10` was removed, `30` was
retained, `σ=-1`.

| Quantity | Value |
|---|---:|
| Working graph after the step | `[30, 80, 101]`, edge `(30,80,2)` |
| Accumulated shift `b` | 5 |
| Full Ising constant `c` | −6 |
| `fixed_energy` | −2 |
| `expected_fixed_energy` at `c_uv=-0.6` | −1.6 |
| `ising_constant_delta` | −1 |
| Restored solution | `{10:-1, 30:+1, 80:-1, 101:+1}` |
| `C_original` / `E_original` | 7 / −7 |

The equality `E_before(s_10=-s_30, s') = E_after(s')` was verified for all
8 assignments of the remainder.

## Verification limitations

- GPU, multithreading and CUDA were not used and were not verified at this
  stage.
- `num_treads > 1` is still rejected.
- Optimizer checkpoint/resume is not implemented; a `ReductionState` snapshot
  is a mathematical state, not a checkpoint.
- RQAOA, LS-RQAOA, pair selection and correlation sources are not
  implemented.
- Notebooks use MathText; a real LaTeX mode is outside stage 02 acceptance.
- No formal ASD-STE100 certification was performed.
