# Git summary for stage 04

**Commit:** `bd62460bd17e0d99ffd1a08907a9e928dfec2616`  
**Message:** `Stage 04: add analytic p=1 correlations and RQAOA source (0.1.0a4)`  
**Date:** 2 November 2026.

The local working tree before this commit contained the completed Stage 03
RQAOA code, which had not been committed separately. The commit therefore
contains both the preserved Stage 03 base and the Stage 04 analytic work.
The package version is `0.1.0a4`.

## Main additions

- `src/QA_max_cut_lib/solvers/analytic_p1.py` — analytic p=1 evaluator and
  correlation source.
- `src/QA_max_cut_lib/solvers/components.py` — dispatch to the analytic source
  validator.
- `src/QA_max_cut_lib/solvers/rqaoa_cpu.py` — source selection, analytic
  resource counters and profile in local history.
- `src/QA_max_cut_lib/tools/reduction_plots.py` — gamma-profile plot.
- `configs/solvers/rqaoa_analytic_p1_cpu.json`.
- `tests/test_analytic_p1.py` — 12 independent analytic tests.
- `notebooks/05_analytic_p1_rqaoa.ipynb`.
- Validation and performance reports in `docs/`.

## Reproduce

```bash
.venv-stage02/Scripts/python.exe -m pytest -q
.venv-stage02/Scripts/python.exe -m build
.venv-stage02/Scripts/python.exe -m sphinx -W -b html docs docs/_build/html
.venv-stage02/Scripts/python.exe -m QA_max_cut_lib.solvers.rqaoa_cpu \
  --graph configs/graphs/complete_signed_7.json \
  --config configs/solvers/rqaoa_analytic_p1_cpu.json --report 1 --logs 0
```

The command returned `completed`, cut `6`, `shots_used=0`.
