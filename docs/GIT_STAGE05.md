# Git summary: Stage 05

**Commit:** `ec4d0973312a1204c02dd1192a8d44a0d01a9618`  
**Message:** `Stage 05: add LS-RQAOA CPU with process execution (0.1.0a5)`  
**Package version:** `0.1.0a5`

The commit continues the local `main` history after Stage 04. It preserves the
Stage 02–04 reduction core and RQAOA coordinator and adds the LS layer:

- `src/QA_max_cut_lib/generators/subgraphs.py`
- `src/QA_max_cut_lib/solvers/ls_components.py`
- `src/QA_max_cut_lib/solvers/ls_worker.py`
- `src/QA_max_cut_lib/solvers/ls_rqaoa_cpu.py`
- `tests/test_ls_rqaoa.py`
- LS configurations, graph recipes, notebook 06, validation and performance
  reports.

The working tree was clean after this commit except for the follow-up
`docs/GIT_STAGE05.md` summary and archive generation.
