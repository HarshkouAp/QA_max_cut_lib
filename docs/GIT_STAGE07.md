# Git Stage 07

Implementation commit: `1ebfd75` — `Stage 07: add hierarchical QAOA-in-QAOA (0.1.0a7)`.

The commit also contains files of stage 06 that had not been recorded earlier
(`QAOAGPU`, the GPU source, GPU-RQAOA/LS, notebook 07 and report a6), because
the local base already contained them before stage 07 started but they were
not a separate commit.

## Contents of stage 07

- `src/QA_max_cut_lib/solvers/hierarchy_components.py` — strict local solver
  validation, pattern normalisation, independent `W/Delta b` projection,
  worker function;
- `src/QA_max_cut_lib/solvers/qaoa_in_qaoa_cpu.py` — shared coordinator, CPU
  pool, quotas, atomic projection and tree;
- `src/QA_max_cut_lib/solvers/qaoa_in_qaoa_gpu.py` — a single GPU owner;
- CPU/GPU/shots/random partitioner configurations and an N=128 graph;
- tests `tests/test_qaoa_in_qaoa.py` and a GPU test;
- Tools `hierarchy_plots.py`;
- notebook `08_qaoa_in_qaoa.ipynb` and an HTML preview;
- documentation, report and profile JSON.

## Checks before the commit

```text
CPU: 124 passed, 1 skipped, 1 warning
GPU: 10 passed
Sphinx -W: success
wheel/sdist: 0.1.0a7
```

The build artifacts are in `dist/` and are not part of the Git commit
(`dist/` is excluded by `.gitignore`); they are added separately to the
delivered zip.
