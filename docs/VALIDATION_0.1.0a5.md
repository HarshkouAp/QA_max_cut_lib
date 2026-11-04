# Build verification 0.1.0a5

**Version:** `QA_max_cut_lib 0.1.0a5`, 2 November 2026.  
**Base:** local build after stage 04 (`0.1.0a4`).  
**Environment:** Python 3.12.14, Windows 10 `10.0.19043`, 8 logical CPUs,
15.7 GiB RAM. CPU profile without CUDA/CuPy.

## 1. What was verified

1. The two subgraph builders and their coverage contract.
2. Induced local problems without boundary edges or fields.
3. Mean aggregation over the observations of each pair.
4. `A_t`/`C_t` coverage, `require_full`, the anchor problem.
5. A single global `GraphReducer` per step and energy restoration.
6. The analytic p=1 source and the statevector p=1/p=2 source.
7. Sequential and parallel modes of the same plan.
8. The shared shot budget, round-robin quotas and stopping.
9. Standalone HTML, saving and reloading.
10. A clean wheel installation and notebook 06 outside the source tree.

## 2. Baseline readiness of the previous stages

The local `0.1.0a4` base contained stages 02–04 and passed 92 tests.
`GraphReducer`, `RQAOACPU`, the statevector and analytic sources, the exact
finisher and the tools were checked. `0.1.0a5` adds only the LS layer; the
earlier reduction mathematics was not duplicated.

## 3. Tests

```text
109 passed, 1 warning
```

The warning concerns `tight_layout` for 15 panels at `p=3` in the old RQAOA
test and is not an LS error. New checks live in `tests/test_ls_rqaoa.py`.

## 4. Sequential and parallel analytic LS

Graph: `configs/graphs/erdos_signed_30.json`, 30 vertices, 128 edges.
Source: `qaoa_analytic_p1_cpu`, 17 gamma points, 24 reductions.

| Mode | Cut | Reductions | Local tasks | Applied P | Shots |
|---|---:|---:|---:|---:|---:|
| CLI, P=1 | 41.7465463 | 24 | 98 | 1 | 0 |
| CLI, P=4 | 41.7465463 | 24 | 98 | 4 | 0 |
| Notebook, P=1 | 41.7465463 | 24 | 98 | 1 | 0 |
| Notebook CLI, P=2 | 41.7465463 | 24 | 98 | 2 | 0 |

The solution and the number of reductions matched. In the notebook the
parallel run is performed by the importable CLI as a subprocess; inside the
CLI a `ProcessPoolExecutor` with `spawn` was used.

## 5. Statevector p=2

Graph: 30 vertices, `qaoa_statevector_cpu`, `layers=2`, local limit 20,
`max_size=8`, P=2. The run completed: 25 reductions, 124 local tasks,
`num_treads_applied=2`, `shots_used=0`. The statevector limit is checked
against the size of the local problem, not against the 30-vertex parent.

## 6. Shot-based statevector and budget stopping

`configs/solvers/ls_rqaoa_shots.json`, 30 vertices, `shots_per_evaluation=64`,
`correlations.shots=128`, `shots_total=6000`, P=2.

```text
status=stopped
termination_reason=shot_budget
reductions=1
shots_used=5760
local_tasks=10
```

A complete solution over all 30 vertices was returned. Total spending did not
exceed 6000. With a small budget of 1000 no task of the step starts: the
mandatory reserve of 10×128 already exceeds the limit, `shots_used=0`.

## 7. Shared CPU profile

Source: `docs/PERFORMANCE_LS_0.1.0a5.json`.
64-vertex graph; 58 reductions; 442 local tasks; 9 gamma points; the same cut
`66.3797621` at P=1/2/4.

| P | Applied P | Time, s | Peak tracemalloc, MiB | Solution match |
|---:|---:|---:|---:|---|
| 1 | 1 | 12.80 | 3.38 | yes |
| 2 | 2 | 7.22 | 3.57 | yes |
| 4 | 4 | 6.65 | 3.07 | yes |

On small local problems the `spawn` overhead is noticeable; speedup is not a
mathematical requirement. The profile shows a real overlap of task time and
the same scientific result.

## 8. Limited 1000-vertex check

Source: `docs/SCALE_LS_0.1.0a5.json`.

```text
N=1000, E=4004
max_size=8, rounds=1, max_reductions=2, P=4
status=stopped, reason=max_reductions
reductions=2, local_tasks=246, skipped=4
applied P=4, shots=0
time=26.49 s, peak tracemalloc=6.02 MiB
complete solution: yes
```

This is a limited check of preparation, memory and two reductions. It is not a
completed solution of a 1000-vertex problem.

## 9. Build, documentation and installation

- `python -m pytest -q`: **109 passed, 1 warning**.
- `python -m build`: wheel and sdist `0.1.0a5`.
- Wheel SHA-256: `cf02444530bcba89e856ab11cc9dfcb38f985fee26e97ed64dba830ec4034e31`.
- sdist SHA-256: `1079ba402cc14d651475ad28b3f59894e283e8f0e6f20f50da716c66812aedff`.
- `python -m sphinx -W -b html docs docs/_build/html`: success.
- The wheel was installed into `.venv-wheelcheck` via `uv pip install --python ...`;
  the import resolves from `site-packages`.
- Notebook 06 was executed through the installed wheel; the CLI subprocess
  used the same interpreter.
- `plot_parameter_trajectories_3d` produces standalone HTML with no external
  `<script src>`; Plotly.js is embedded.

## 10. Limitations

- Basic LS only: `mean`, `refresh_all`, induced problems without fields, the
  exact finisher.
- No QAOA-in-QAOA, Meta-LS, partial updates, fields or GPU.
- On Windows/Jupyter a direct `spawn` depends on the main module; the notebook
  uses the documented CLI subprocess.
- `max_local_tasks_total` is a hard limit on the step plan, not a dynamic
  scheduler.
- The 3D scene was verified statically; this environment's headless browser
  does not support WebGL.
