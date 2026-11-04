# Verification 0.1.0a11

Date: 2 November 2026.

## Base

- Python 3.12.14, Windows 10; NumPy 2.5.4.
- `QA_max_cut_lib` `0.1.0a11`.
- CPU tests: `161 passed, 1 skipped, 1 warning`.
- GPU tests: `10 passed` (old GPU profile; GPU warm start is not implemented).
- Sphinx `-W`: see the build-run report.
- Wheel/sdist `0.1.0a11` installed into `.venv` and `.venv-wheelcheck`.

## Inventory/catalog

```text
archive sha256: b4c684a5d4da469adc25bd99a405e402892d1a2524465e4c97ab6229c71a4aed
Python members: 89
parsed:         89
parse errors:   0
directories:    Diplom 28, ENV 3, Generators 3, Laboratory 22,
                Other 2, RQAOA_exp 2, Solvers 18, Tools 11
```

- `Diplom/Mk_5.py` and `Laboratory/LS_RQAOA_fast_5.py`: identical AST
  (`ast.dump(..., include_attributes=False)`), different raw SHA-256;
  equivalence marked provisional.
- All 89 file records carry a path, SHA-256, role, variant ID and
  review/runtime status.
- `list/show` does not import CuPy/CVXPY and does not run the old archive.

## Mathematical controls

A: `q=(0.25,0.75)`, `P(cuts)=(3/16,1/16,9/16,3/16)`, initial expected
C=1.25, source cut=2.

B: epsilon=1/2, p=1 and p=2, several signed graphs:
`rounded_modified` with beta and `warm_aligned` with `-beta` give the same
full distribution as a plain QAOACPU with the same gamma/beta (max error 0).

C: epsilon=0, aligned/modified preserve the basis state; probability mass 1,
no NaN.

D: gamma=0 aligned preserves the product distribution; modified changes it;
epsilon=1/4, gamma=0, beta=pi/2 gives a complement seed with the same C.

E: `q=(0.1,0.6,0.95)`, epsilon=0.2 → `(0.2,0.6,0.8)`; q>1, epsilon<0,
epsilon>0.5, missing/duplicate node labels are rejected.

## Runs

```text
provided_spins: 544 shots = 480 training + 64 readout
pause after 1 iteration: known=160, held=64, available=320
resume: status=completed, shots=544, angles equal continuous
```

`WarmStartQAOACPU` with a checkpoint:
- source/q/epsilon/mixer frozen;
- `restore()` does not start the solver;
- `resume()` continues the canonical run;
- the classical source is not solved again on resume;
- `solve()` on a restored object is a new run.

## H

```text
3 graph instances × 2 trial seeds × 7 configs = 42 runs
base + {epsilon 0.25,0.50} × {x, warm_aligned, rounded_modified}
quantum budget = 42 × 544 = 22 848 shots
```

Actual `ExperimentRunner`:
`planned=42 valid=42 failed=0`, success(valid)=0.8333, wall 0.203 s
(in the notebook, 4-node graphs). The exact reference is a separate reference
job. This is not proof of warm-start superiority.

## Classical source

`gw_cpu` and `simulated_annealing_cpu` were actually executed in the classical
profile (`.venv-stage02`):
- GW: 8 hyperplanes, SDP call, `source_cost` contains `sdp_calls`,
  `hyperplanes`, `solve_seconds`, `total_seconds`;
- SA: 4 reads, 64 sweeps, `source_cost` contains reads/sweeps/time;
- quantum shots remain separate (`224` for p=1/1 step).

## Limitations

- GPU warm start is not implemented; full CPU statevector only.
- `provided_*` source cost is external/unaccounted.
- No warm-start RQAOA, LS/hierarchy, fields/environment, MI strategies.
- No automatic multiple-start/aggregation.
- H is not proof of an algorithmic advantage.

## Note on the archive

The RAR itself is not in the working tree; the inventory was built from the
extracted `reference/RQAOA_main` (89 files). The archive SHA-256 value is taken
from the assignment; all 89 member hashes are recorded in the inventory.
Verification of the RAR container as a whole was not performed because the
container is absent.
