# Git Stage 09–10

Implementation commit: `eef2fa4` — `Stage 10: add QAOACPU checkpoints and resume (0.1.0a10)`.

## Contents of a10

- `core/checkpoint.py`: immutable `json_npy_v1` generations, atomic
  staging/rename/latest, SHA-256 manifest, `RunControl`, `ResourceJournal`,
  inspect/restore.
- `core/checkpoint_capabilities.py`: explicit capability matrix; unknown
  components are rejected.
- `QAOACPU`: pause after completed ADAM iterations, best evaluation ID, full
  PCG64 state, `training_done`/`readout_done`, a single readout token.
- `tools/checkpoints.py`: inspect/resume CLI.
- `examples/checkpoint_resume.py`, `configs/solvers/qaoa_cpu_checkpoint.json`.
- `notebooks/11_checkpoints_and_resume.ipynb` and preview.
- Stage 09: `core/experiment_*`, `core/profile.py`, runner/references/series,
  `tools/experiments.py`, `tools/parallelism.py`, notebook 10, configs and
  `EXPERIMENTS.md`.

## Checks

```text
CPU: 151 passed, 1 skipped, 1 warning
GPU: 10 passed
Sphinx -W: success
wheel/sdist: 0.1.0a10
examples + CLI from clean wheelcheck: pause/inspect/resume
Stage 09 small series: 56 valid, 0 failed, 2 exact refs
Stage 09 profile: 84 points, 84 warmups, 252 measured, 336 calls
concurrent batch: 4 valid, 0 failed
mixed CPU/GPU: 3 valid, qaoa_gpu completed, gpu_synchronized=true
```

RQAOA/LS/hierarchy/Exact/GW/SA/refiner checkpoint are not part of a10 and are
rejected explicitly; the capability matrix is in `docs/CHECKPOINTS.md`.
