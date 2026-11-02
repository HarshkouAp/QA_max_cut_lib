# Git Stage 09–10

Коммит реализации: `eef2fa4` — `Stage 10: add QAOACPU checkpoints and resume (0.1.0a10)`.

## Состав a10

- `core/checkpoint.py`: immutable `json_npy_v1` generations, atomic
  staging/rename/latest, SHA-256 manifest, `RunControl`, `ResourceJournal`,
  inspect/restore.
- `core/checkpoint_capabilities.py`: явная capability matrix; неизвестные
  компоненты отклоняются.
- `QAOACPU`: pause после завершённых ADAM-итераций, best evaluation ID,
  полный PCG64 state, `training_done`/`readout_done`, один readout token.
- `tools/checkpoints.py`: CLI inspect/resume.
- `examples/checkpoint_resume.py`, `configs/solvers/qaoa_cpu_checkpoint.json`.
- `notebooks/11_checkpoints_and_resume.ipynb` и preview.
- Stage 09: `core/experiment_*`, `core/profile.py`, runner/references/series,
  `tools/experiments.py`, `tools/parallelism.py`, notebook 10, configs и
  `EXPERIMENTS_RU.md`.

## Проверки

```text
CPU: 151 passed, 1 skipped, 1 warning
GPU: 10 passed
Sphinx -W: успешно
wheel/sdist: 0.1.0a10
examples + CLI from clean wheelcheck: pause/inspect/resume
Stage 09 small series: 56 valid, 0 failed, 2 exact refs
Stage 09 profile: 84 points, 84 warmups, 252 measured, 336 calls
concurrent batch: 4 valid, 0 failed
mixed CPU/GPU: 3 valid, qaoa_gpu completed, gpu_synchronized=true
```

RQAOA/LS/hierarchy/Exact/GW/SA/refiner checkpoint не входят в a10 и явно
отклоняются; capability matrix находится в `docs/CHECKPOINTS_RU.md`.
