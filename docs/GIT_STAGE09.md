# Git Stage 09

Основной коммит реализации Stage 09 и Stage 10 объединён в финальной сборке
`0.1.0a10`; отдельный a9-архив не выпускался, потому что a9 не был упакован
до начала Stage 10.

## Состав

- `core/experiment_ids.py`, `experiment_spec.py`, `runner.py`,
  `references.py`, `measurement.py`, `profile.py`, `series_analysis.py`,
  `series_store.py`;
- `tools/experiments.py`, `tools/parallelism.py`, `tools/experiment_plots.py`;
- `experiments/benchmark_small.json` и дополнительные CPU/GPU/smoke configs;
- `notebooks/10_experiments_and_parallelism.ipynb`;
- `docs/EXPERIMENTS_RU.md`, `VALIDATION_0.1.0a9.md`.

## Проверки

```text
small series: 56 valid, 0 failed, 2 exact references
full profile: 84 points, 84 warmups, 252 measured, 336 calls
concurrent batch: 4 valid, 0 failed
mixed CPU/GPU: 3 valid, qaoa_gpu real, gpu_synchronized=true
```
