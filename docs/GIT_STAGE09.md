# Git Stage 09

The main implementation commit for Stage 09 and Stage 10 was combined in the
final `0.1.0a10` build; a separate a9 archive was not released because a9 was
not packaged before Stage 10 started.

## Contents

- `core/experiment_ids.py`, `experiment_spec.py`, `runner.py`,
  `references.py`, `measurement.py`, `profile.py`, `series_analysis.py`,
  `series_store.py`;
- `tools/experiments.py`, `tools/parallelism.py`, `tools/experiment_plots.py`;
- `experiments/benchmark_small.json` and additional CPU/GPU/smoke configs;
- `notebooks/10_experiments_and_parallelism.ipynb`;
- `docs/EXPERIMENTS.md`, `VALIDATION_0.1.0a9.md`.

## Checks

```text
small series: 56 valid, 0 failed, 2 exact references
full profile: 84 points, 84 warmups, 252 measured, 336 calls
concurrent batch: 4 valid, 0 failed
mixed CPU/GPU: 3 valid, qaoa_gpu real, gpu_synchronized=true
```
