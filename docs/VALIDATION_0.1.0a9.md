# Verification 0.1.0a9

Date: 2 November 2026.

## What was verified

- `BenchmarkSpec` and the full product: 2 graphs, 28 cases, 56 scientific
  trials, 2 exact references; `alpha` does not appear in mean/SA; topology and
  weight seeds are stored separately.
- `ExperimentRunner`: the small `small_comparison` series — 56 valid, 0
  failed; exact maximum for the graphs 8/10; C/energy recomputed from the full
  assignment.
- `ReferencePolicy`: exact max/min, GW heuristic, best-known source and a
  separate `ReferenceRecord`.
- `ParallelismProfiler`: full grid of 84 case/candidate points, 84 warmups and
  252 measured calls; 1/2/4; work counters, Karp–Flatt, Amdahl fit, paired-block
  bootstrap and a global recommendation.
- `concurrent_batch`: 4 independent trials, 0 failed, process backend.
- mixed CPU/GPU: real `qaoa_gpu` on the RTX 3050, `gpu_synchronized=true`.
- save/write-once, summary versioning, completed reuse, no-solver plot.

## Numbers

```text
CPU tests: 141 passed, 1 skipped, 1 warning
GPU tests: 10 passed
small series: 56 valid / 56 planned; 2 exact references
profile: 84 points; 336 calls; recommended_num_treads=1
```

The small profile includes QAOACPU (sequential, effective=1) and SA; therefore
the overall curve is flat or degrades at 2/4. This is not a claim of speedup
for a sequential implementation. The SA profile with 1/2/4 was measured
separately.

## Limitations

- The concurrent profiler was not run on a single GPU.
- RQAOA/LS/hierarchy/GPU checkpoint were not part of a9.
- The GPU profile is not part of the CPU notebook.
