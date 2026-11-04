# Experiment series: expand a plan, run it, and profile CPU threads

`ExperimentRunner` and `ParallelismProfiler` run many solver trials from one
declarative `BenchmarkSpec`. They call the existing solvers through
`SolverFactory`; no algorithm is reimplemented in the experiment layer.

This page shows how to inspect the plan before any work starts, read the series
summary, and measure whether extra `num_treads` values actually help. By the end
you can reproduce the 56-trial small benchmark and a three-candidate thread
sweep.

## Plan first, then run

```python
from QA_max_cut_lib import ExperimentRunner, ParallelismProfiler, tools

spec = tools.load_json("experiments/benchmark_small.json")
plan = tools.plan_experiments(spec)
print(plan.summary)
# graphs=2 cases=28 trials=56 rejected=0 references=2

series = ExperimentRunner(spec).run()
print(series.summary)
# planned=56 valid=56 failed=0 skipped=0 success(valid)=0.6607142857142857 wall=...

print(series.analysis["by_solver"]["qaoa_mean"]["planned"])   # 16
```

`plan_experiments` only expands the specification: graph recipes, solver grids,
seeds, reference jobs, and resource upper bounds. It does not call a solver and,
without `save`, writes no files.

The wall time in `series.summary` is machine-dependent. The counts and the
success fraction in the committed example are stable with the fixed seeds.

## What `BenchmarkSpec` expands

`BenchmarkSpec` validates a closed set of top-level fields and expands in this
order:

1. expand `graph_cases` along explicit `grid` axes;
2. expand `solver_cases` independently along their axes and `overrides`;
3. validate every pair of graph instance and accepted solver config through the
   real solver `_validate`;
4. take the Cartesian product of graph instance, solver config, and trial seed;
5. assign stable `case_id` and `trial_id` values from a SHA-256 of canonical
   JSON;
6. add reference jobs according to the selected policy.

An unknown field, a repeated or overlapping axis, a missing `grid` path, a
duplicate graph instance, or a `planner_limits` overflow blocks the plan. An
incompatible individual combination is recorded in `plan.rejected` and is not
run. Each rejection names the graph case, solver case, config, error type, and
reason.

Axes are explicit only:

```json
{"path": ["parameters", "layers"], "values": [1, 2]}
```

A continuous interval is not an axis. Without a finite list of values it is not
interpreted as "try every value".

## Identifiers

| Id | Covers |
|---|---|
| `graph_content_id` | canonical vertices, edges, exact weights, isolates, schema version; insertion order does not matter |
| `graph_instance_id` | content plus `node_order`; a statevector cannot be reused across a permutation |
| `case_id` | graph instance and scientific configuration without `seed`, `num_treads`, output/save, checkpoint, report/logs, or profiled CPU budget |
| `trial_id` | case, algorithm seed, role, repetition, `num_treads`, measurement protocol, implementation version |
| `attempt_id` | one execution attempt; a retry is visible, not hidden |
| `reference_id` | graph, reference method and settings, seed, and meaning of the value |

Dictionary key order and process completion order do not change semantic ids.
Python `hash()`, process ids, and CSV row numbers are not used.

## Execution modes

`execution.mode` selects how trials share the machine:

- `isolated` — one measured solver trial is active at a time;
- `concurrent_batch` — independent trials run in a process pool
  (`backend="process"`, `spawn`) or in threads; the number of active trials and
  `cpu_slots` are explicit limits.

`ProcessPoolExecutor` needs an ordinary importable script guarded by
`if __name__ == "__main__":`. In Jupyter, use an importable CLI subprocess
instead of a notebook cell.

`num_treads` is the CPU budget of one solver. When an implementation is
sequential, the record stores `requested_num_treads` greater than
`effective_num_treads` and sets `capability_limited=true`. The runner does not
change the backend to another algorithm and does not report a fictitious
speedup.

## References

`ReferencePolicy` supports a tiered, exact, GW, best-known, or no-reference
policy. The reference kinds are:

| Kind | Meaning |
|---|---|
| `exact_max_cut` | proven full maximum after `exhaustive_search` |
| `exact_min_cut` | a separate `ExactCPU` run on the graph with weights `-w` |
| `gw_rounded_cut` | heuristic feasible GW cut |
| `best_known_cut` | external value keyed by `graph_content_id`, with provenance |

`exact_min_cut` is obtained as `C_min(G) = -C_max(-G)`. `q_range` needs both
proven extrema; for a constant objective it is `null` with a reason. A GW value
is not announced as a strict SDP bound, and `sdp_upper_bound` stays `null`
without a dual certificate.

Exact references are bounded by `exact.max_vertices` (default `0`, which means
no exact reference). GW references are bounded by `gw.max_vertices` (default
`0`). The `best_known` section accepts only entries with `graph_content_id` and
a finite `value`.

## Parallelism profile

`ParallelismProfiler` takes the same `BenchmarkSpec`, one selected seed, and all
admissible cases. The `num_treads_values` axis adds the profiler candidates;
`1` is mandatory.

```python
spec = tools.load_json("experiments/benchmark_small.json")
config = tools.load_json("tools/parallelism_sweep.json")
profile = ParallelismProfiler(spec, config).run()

print(profile.recommended_num_treads)          # 1
print(profile.analysis["recommendation"]["search_complete"])  # True
```

For each case and candidate the profiler runs warmups and repetitions. A warmup
is never used as a timing sample. Candidate order within a repetition can be
shuffled by a separate RNG. The committed sweep uses `warmup_runs=1`,
`repetitions=3`, and candidates `[1, 2, 4]`.

The result contains:

- `measurements` and `warmups`, each with requested/effective resources and work
  counters;
- `case_curves`, `aggregate`, and `per_solver`;
- an Amdahl fit and a Karp–Flatt `f_eff`, without clipping negative or large
  values;
- a paired-block bootstrap when explicitly requested;
- `recommendation` with `raw_argmin`, `recommended_num_treads`, `tolerance_rel`,
  `search_complete`, `best_observed_complete`, and a reason.

An incomplete candidate is excluded from the global recommendation; its time is
not replaced by zero. `best_observed_complete` is not called the optimum of all
requested values.

## Saving a series

`save` uses the common schema `{path, filename, atts, save_original_data,
makedir, savedir}`. The coordinator is the only writer. Each trial and reference
is published as a separate JSON document with `export_complete=true`; a short
name is checked against the full id.

The summary manifest is versioned, not replaced: the first file is
`<filename>_summary.json`, a repeat creates `<filename>_summary_v2.json`, then
`_v3`, and so on. Without `save`, no folders or files are created.

`ExperimentRunner(..., reuse_completed=True)` reuses only completed records whose
trial id, measurement protocol, and environment id match. A reused record is not
a new timing repeat.

## Limitations

- The profiler records `execution_mode`, but this release executes the profile
  grid sequentially; a truly concurrent profile requires a later extension.
- Concurrent profiling on one GPU is not supported.
- A full GPU profile is limited by available VRAM and must be checked
  separately.
- A series does not turn a heuristic reference into an optimality proof; read
  `reference_kind` and `optimality_gap` together.
