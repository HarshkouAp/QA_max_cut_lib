# Troubleshooting: from an error message to a cause

Use this page when an install, a solve, a save, a checkpoint, or a documentation
build fails. Each entry names the symptom, the cause, and the smallest check that
distinguishes it from a neighboring problem.

## The version or the example is wrong

Check the build before debugging anything else:

```bash
python -c "import QA_max_cut_lib; print(QA_max_cut_lib.__version__)"
python examples/public_quickstart.py
```

The version must be `0.1.0rc2`; the example must print `3.500000`, `3.044365`,
`3.500000`, and `0.805755`. If the version differs, an older wheel is installed
in front of the source tree. Reinstall from the repository root with
`python -m pip install -e ".[cpu]"` and confirm that `python -c "import
QA_max_cut_lib; print(QA_max_cut_lib.__file__)"` points at the source you expect.

## Import failures

| Symptom | Cause | Check |
|---|---|---|
| `ModuleNotFoundError: numpy` or `scipy` | core dependencies are missing | `python -m pip install -e .` |
| `ModuleNotFoundError: cupy` during a CPU run | a GPU solver or GPU profile was selected | install only `.[cpu]`, or install the matching `gpu-cu12`/`gpu-cu13` profile |
| `ModuleNotFoundError: cvxpy` | `gw_cpu` or another classical baseline was selected | install `.[cpu,classical]` |
| `ModuleNotFoundError: matplotlib` while plotting | the `cpu` extra is missing | `python -m pip install -e ".[cpu]"` |

A CPU install must not require CUDA. If importing `QA_max_cut_lib` itself raises
a CuPy error, the environment has a broken or partially installed CuPy package;
remove it and use the CPU profile, or reinstall exactly one GPU profile.

## Solver selection and configuration

### `Solver '...' is not implemented in 0.1.0rc2`

The `type` string is unknown or belongs to a class that is not registered in
this build. Print the registry:

```python
from QA_max_cut_lib import SolverFactory
print(SolverFactory.available())
```

Use one of the listed names. The factory never substitutes a similar algorithm.

### `Unknown solver fields: {...}`

The configuration contains a key the selected class does not accept. Remove it,
or use the correct class-specific field. A misspelled `optimizer` or `readout`
sub-field is reported by that section's own validator, not by this message.

### `Parallel CPU execution is not implemented for this solver; use num_treads=1.`

`num_treads` above `1` was passed to a sequential solver. Use `1`. This does not
mean the BLAS library is single-threaded; native threads are controlled by
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS`.

### `RunControl is not supported by ...`

The component has no cooperative pause boundary. `RunControl` is accepted by
components with a declared checkpoint codec: `QAOACPU`,
`laboratory.warm_start_qaoa_cpu`, `laboratory.qaoa_mi_cpu`,
`laboratory.block_mean_field_qaoa_cpu`, and
`laboratory.distributed_ls_rqaoa_cpu`. The capability matrix in
`CHECKPOINTS.md` lists the exact boundaries.

## The result looks wrong

1. Confirm the seed and the graph. A fixed `seed` makes the run reproducible,
   but a different node order changes the statevector indexing.
2. Check `graph.cut_value(solution)` instead of trusting a hand-written cut.
   Vertex labels are external integers; they are not positions in the array.
3. Confirm `readout.mode` and `limits.shots_total`. With `best_of_shots`, the
   selected cut depends on the measured sample; a smaller budget can return a
   worse partition.
4. Compare `expected_cut` with `cut_value` only as different quantities: the
   first is a state mean, the second is one partition.
5. Look at `data_status`. A field that could not be produced is marked
   unavailable; it is not replaced by zeros.

### A paused run reports a cut

A paused `QAOACPU` run can already expose a best-so-far solution. The run is not
finished: `status` is `paused`, `readout_completed` is `False`, and the readout
shots are still reserved. Resume with `restore(path).resume()` and read the
result after completion.

## Saving fails

| Message | Cause | Fix |
|---|---|---|
| `Export folder already exists: ...` | `savedir=True` and the folder exists | choose a new `filename` or `path` |
| `Export targets already exist: ...` | one of the files exists | choose a new name; there is no overwrite mode |
| `FileNotFoundError` on `save.path` | the parent directory does not exist and `makedir` is `False` | create it or set `makedir=True` |
| `Unknown result attribute: ...` | `atts` or `save_original_data` names a field the record does not have | inspect `solver.result` keys first |
| `... is not an available numeric NumPy array.` | `save_original_data` names a scalar, list, or object array | save it through `atts`, or select a numeric array |

A missing `.npy` sidecar usually means the name was listed in `atts` but not in
`save_original_data`. The arrays are opt-in separately from the JSON attributes.

## Checkpoint and resume

### `Checkpoint/resume is not implemented for ... in 0.1.0rc2`

`checkpoint.enabled` was set for a component without a state codec. The
capability matrix in `CHECKPOINTS.md` is authoritative; enabling the section
does not silently degrade to a partial snapshot.

### `restore(path)` succeeds but the run does not continue

`restore()` returns an object; it does not resume by itself. Call `resume()`.
Calling `solve(graph)` on a restored object starts a new run with a new
`run_id`, which is the documented behavior for a fork.

### Restore rejects a changed configuration

A strict resume compares the graph identity, `node_order`, source data, and the
scientific parameters stored in the checkpoint. Changing `epsilon`, the
warm-start source, the mixer, or the graph prefix is a new scientific task, not
a continuation. Create a new run instead of editing the checkpoint or the
configuration prefix.

### A run paused, then the process died

The last published generation plus the durable resource journal is the recovery
point. A `BEGIN` entry without `COMMIT` or `RESOLVE` is charged conservatively at
its upper bound and closed by a `RESOLVE` record on recovery. This prevents
under-counting the shot budget, but it does not promise exactly-once physical
measurements; the charged amount is a conservative accounting value.

## Experiments and plans

### `completion_status=coverage_shortfall`

The plan does not cover every parent edge. Inspect it explicitly:

```python
from QA_max_cut_lib import tools

analysis = tools.analyze_plan(graph, plan)
print(analysis["completion_status"])
print(analysis["uncovered_edges"])
```

`coverage_shortfall` is not full coverage. Increase the task limit, change the
partition recipe, or report the uncovered edges as a limitation of the series.

### A plan is rejected before running

Typical causes are an unknown top-level field, an axis that is not an explicit
`{"path": [...], "values": [...]}` pair, a missing key in `path`, a duplicate
graph instance, or a `planner_limits` overflow (`max_cases` default `50000`,
`max_trials` default `200000`). The message names the failing section; fix the
spec rather than catching the exception.

### `concurrent_batch` fails under Jupyter

`ProcessPoolExecutor` uses `spawn` and needs an importable script guarded by
`if __name__ == "__main__":`. A notebook cell is not importable. Run the series
through a CLI subprocess, or use `execution.mode="isolated"`.

### A timing comparison is not reproducible

Warmups are not timing samples. Confirm that `warmup_runs` is at least `1`, that
the same `num_treads_values` are used, and that the BLAS thread count is fixed
before Python starts. A reused record with `reuse_completed=True` is not a new
repeat; it is excluded from new timing evidence.

## GPU failures

### `cupy.cuda.runtime.CUDARuntimeError`

No usable CUDA device or driver for the installed CuPy profile. Check the device
with the command in `INSTALLATION.md`; a CPU solver remains available.

### `GPU memory preflight rejected N=...`

The estimated state plus workspace exceeds the free-memory fraction. Reduce
`parameters.max_vertices`, use `representation="spin_flip_pairs"`, use
`complex64`, or free memory held by other processes. The preflight counts this
run's large arrays; it does not treat another process's CuPy pool as exclusive.

### `GPU QAOA requires 1..max_vertices nodes.`

The graph is larger than `parameters.max_vertices` (default `30` for `QAOAGPU`,
default `20` for the GPU correlation source; the hard cap is `30`). Reduce the
graph or raise the limit only within available memory.

## Documentation build

A Sphinx warning is an error for this project:

```bash
python -m sphinx -W -b html docs docs/_build/html
```

Read the first warning, not the last: a missing reference or a duplicate label
usually produces a cascade. `docs/_build/html` is disposable; delete it before a
clean rebuild if an old generated file confuses the output.
