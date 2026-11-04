# Results and saving: read one run, export it without overwriting data

A completed or paused `solve()` returns a spin dictionary and leaves one
independent `RunRecord` behind. This page shows which attributes describe that
run, how to read them without rerunning anything, and how to export a result
with `save_result` so that no existing file is replaced.

## Read a result

```python
from QA_max_cut_lib import QAOACPU, WeightedGraph

graph = WeightedGraph(
    edges=[(10, 30, 1.5), (10, 80, 2.0), (30, 80, -0.5)],
    nodes=[10, 30, 80, 101],
)
solver = QAOACPU({
    "seed": 42,
    "parameters": {"layers": 1},
    "estimation": {"mode": "exact"},
    "initialization": {"type": "fixed_angles", "parameters": {"values": [-0.8, 0.4]}},
    "optimizer": {"parameters": {"max_steps": 2}},
    "readout": {"mode": "best_of_shots", "shots": 8},
    "limits": {"shots_total": 8},
})
solution = solver.solve(graph)

print(solution)               # {10: 1, 30: -1, 80: -1, 101: 1}
print(solver.cut_value)       # 3.5
print(solver.solution_energy) # -3.5
print(solver.status)          # completed
```

`solution` maps every original vertex label to `+1` or `-1`. No label is
renumbered and no vertex is dropped, including isolated vertices.

## Three views of the same run

| View | Type | Use |
|---|---|---|
| `solver.solve(graph)` | `dict` | the selected partition |
| `solver.<attribute>` | native Python/NumPy value | analysis and plotting |
| `solver.result` | JSON-compatible `dict` | logging, transport, inspection |
| `solver.last_run` | `RunRecord` | an immutable snapshot of the run |

Direct attributes such as `solver.probabilities` return native NumPy arrays.
`solver.result` encodes arrays as `{"__type__": "ndarray", "dtype": ..., "shape": ...}`
so the dictionary can be serialized without losing dtype or shape. Use
`tools.load_result(path)` to read an exported record back into a `RunRecord`.

A later `solve()` replaces `last_run` for that solver object. It never mutates a
record you already hold; `RunRecord` copies its fields on construction.

## Fields worth checking first

| Field | Meaning |
|---|---|
| `run_id` | unique identity of this run |
| `status` | `completed`, `paused`, `cancelled`, or `failed` |
| `termination_reason` | why the run stopped |
| `algorithm_type`, `package_version` | what produced the record |
| `config` | fully resolved configuration, including defaults |
| `graph` | graph snapshot, node order, and fingerprints |
| `solution`, `cut_value`, `solution_energy` | selected partition and its weight |
| `expected_cut`, `expected_energy` | state expectation, when defined |
| `optimal_angles` | final variational parameters |
| `probabilities` | basis-state probabilities in `node_order` |
| `resources` | shots, evaluations, optimizer steps, threads |
| `timing` | measured compute time |
| `data_status` | whether every requested field is available |
| `checkpoint_info` | checkpoint identity when checkpointing was enabled |

`expected_cut` and `cut_value` answer different questions: the first is a mean
over the final state, the second is the weight of one selected partition.
`data_status` is the place to look when a requested field is missing; the
library does not silently substitute zeros.

## Save a result

Saving is opt-in. The export selects attributes explicitly or defaults to every
collected field:

```python
saved = tools.save_result(solver, {
    "path": "results",
    "filename": "signed_example",
    "atts": ["cut_value", "solution", "expected_cut", "probabilities"],
    "save_original_data": ["probabilities"],
    "makedir": True,
    "savedir": True,
})
print(saved.primary)
```

With `savedir=True` and `filename="signed_example"` the exporter creates

```text
results/signed_example/
  signed_example.json
  signed_example_probabilities.npy
```

The JSON file is the completion marker and is published last. The `.npy`
sidecars keep the original dtype and shape; integer graph labels stay integers
in the JSON record. A sidecar is written only for a name listed in
`save_original_data`, and only for a numeric NumPy array.

### Save fields

| Field | Required | Meaning |
|---|---|---|
| `path` | yes | destination directory |
| `filename` | yes | base name without a path or reserved characters |
| `atts` | no | result fields copied into the JSON document; default is all collected fields |
| `save_original_data` | no | numeric arrays written as separate `.npy` files |
| `makedir` | no | create a missing `path`; default `False` |
| `savedir` | no | create one exclusive folder named `filename`; default `False` |

An unknown attribute, a duplicate name, or an array with an object dtype raises
`ValueError` before publication. The exporter uses temporary files in the
destination directory and publishes each file with an exclusive hard link.

## No-overwrite behavior

If the target file or the `savedir` folder already exists, the export raises
`FileExistsError` and leaves existing data untouched. There is no automatic
rename, merge, or version suffix. To keep a second result, choose a new
`filename` or a new `path`.

This behavior is deliberate: a silent replacement would make two different runs
indistinguishable on disk. It also means that re-running a script that saves to
the same name fails on the second run; that failure is the guarantee, not a bug.

Check the export by reading it back:

```python
record = tools.load_result("results/signed_example/signed_example.json")
print(record.status, record.cut_value, record.solution)
```

Pass `load_original_data=True` to `tools.load_result` when the sidecar arrays are
needed; by default the descriptors are returned without reading the `.npy`
files.

## Result, series, and checkpoint are different artifacts

| Artifact | Produced by | Contains |
|---|---|---|
| Result record | one `solve()` or `save_result` | one partition, its configuration, resources, timing |
| Experiment series | `ExperimentRunner.run()` | many trial and reference records plus an analysis |
| Checkpoint | `checkpoint.enabled` | resumable optimizer state, RNG state, budget journal |

A checkpoint is not a scientific result and does not replace one. A saved
result cannot be resumed: it stores the finished record, not the live optimizer
state. The experiment manifest is versioned rather than overwritten; see
`EXPERIMENTS.md`.

## Limitations

- The JSON exporter writes one primary file plus `.npy` sidecars; CSV, Parquet,
  and image export are not part of `save_result`.
- Object-dtype arrays are rejected; only numeric arrays are supported.
- The export requires a filesystem that supports hard links.
- `save_result` does not create a directory tree beyond one `path` level with
  `makedir=True` and one `savedir` folder.
