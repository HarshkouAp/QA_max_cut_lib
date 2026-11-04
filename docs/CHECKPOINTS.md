# Checkpoints: pause a run, inspect it, and resume the same run

QA_max_cut_lib `0.1.0rc2` can checkpoint a supported CPU solver at declared
phase boundaries, publish an immutable generation to disk, and later continue
the same canonical run. This page shows the exact call sequence, what is inside
a checkpoint, which components support it, and how the shot journal is charged
after a crash.

By the end you can pause a QAOA run after two optimizer iterations, inspect its
manifest without loading CUDA, and resume it to completion with the same
`run_id`.

## Four ways to enter a run

| Call | Behavior |
|---|---|
| `solver.solve(graph)` | a new independent run; new `run_id`, new checkpoint prefix |
| `solver.solve(graph, control=RunControl(...))` | cooperative pause at a supported boundary |
| `restore(path).resume()` | continue the same canonical run |
| `restore(path).solve(graph)` | a new run; restore alone never resumes |
| crash recovery | last published generation plus the durable journal; no exactly-once promise |

## Pause, inspect, resume

```python
from QA_max_cut_lib import (
    GraphFactory, RunControl, SolverFactory, inspect_checkpoint, restore, tools,
)

recipe = tools.load_json("configs/graphs/warm_start_4.json")
graph = GraphFactory.from_recipe(recipe)

config = tools.load_json("configs/solvers/qaoa_cpu_checkpoint.json")
solver = SolverFactory.from_config(config)

spins = solver.solve(
    graph,
    control=RunControl(pause_after={"optimizer_iterations": 2}),
)
assert solver.result["status"] == "paused"

path = solver.last_checkpoint.path
summary = inspect_checkpoint(path)
print(summary["generation"])          # 2
print(summary["phase"])               # training
print(summary["next_operation"])      # {'optimizer_iteration': 3}

continued = restore(path)
final_spins = continued.resume()
print(continued.result["status"])     # completed
print(continued.result["resources"]["shots_used"])  # 704
```

The committed `qaoa_cpu_checkpoint.json` uses `layers=1`, `max_steps=4`,
`shots_per_evaluation=32`, `readout.shots=64`, and `limits.shots_total=704`.
With one checkpoint per optimizer iteration, pausing after iteration 2 leaves
generation `2` and points at iteration `3`.

`inspect_checkpoint(path)` reads the manifest, schema, generation, phase, next
operation, array headers and checksums, and the journal cutoff. It does not
construct a solver and does not create a CUDA context.

The command-line equivalent is:

```bash
python -m QA_max_cut_lib.tools.checkpoints inspect --checkpoint PATH
python -m QA_max_cut_lib.tools.checkpoints resume --checkpoint PATH --report 1
```

## What a checkpoint directory contains

For `checkpoint.path="results/checkpoints/qaoa_control"` and one run, the layout
is:

```text
results/checkpoints/qaoa_control/
  <run_id>/
    latest.json
    owner.lock
    resource_journal.jsonl
    gen_000001/
      manifest.json
      state.json
      theta.npy
      moment.npy
      variance.npy
      best_theta.npy
```

`latest.json` points at the newest complete generation. The generation directory
is created by an atomic rename only after every payload is flushed and the
manifest is written. Older generations are never modified.

## Supported components and boundaries

| Component ID | Public class | Declared boundaries |
|---|---|---|
| `qaoa_cpu` | `QAOACPU` | completed ADAM iteration, `training_done`, `readout_done` |
| `laboratory.warm_start_qaoa_cpu` | `WarmStartQAOACPU` | same boundaries; frozen source/q/epsilon/mixer |
| `laboratory.qaoa_mi_cpu` | `QAOAWithMIObserverCPU` | same boundaries; MI is recomputed passively |
| `laboratory.block_mean_field_qaoa_cpu` | `BlockMeanFieldQAOACPU` | complete sweep commit |
| `laboratory.distributed_ls_rqaoa_cpu` | `DistributedLSRQAOACPU` | `initialized`, `reduction_committed` |

Everything else is rejected: `RQAOACPU`, `LSRQAOACPU`, `QAOAInQAOACPU`,
`ExactCPU`, `GoemansWilliamsonCPU`, `SimulatedAnnealingCPU`, `RefinedSolver`,
`ExactBlockRefiner`, the GPU solvers, and the Laboratory controllers without a
declared codec. Enabling `checkpoint.enabled` for such a component raises
`ValueError` (`Checkpoint/resume is not implemented for ...`), not a silent
partial snapshot. An unknown plugin without a codec is rejected the same way.

`WarmStartRQAOACPU`, `ConditionalQAOACPU`, `EnvironmentLSRQAOACPU`,
`IncrementalLSRQAOACPU`, `BatchLSRQAOACPU`, and `RQAOAWithMIObserverCPU` have no
declared durable codec in this release.

## What is saved for QAOACPU

- `run_id`, generation and parent, component id, package version;
- graph snapshot, node order, and content/instance fingerprints;
- resolved configuration, limits, and checkpoint policy;
- `theta`, `m`, `v`, `best_theta`, `best_loss`, and evaluation ids;
- `evaluations`, `optimizer_steps`, and training-shot counters;
- optimizer and objective histories within the configured collection scope;
- the full `numpy.random.Generator` state (PCG64 JSON state);
- `budget_total`, `budget_used`, and the readout reservation/result;
- phase, `next_iteration`, and consumed control flags;
- the durable journal cutoff.

Live Python objects, a statevector that can be reconstructed from angles, CUDA
pointers, CuPy RNG objects, pools, and locks are not saved.

## Boundaries and numerical exactness

A checkpoint is taken only at the end of a full ADAM iteration: all `4p+1`
central-difference estimates for `p` layers have finished, the convergence
decision has been made, and `m`, `v`, and `theta` are updated atomically. A
checkpoint never contains a mixture of a new `m` and an old `theta`.

`max_steps` bounds the whole run, not each resume. After two iterations of a
four-iteration budget, two remain; a resume does not receive four more.

`best_theta` may come from a gradient probe rather than from the last central
point. It is stored separately from `theta` for exactly that reason. When
`gradient_tolerance` stops training, no parameter update is applied; the phase
becomes `training_done`, not "repeat this iteration".

## Shot accounting across a pause

For `p=1`, `max_steps=4`, `M_train=32` per evaluation, and `M_readout=64`:

| Boundary | iterations | accepted training shots | held readout | available |
|---|---:|---:|---:|---:|
| before training | 0 | 0 | 64 | 640 |
| pause after 2 | 2 | 320 | 64 | 320 |
| `training_done` | 4 | 640 | 64 | 0 |
| completed | 4 | 640 | 0 | 0 (spent 704) |

The final `best_of_shots` readout reserves 64 shots once. A resume restores that
reservation; re-reading a completed generation does not perform new
measurements. If a readout started and its result was lost, repeating it needs a
new resource; it is not relabeled as a successful readout.

## Publication format

The format is `json_npy_v1`: a strict `manifest.json`, a `state.json`, and
numeric `.npy` sidecars written with `allow_pickle=False`. Publication proceeds
as follows:

1. create a staging directory next to the generation on the same filesystem;
2. write payload files and flush/fsync them;
3. write the manifest with SHA-256, sizes, dtype, and shape;
4. atomically rename staging to `gen_000NNN`;
5. atomically replace `latest.json`; the `latest-index` belongs to the manager.

A failure while writing staging or the manifest leaves the previous generation
in place. `latest.json` is not the only source of truth: the loader checks the
manifest and checksums. `os.replace` is not claimed as a universal transaction
on every network filesystem; supported local durability is `local_fsync` on a
tested local filesystem.

## Resource journal and crash charging

`ResourceJournal` stores append-only `BEGIN`, `COMMIT`, and `RESOLVE` records
with fsync. On recovery, a `BEGIN` without `COMMIT` or `RESOLVE` is charged
conservatively at its upper bound and closed with a `RESOLVE` record. Repeated
reconciliation does not charge the same operation twice. Parent and child
reservations are not summed: the ledger has one owner.

A recorded crash scenario:

1. pause after two iterations: 320 known, 64 held, 320 available;
2. crash with `BEGIN` 160 and no result;
3. resume: `known_spent=544`, `uncertain_upper_charge=160`, `total_charged=704`,
   `status="stopped"`, `optimizer_steps=3`.

The `704` is a conservative charge, not proof of the number of physical
measurements performed by the unknown attempt.

## Limitations

- RQAOA, LS-RQAOA, hierarchy, exact enumeration, GW, SA, and refiner components
  have no checkpoint codec.
- GPU checkpointing is absent: a CuPy random state cannot be serialized as a
  NumPy state; a portable operation-addressed sampler is required first.
- There is no distributed ownership and no exactly-once guarantee for physical
  measurements.
- Generations are not deduplicated and retention is `keep_all` only.
- A campaign reuses completed child records, but a paused child codec is not
  claimed.
- Checkpoint I/O time and recovery rebuild are accounted separately from active
  compute; wall-time budgets with checkpoint overhead are checked separately.
