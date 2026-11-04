# Verification 0.1.0a10

Date: 2 November 2026.

## Actual base

- Python 3.12.14, Windows 10; CPU 4 physical / 8 logical cores;
- NumPy 2.5.4, SciPy 1.18.1, CVXPY 1.9.3, SCS 3.3.1,
  `dwave-samplers 1.8.0`, `dimod 0.12.23`;
- GPU profile: NVIDIA GeForce RTX 3050 Laptop, CC 8.6, 4096 MiB,
  CuPy 14.2.0, driver 591.74;
- wheel `0.1.0a10` installed into `.venv` and `.venv-wheelcheck` outside
  `src`. `examples/checkpoint_resume.py` and the `tools.checkpoints` CLI were
  run from the installed wheel.

## Mandatory checks

```text
CPU: 151 passed, 1 skipped, 1 warning
GPU: 10 passed
Sphinx -W: success
wheel: QA_max_cut_lib-0.1.0a10-py3-none-any.whl
sdist: QA_max_cut_lib-0.1.0a10.tar.gz
```

## A. QAOA/ADAM pause and resume

Control p=1, `max_steps=4`, `M_train=32`, `M_readout=64`, `B=704`.

| Point | iterations | known training | held readout | available |
|---|---:|---:|---:|---:|
| pause after 2 | 2 | 320 | 64 | 320 |
| resumed completed | 4 | 640 | 0 | 0 |

After the pause `status="paused"`, `readout_completed=false`;
`inspect_checkpoint` shows `phase="training"` and
`next_operation={"optimizer_iteration": 3}`.
`restore(path).resume()` on a new object produced:
- the same canonical `run_id`;
- identical `optimal_angles`, complete assignment, optimizer history and
  `objective_evaluations=20`;
- final `shots_used=704`.

Continuous a10 and resumed a10 agreed for p=1/p=2, exact and shots, with
cvar/mean and with early gradient stop. `best_evaluation_id` is stored
separately. The check was performed on a real `QAOACPU`, not on a JSON
round-trip.

## B. Durable journal and a lost attempt

After the pause (known=320, held=64) a BEGIN for the third iteration with
upper=160 was written to the journal without a COMMIT.

```text
known_spent=544
uncertain_upper_charge=160
total_charged=704
readout_shots=64
optimizer_steps=3
status=stopped
```

The fourth iteration does not fit. Repeated reconciliation after RESOLVE does
not double the 160. The number 704 is a conservative charge, not a proven
physical number of shots of the unknown attempt. The readout token is reserved
once; repeating a completed generation does not make new measurements.

## C. Readout and max_steps

- `max_steps=0`: zero optimizer steps, initial angles are preserved.
- `gradient_tolerance`: phase `training_done`, the update is not performed
  again.
- checkpoint after `readout_done`: `resume()` does not start the solver and
  does not add a generation.
- a new `solve()` on a restored object has a new `run_id`.

## D–G. Format, locking, corruption

- `json_npy_v1`: manifest + state.json + `theta/moment/variance/best_theta`
  `.npy`; SHA-256, dtype/shape/size are checked before use.
- `latest.json` is updated after an immutable generation; older generations
  remain.
- Substitution or corruption of a new payload raises `CheckpointCorrupt`; the
  old generation is still readable.
- A second writer of the same managed run gets `CheckpointLocked`; after
  release the lock can be taken again.
- A copy of the generation tree in another directory is restored; the
  `checkpoint.path` is recomputed from the actual tree.

## Capability matrix

| Component | a10 |
|---|---|
| `QAOACPU` | full codec: ADAM iteration, training_done, readout_done |
| `QAOAGPU` | rejected; a portable CuPy sampler protocol is required |
| `RQAOACPU`, `RQAOAGPU` | rejected; source/reduction codec not implemented |
| `LSRQAOACPU`, `LSRQAOAGPU` | rejected; child task codec not implemented |
| `QAOAInQAOACPU`, `QAOAInQAOAGPU` | rejected; level/local-pattern codec not implemented |
| `ExactCPU`, `GW`, `SA`, `Refiner`, `Refined` | rejected explicitly |
| `ExperimentRunner` | completed reuse works; paused child codec not claimed |

## Stage 09 re-verified on a10

```text
small series: 56 planned / 56 valid / 0 failed; 2 exact refs
profile: 84 points, 84 warmups, 252 measured, 336 calls
concurrent batch: 4 valid, 0 failed
mixed CPU/GPU: 3 valid, qaoa_gpu completed, gpu_synchronized=true
```

## Known limitations

- RQAOA/LS/hierarchy/Exact/GW/SA/refiner resume are not implemented; enabling
  checkpoint for them is rejected.
- GPU checkpoint is not claimed; CuPy random state is not serialised like
  NumPy.
- There is no distributed ownership and no exactly-once physical measurement.
- No incremental/dedup generations; retention is `keep_all` only.
- Campaign paused child codec is not implemented; completed trials are reused.
- Checkpoint I/O and recovery rebuild are not part of active compute time.
