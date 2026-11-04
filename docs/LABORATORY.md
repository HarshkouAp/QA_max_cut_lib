# Laboratory: read the catalog and run a warm-start QAOA

The Laboratory is the reviewed bridge from the historical research archive to
the current public API. This page shows how to inspect the versioned catalog
without importing CUDA or running a solver, which variants are runnable in
`0.1.0rc2`, and how to run the warm-start product-state QAOA with a checked
shot budget.

By the end you can list the catalog, run
`laboratory.warm_start_qaoa_cpu` on a four-vertex graph, and verify that the run
uses 544 shots rather than a new budget on every pause.

## Read the catalog

```bash
python -m QA_max_cut_lib.laboratory list
python -m QA_max_cut_lib.laboratory list --implemented-only
python -m QA_max_cut_lib.laboratory show --id warm_start_product_qaoa_cpu_v1
```

`list` and `show` read versioned JSON records. They do not import CuPy or CVXPY
and do not construct or run a solver.

Current catalog counts:

| Item | Value |
|---|---|
| Archive SHA-256 | `b4c684a5d4da469adc25bd99a405e402892d1a2524465e4c97ab6229c71a4aed` |
| Python members in the archive | 89 |
| Parsed members | 89 |
| Variant cards | 15 |
| Runnable variants | 10 |

The data files are:

```text
src/QA_max_cut_lib/laboratory/data/inventory.json
src/QA_max_cut_lib/laboratory/data/catalog.json
src/QA_max_cut_lib/laboratory/data/inventory-v1.schema.json
src/QA_max_cut_lib/laboratory/data/catalog-v1.schema.json
```

`source_to_variant` covers all 89 records. Every variant card is runnable:
historical LS/MK, mean-map, correlation, edge-extension, decomposition, scanning,
and QAOA variants map to `laboratory.legacy` adapters; the remaining controllers
use their dedicated assemblies. No card remains in `deferred_variant_ids`.

## Runnable variants

| Variant id | Public class / type | Checkpoint | Chapter |
|---|---|---|---|
| `warm_start_product_qaoa_cpu_v1` | `WarmStartQAOACPU` / `laboratory.warm_start_qaoa_cpu` | ADAM boundaries | this page |
| `warm_start_ensemble_rqaoa_cpu_v1` | `WarmStartRQAOACPU` / `laboratory.warm_start_rqaoa_cpu` | not declared | `WARM_START_RQAOA.md` |
| `conditional_fields_cpu_v1` | `ConditionalQAOACPU` / `laboratory.conditional_qaoa_cpu` | not declared | `CONDITIONAL.md` |
| `block_mean_field_gauss_seidel_cpu_v1` | `BlockMeanFieldQAOACPU` / `laboratory.block_mean_field_qaoa_cpu` | complete sweep | `BLOCK_MEAN_FIELD.md` |
| `block_mean_field_jacobi_cpu_v1` | `BlockMeanFieldQAOACPU` / `laboratory.block_mean_field_qaoa_cpu` | complete sweep | `BLOCK_MEAN_FIELD.md` |
| `distributed_ls_rqaoa_carry_cpu_v1` | `DistributedLSRQAOACPU` / `laboratory.distributed_ls_rqaoa_cpu` | outer reduction | `DISTRIBUTED_LS.md` |
| `distributed_ls_rqaoa_cold_cpu_v1` | `DistributedLSRQAOACPU` / `laboratory.distributed_ls_rqaoa_cpu` | outer reduction | `DISTRIBUTED_LS.md` |
| `qaoa_mi_observer_cpu_v1` | `QAOAWithMIObserverCPU` / `laboratory.qaoa_mi_cpu` | ADAM boundaries | `MUTUAL_INFORMATION.md` |
| `rqaoa_mi_observer_cpu_v1` | `RQAOAWithMIObserverCPU` / `laboratory.rqaoa_mi_cpu` | not declared | `MUTUAL_INFORMATION.md` |
| `incremental_ls_mk_components_v1` | `IncrementalLSRQAOACPU` / `laboratory.incremental_ls_rqaoa_cpu` | not declared | `INCREMENTAL_LS.md` |

Two additional public Laboratory types, `laboratory.batch_ls_rqaoa_cpu`
(`BatchLSRQAOACPU`) and `laboratory.quantized_qaoa_cpu` (`QuantizedQAOACPU`),
are available through `SolverFactory` and have solver configs in
`configs/solvers/`. They do not have catalog cards.

## Warm-start product QAOA

The variant prepares a product state from a classical or supplied start, then
trains the usual QAOA angles with a fixed preparation:

\[
q_i=\frac{1-s_i^{(0)}}{2},\qquad
\tilde q_i=\operatorname{clip}(q_i,\varepsilon,1-\varepsilon).
\]

For `provided_probabilities`, the input is already the probability of bit `1`
and is clipped only; no affine conversion is applied. Inputs outside `[0,1]`,
NaN, and infinity are rejected before clipping.

\[
\alpha_i=2\arcsin\sqrt{\tilde q_i},\qquad
|\phi_0\rangle=\bigotimes_i
\left(\sqrt{1-\tilde q_i}|0\rangle+\sqrt{\tilde q_i}|1\rangle\right).
\]

`alpha` is a fixed preparation angle, not a trainable gamma or beta. The
default `epsilon` is `0.25`; the accepted range is `[0, 0.5]`.

Two parameterized mixers are available:

\[
U_i^{\text{aligned}}(\beta)=R_Y(\alpha_i)R_Z(-2\beta)R_Y(-\alpha_i)
=\cos\beta I+i\sin\beta(\cos\alpha_i Z+\sin\alpha_i X),
\]

\[
U_i^{\text{modified}}(\beta)=R_Y(-\alpha_i)R_Z(-2\beta)R_Y(\alpha_i)
=\cos\beta I+i\sin\beta(\cos\alpha_i Z-\sin\alpha_i X).
\]

The third mixer, `x`, is the ordinary `exp(-i beta X)`. At `epsilon=1/2`, the
modified mixer coincides with ordinary `QAOACPU` at the same `beta`, and the
aligned mixer at `-beta`.

## Start sources

| Source | Meaning | Cost accounting |
|---|---|---|
| `provided_spins` | a full assignment of external labels, each exactly `+1` or `-1` | external/unaccounted |
| `provided_probabilities` | full `q = P(bit=1)`, finite, in `[0,1]` | external/unaccounted |
| `classical_solver` | an explicit CPU baseline: `gw_cpu`, `simulated_annealing_cpu`, or `exact_cpu` | measured `source_cost` |

`provided_*` entries are checked for full coverage of the current graph: an
unknown, missing, or repeated vertex is an error, and isolates are covered the
same way. The source is not executed in the constructor or at import; every new
`solve()` obtains data for its own graph. A cyclic source (`laboratory.*` inside
`classical_solver`) is rejected.

A classical child returns a full assignment validated on the original graph. A
GW SDP matrix by itself is not converted into `q`.

## Run it

```bash
python -m QA_max_cut_lib.solvers.warm_start_qaoa_cpu \
  --graph configs/graphs/warm_start_4.json \
  --config configs/solvers/warm_start_qaoa_cpu_provided.json
```

The committed configuration uses `p=1`, `max_steps=3`,
`shots_per_evaluation=32`, and `readout.shots=64`. Expected output begins:

```text
QA_max_cut_lib | laboratory.warm_start_qaoa_cpu | completed
Cut: 2   Energy: -2
```

The resource block contains:

```text
"training_shots": 480,
"readout_shots": 64,
"shots_used": 544
```

The arithmetic is visible: `3 * 5 * 32 = 480` training shots for three ADAM
iterations and `4p+1=5` central-difference evaluations each, plus a 64-shot
`best_of_shots` readout, for 544 total.

## Pause and resume

Pausing after one iteration gives `known_spent=160`, `readout_reserved=64`, and
`shots_available=320`. The readout reservation exists before training and is not
granted a second time.

After a resume, the completed run still has 544 shots in total; the resume does
not create a second independent limit. A classical preprocessing cost is
reported as `source_cost` and is never disguised as quantum shots.

`laboratory.warm_start_qaoa_cpu` supports the same checkpoint boundaries as
`QAOACPU`: a completed ADAM iteration, `training_done`, and `readout_done`. The
source result, `q`, `epsilon`, mixer, `node_order`, `theta`/`m`/`v`, RNG state,
budget, and readout token are frozen in the checkpoint. `restore` creates the
object through the factory; `resume` continues the canonical run; `solve()` is
always a new run.

A strict resume rejects a changed `epsilon`, source, or mixer as a new
scientific task. A classical source is not re-solved after restore.

## Result fields

The common `RunRecord` fields are present, plus:

| Field | Meaning |
|---|---|
| `variant_id` | the catalog id, e.g. `warm_start_product_qaoa_cpu_v1` |
| `warm_start_data` | source, raw and clipped `q`, preparation angles, provenance |
| `mixer_id` | `warm_aligned`, `rounded_modified`, or `x` |
| `source_type` | `provided_spins`, `provided_probabilities`, or `classical_solver` |
| `source_cut` | cut of the discrete seed, when the source provides one |
| `source_run` | the classical child run when applicable |
| `source_cost` | measured classical preprocessing resources |
| `source_cost_status` | `measured` or `external_unaccounted` |
| `initial_expected_cut` | cut expectation of the prepared product state |
| `initial_expected_energy` | negative of that expectation |

`source_cut` and `initial_expected_cut` are different numbers: the first belongs
to a discrete assignment, the second to the product state before training.
Neither is a substitute for the completed run's `cut_value`.

## Small controlled series

`experiments/benchmark_warm_start.json` expands to:

```text
graphs=3 cases=21 trials=42 rejected=0 references=3
```

That is 3 graph instances × 2 algorithm seeds × 7 configurations. The seven
configurations are the base plus `{epsilon 0.25, 0.50} × {x, warm_aligned,
rounded_modified}`. With 544 shots per run the series uses `42 × 544 = 22848`
quantum shots. Exact references are counted separately.

This is an equal-budget controlled comparison. It does not prove that warm-start
is always better than the base solver, GW, or SA.

## Limitations

- CPU full statevector only; memory and time grow as `2^N`.
- GPU warm-start is not implemented; the pair-symmetry representation does not
  apply directly.
- No analytic `p=1`, RQAOA, LS, hierarchy, fields, or environment variants on
  this page.
- Multiple warm starts are not aggregated for recursion.
- The source is not selected automatically across GW roundings.
- No quality improvement is promised; endpoint invariants are not proof that
  the final ADAM solution is better.
