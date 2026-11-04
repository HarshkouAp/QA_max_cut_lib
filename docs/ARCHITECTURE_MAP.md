# Architecture map and migration state

**Version:** `0.1.0rc2`.
**Source architecture:** `docs/reference/MaxCut_Library_Architecture.md`,
document version 0.10.

This page maps the architecture requirements onto the current code and states what
is not implemented. It does not replace the per-method pages and does not declare
planned capabilities as finished. The dependency rules and the invariants at the
end of this page change slowest; the status tables are updated with every release
candidate.

## Layers and dependency rules

| Layer | Modules | Rule |
|---|---|---|
| Core | `core.graph`, `core.records`, `core.saving`, `core.factory`, `core.budget`, `core.reduction` | Shared types, result record, serialization, assembly, shot budget, Ising reduction. Does not depend on algorithms. |
| Solvers | `solvers.base`, exact, QAOA CPU/GPU, RQAOA/LS, QAOA-in-QAOA, GW/SA, refiners, analytic p=1, warm start, conditional, block mean field, distributed, incremental, batch, quantized | Full algorithms and their computational components. May use Core and the numerical functions of Tools. |
| Experiments | `core.experiment_spec`, `core.experiment_ids`, `core.runner`, `core.references`, `core.profile`, `core.measurement`, `core.series_analysis`, `core.series_store` | Shared plan, independent trials, references, measurement protocol, and profile. Solver loops are not copied. |
| Checkpoint | `core.checkpoint`, `core.checkpoint_capabilities`, `tools.checkpoints` | Immutable JSON/NPY generations, `RunControl`, durable journal, inspect/restore/resume. The capability matrix rejects unknown codecs. |
| Generators | `generators.graphs`, `generators.recipes`, `generators.structured`, `generators.subgraphs`, `generators.plans`, `generators.benchmarks`, `generators.incremental` | Graph recipes, structured weights, benchmark resolvers, subgraphs, partitions, typed plans. Does not depend on solvers. |
| Tools | `tools.numerics`, `tools.plotting`, `tools.landscape`, `tools.reporting`, `tools.demo`, `tools.mutual_information`, `tools.mi_plots`, `tools.reduction_plots`, `tools.hierarchy_plots`, `tools.classical_plots`, `tools.experiment_plots`, `tools.experiments`, `tools.parallelism`, `tools.partition_analysis` | Numerical functions, analysis, visualization, reports, demonstrations. Numerical functions do not import graphics or solvers. |
| Laboratory | `laboratory.catalog`, `laboratory.*` controllers, `solvers.warm_start*` | Versioned inventory and catalog, explicit source aliases, CPU assemblies of historical variants. May use all layers; the base layers do not import Laboratory. |

The existing dependency `core.saving -> core.records` is normal. `solvers.base`
imports `core.saving` for automatic export; when the `save` protocol grows, that
import should become a shared service without cyclic imports.

## Implemented areas

| Area | State | Where |
|---|---|---|
| Mutable weighted graph, external integer labels, gaps, isolated vertices | yes | `core/graph.py` |
| Graph snapshot, recipe, SHA-256 fingerprint, restoration | yes | `WeightedGraph.snapshot/from_snapshot` |
| Exact pair reduction Max-Cut to Ising, shift, restoration, JSON snapshot | yes | `core/reduction.py` |
| Shared shot budget and readout reserve | yes | `core/budget.py` |
| `solve(graph)` returns `dict[int, int]` over all vertices | yes | `solvers/base.py` |
| New run on every `solve`, seed re-initialized | yes | `BaseSolver.solve` |
| Independent `RunRecord`, full JSON `algorithm.result`, direct attributes | yes | `core/records.py`, `BaseSolver.__getattr__` |
| Exact CPU solver with a guard limit | yes | `solvers/exact_cpu.py` |
| QAOA CPU: full statevector, `p` layers, ADAM, central differences | yes | `solvers/qaoa_cpu.py`, `qaoa_state.py`, `qaoa_training.py`, `objectives.py` |
| RQAOA CPU: correlation source, pair selection, reductions, exact remainder, fallback | yes | `solvers/rqaoa_cpu.py`, `components.py` |
| Analytic depth one: Mean Cut and all ZZ correlations without `2**N`, exact beta | yes | `solvers/analytic_p1.py` |
| LS-RQAOA CPU: two builders, local tasks, mean aggregator, coverage, global reducer, spawn pool | yes | `solvers/ls_rqaoa_cpu.py`, `generators/subgraphs.py` |
| QAOA-in-QAOA CPU and GPU: block hierarchy with exact projection | yes | `solvers/qaoa_in_qaoa_cpu.py`, `hierarchy_components.py`, `qaoa_in_qaoa_gpu.py` |
| Classical baselines: GW/SCS, SA, exact block refiner, refined wrapper | yes | `solvers/goemans_williamson_cpu.py`, `simulated_annealing_cpu.py`, `exact_block_refiner.py`, `refined.py` |
| GPU QAOA full and pair states, sampler, exact Mean Cut adjoint | yes | `solvers/qaoa_gpu.py`, `gpu_state.py` |
| GPU RQAOA and LS-RQAOA with one device owner | yes | `solvers/rqaoa_gpu.py`, `ls_rqaoa_gpu.py` |
| Conditional fields and environment LS | yes | `solvers/conditional.py`, `conditional_qaoa_cpu.py`, `conditional_source.py`, `environment_ls_rqaoa_cpu.py` |
| Block mean-field exchange, Gauss-Seidel and Jacobi | yes | `solvers/block_mean_field.py`, `block_mean_field_qaoa_cpu.py` |
| Distributed LS-RQAOA with mean-field episodes | yes | `solvers/distributed_ls.py`, `laboratory/distributed_ls_rqaoa_cpu.py` |
| Mutual-information diagnostics and passive observers | yes | `tools/mutual_information.py`, `tools/mi_plots.py`, `solvers/mi_observer.py` |
| Incremental LS-RQAOA with an epoch correlation bank | yes | `solvers/incremental_ls.py`, `generators/incremental.py` |
| Batch LS controller and quantized-angle wrapper | yes | `solvers/batch_ls.py`, `solvers/quantized_qaoa.py`, `laboratory/batch_ls_rqaoa_cpu.py`, `laboratory/quantized_qaoa_cpu.py` |
| Series: plan, stable IDs, runner, references, measurement, profile, analysis, store | yes | `core/experiment_*.py`, `tools/experiments.py`, `tools/parallelism.py`, `tools/experiment_plots.py` |
| Checkpoint and resume | yes, for the codecs listed in `core/checkpoint_capabilities.py` | `core/checkpoint.py`, `tools/checkpoints.py` |
| Generators: complete, `G(n,m)`, `G(n,p)`, three weight families, structured Wishart/Hebbian candidates, benchmark resolvers, subgraphs, two partitions, typed plans | yes | `generators/` |
| Saving JSON/NPY, `atts`, original data, overwrite guard | yes | `core/saving.py` |
| Report and log levels, colour | yes | `solvers/base.py`, `tools/reporting.py` |
| Static plots, `p=1` map, reduction and hierarchy views, classical and series panels, standalone 3D HTML | yes | `tools/plotting.py`, `tools/landscape.py`, `tools/reduction_plots.py`, `tools/hierarchy_plots.py`, `tools/classical_plots.py`, `tools/experiment_plots.py` |
| Laboratory catalog: 89 file records, 15 runnable variant cards, compatibility adapters, `list`/`show` CLI | yes | `laboratory/catalog.py`, `laboratory/data/*.json`, `laboratory/legacy.py` |
| Notebooks 00–19 and the Sphinx reference | yes | `notebooks/`, `docs/` |

The catalog counts are taken from `laboratory/data/inventory.json` and
`catalog.json`, not from the prose of earlier reports.

## What is not implemented

### Source closure

`SOURCE_CLOSURE.json` records all 89 archive members and
`full_archive_migration_complete=true`. The 18 formerly deferred branches are
available through `QA_max_cut_lib.laboratory.legacy` compatibility adapters:
`MK6V1`/`MK6V2`/`MK7V1`/`MK7V2`, `LSRQAOAExp` and `LSRQAOAFast1`–`Fast4`,
`QAOAAdam`/`QAOAF2`/`QAOAFM`, `CorrectedMeanMap`, `CorrelationExtractor`,
`EdgeExtender`, `GraphDecomposerLight`, `GraphScanner`, and
`partition_smoke_test`. They delegate to the shared controllers and are exercised
by `tests/test_legacy.py`; the release position is in `RELEASE_READINESS.md`.

### Run contracts and records

- There is no `RunState` object sufficient to continue a run.
- There is no `RunContext` carrying budget, randomness, data collection, and
  events.
- There is no general-purpose `ParallelContext`; `num_treads>1` is implemented in
  `LSRQAOACPU`, `IncrementalLSRQAOACPU`, `QAOAInQAOACPU`, and
  `SimulatedAnnealingCPU`, while `BatchLSRQAOACPU` records the request and applies
  1. GPU compositions keep one device owner.
- The run state of `BaseSolver` lives in local variables of `_run`, not in a
  separate object.
- A failed `_validate` does not create a `failed` record; the architecture asks for
  an available partial record on a failed start.
- There is no registry of result fields with types, units, and serialization
  rules; `optional_fields` is a temporary list.

### Solvers

- No D-Wave quantum annealer; the classical SA is implemented.
- No `D_RQAOA`, `DLS_RQAOA`, Meta-LS, or arbitrary partial updates; the LS family
  is limited to the documented base policy and its explicit variants.
- The hierarchy has no CE or updatable environment, no arbitrary block
  intersections, and no shared quantum state across levels.
- Public `solve(graph)` accepts only a graph without fields. Fields are supported
  in `IsingProblem` and in the conditional local problem, not in the public entry
  point.
- No multi-GPU execution, CUDA Graphs, Walsh tiles, or `batch>1` on the GPU path.
- Warm-start RQAOA has no resume; conditional, environment, and incremental
  controllers have no checkpoint codec.

### Tools and documentation

- No histogram of a large distribution drawn as columns on a plane.
- No single interactive profile for every plot type and no unified LaTeX style for
  all interactive views.
- No `LandscapeScanner` object with resume.
- No complete event formatter and colour theme for nested calls.
- `ASD-STE100` is not certified. Texts are written in plain technical English; no
  formal conformance check was performed.
- No public distribution license and no PyPI publication.

## Requirement groups

`REQUIREMENTS_TRACEABILITY.json` records 65 requirements: **65 passed, 0 blocked**.
The matrix, not this page, is the authority for individual requirement evidence.

| Group | Status | Comment |
|---|---|---|
| R01–R04: four sections, OOP, replaceable components, `solve(graph)` | passed | replaceable `type` components and the registry factory are in place |
| R05–R08: new run, seed, mutable graph, external labels | passed | covered by the graph and science tests |
| R09–R11: graph recipe, complete graphs up to 1000, weighted edges | passed | recipes and typed plans; simulation limits are separate |
| R12–R18: CPU/GPU, spin dictionary, reduction, QAOA, objectives | passed | CPU and NVIDIA paths; objectives on both |
| R19–R28: Laboratory, series, budgets, checkpoints, references | passed | all 18 former source records have compatibility adapters |
| R29–R34: landscapes, histograms, 3D, trajectories | passed | static, 3D, and trajectory views are implemented |
| R35–R40: `result`, direct attributes, report, logs, curves | passed | `result` and direct attributes share one record |
| R41–R48: graphs, LS energies, parameters, LaTeX, JSON config | passed | `style.text_engine` supports MathText and LaTeX |
| R49–R56: `save`, metadata, NPY, overwrite guard, notebooks | passed | notebooks 00–19 are present; 00 and 01 are executed for rc2 |
| R57–R60: CPU/GPU profiles, NumPy docstrings, `num_treads`, parallelism | passed | measured parallelism is reported with its limits |
| R61–R64: profiler, curves, Seaborn/LaTeX | passed | `ParallelismProfiler` and the plotting theme |
| R65: package name `QA_max_cut_lib` | passed | matches |

## Extension points that remain

The order below shows dependencies, not a release schedule.

1. **Core contracts.** `RunState`, `RunContext`, a typed result-field registry, and
   `MaxCutSolver` as an explicit protocol.
2. **Checkpoint coverage.** Codecs for RQAOA, LS, hierarchy, GPU, conditional, and
   incremental runs; each must declare its boundary in the capability matrix.
3. **Field-aware entry points.** A public path for `IsingProblem` fields and for
   conditional blocks without weakening the graph-only contract.
4. **Hierarchy variants.** CE and updatable environments as separate mathematical
   strategies, not hidden branches of the base hierarchy.
5. **Parallelism.** Concurrent profiling on one GPU and more scheduler policies;
   multi-GPU only after a separate numerical comparison with the CPU path.
6. **Presentation.** A single event formatter, a shared interactive style, and a
   resumable landscape scanner.
7. **Release.** A distribution license, package publication, and a formal
   terminology check if ASD-STE100 conformance is claimed.

## Invariants that must not be lost

- The solution dictionary covers every vertex, and its cut is evaluated on the
  original graph.
- A previously issued `RunRecord` does not change after a new `solve`.
- The shot budget is strict at every level.
- Original labels are restored after reductions.
- `C_original = b + C_reduced` holds for every reduction step.
- One energy convention is used everywhere: `E = -C`.
- A physical correlation and a pair-selection score are different quantities.
- No side files are written without `save`.
- Importing the package does not start a computation.
- A GPU request never falls back silently to the CPU.
- `result` and the direct attributes are read from one data source.

## Definition of done for a stage

1. New capabilities are described in docstrings and in the documentation.
2. A solver has a module demonstration run.
3. An independent check on a small problem exists.
4. The full test suite passes.
5. `MIGRATION_STATUS.md` and the coverage table are updated.
6. A separate git commit is made.
