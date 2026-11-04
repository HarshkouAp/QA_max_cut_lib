# Migration status for 0.1.0rc2

This page records which algorithm families are present in the current build, which
mathematics they implement, and what is still missing. It is an engineering status
page, not a user guide: method selection is in `CHOOSE_ALGORITHM.md`, exact fields
are in `API.md`, and the release position is in `RELEASE_READINESS.md`.

`0.1.0rc2` has all 65 requirements passed. R19, the closure of the 89-file source
archive, is complete: 18 formerly deferred branches are available through
`QA_max_cut_lib.laboratory.legacy` compatibility adapters. See
`SOURCE_CLOSURE.json` for the mapping.

## Status by area

| Area | In this build | Next scope |
|---|---|---|
| Graph | `WeightedGraph`, original integer labels, gaps, negative and zero weights, snapshot and fingerprint | additional operations required by reductions |
| Reductions | `IsingProblem`, `SpinConstraint`, `ReductionState`, `GraphReducer`; exact energy equality, `b_t` shift, full constant `c_t`, reverse restoration, JSON snapshot | connect other correlation sources and subgraphs |
| Solvers | `ExactCPU`, `QAOACPU`, `RQAOACPU`, `LSRQAOACPU`, `BatchLSRQAOACPU`, `QAOAGPU`, `RQAOAGPU`, `LSRQAOAGPU`, `QAOAInQAOACPU`, `QAOAInQAOAGPU`, `GoemansWilliamsonCPU`, `SimulatedAnnealingCPU`, `RefinedSolver`, `QuantizedQAOACPU`; shared lifecycle, JSON result, direct attributes | further classical series and fine-tuning policies |
| Classical | GW SDP/SCS with hyperplane rounding; SA reads/sweeps/seeds; `ExactBlockRefiner` with environment fields; `RefinedSolver` with base and refinement | benchmark series are in place; checkpoint is not |
| Hierarchy | blocks, anchor normalization, local readout, `GraphReducer` constraints, `W`/`Delta b`, `(level, anchor)` tree, CPU pool and GPU owner, `level_round_robin` | CE and updatable environment, fields, arbitrary intersections, shared quantum state |
| RQAOA | QAOA training or analytic `p=1`, physical ZZ correlations, pair selection by `|c|`, `edges`/`all_pairs`, exact reduction, exact remainder, fallback, strict budget; the GPU source uses the same coordinator | other local and hierarchical variants |
| LS-RQAOA | two builders, induced local tasks, `mean`, coverage, anchor task, one global reducer, `refresh_all`, `spawn` pool, shared shot and CPU budget; GPU LS keeps one device owner | other LS variants, fields, partial updates, Meta-LS |
| Analytic `p=1` | `AnalyticP1Evaluator`, exact beta, finite gamma grid, Mean Cut, all pairs without `2**N`, `shots_used=0`; remains the CPU reference | fields, `p>1`, arbitrary mixer, GPU, distribution |
| Assembly | replaceable `type` components, a registry factory for all public implementations | historical named Laboratory assemblies |
| QAOA | CPU and NVIDIA GPU: ADAM, central differences, `p` layers, five objectives, exact and shot estimation, two readout modes; GPU full/pairs, sampler, exact Mean Cut adjoint | CUDA Graphs, Walsh tiles, batch, other optimizers |
| GPU | `QAOAGPU`, `QAOAGPUCorrelationSource`, `RQAOAGPU`, `LSRQAOAGPU`; CuPy 14.2.0, RTX 3050 Laptop CC 8.6, 4 GiB; CPU import without CUDA | RTX 5060, V100 and other cards need separate checks; multi-GPU |
| Generators | complete, `G(n,m)`, `G(n,p)`, three weight families, induced subgraphs, `random_partition`, `greedy_abs_weight_partition`; schema2 families, BenchGraph16/Benchmark297 resolvers, Wishart/Hebbian candidates, typed vertex/induced/edge plans, coverage analyzer | research generators and subgraph strategies from the archive |
| Tools | correlations, quality, static plots, `p=1` map, reduction weights and energies, LS coverage, `binomial(2p,2)` trajectories, hierarchy overview/3D/tree, GW/SA/refinement panels, series coverage and quality, parallelism panels, standalone HTML, MI heatmaps and profiles | large-distribution histograms, an interactive profile |
| Laboratory | 89 file records and 15 runnable variant cards; source aliases and compatibility adapters; `laboratory list/show` | warm-start RQAOA resume, LS/MK families, fields and environment, MI strategies |
| Series | `BenchmarkSpec`, stable graph/case/trial IDs, full product, rejected records, isolated and concurrent runner, exact/GW/best-known references, write-once records | checkpoint/resume, distributed campaigns |
| Parallelism | `LSRQAOACPU`, `IncrementalLSRQAOACPU`, `QAOAInQAOACPU`, SA; shared `ParallelismProfiler` with 1/2/4, warmups and repetitions, work counters, Amdahl and Karp–Flatt, paired bootstrap and a global recommendation | concurrent profile on one GPU, more scheduler policies |
| Budgets | strict shared budget for QAOA/RQAOA/LS; per-step correlation reserve and round-robin quotas | nested budgets for QAOA-in-QAOA |
| Checkpoints | `QAOACPU` JSON/NPY generations, ADAM `theta/m/v/best`, PCG64 state, budget and readout token, durable journal, atomic latest; four Laboratory codecs; other components are rejected explicitly | RQAOA, LS, hierarchy, and GPU codecs |
| Result and export | JSON, ndarray/complex, NPY, `atts`, snapshots, overwrite guard | serializers for special objects of future algorithms |
| Presentation | Seaborn with Computer Modern/MathText; standalone Plotly HTML | a shared theme for all future interactive plots |
| Documentation | notebooks 00–19, API reference, mathematics, Sphinx/numpydoc | extend together with each ported algorithm |

## Reduction core

Implemented:

- weighted Max-Cut to Ising with `J=w/2`, `h=0`, `c=-Σw/2`;
- pair constraint `s_u = σ s_v` for arbitrary pairs, including non-adjacent ones;
- exact energy equality for every remainder assignment;
- accumulated shift `b_t` and full constant `c_t` kept as different quantities;
- reverse restoration of the original labels, including chains where the retained
  vertex is removed later;
- compact event trace: weight, correlation, score, `fixed_energy`,
  `expected_fixed_energy`, `Δb`, `Δc`, and structural statistics;
- a JSON-compatible snapshot of the mathematical state and continuation through
  `from_snapshot`;
- atomic failure: a rejected request does not change the state.

Limits of the core:

- there is no absolute fixation of a single spin;
- the core does not choose the pair or the correlation; `RQAOACPU` does;
- the snapshot is not an optimizer checkpoint/resume;
- fields are supported in `IsingProblem`, but the public `solve(graph)` still
  accepts only a graph without fields.

## RQAOA

Implemented:

- `QAOAStatevectorCorrelationSource` trains the existing full statevector for
  `p>=1` and returns all physical ZZ correlations;
- `MaxAbsCorrelationSelector` supports `edges` and `all_pairs`, lexicographic
  tie-breaking, `zero_sign`, and a separate `score=abs(c)`;
- `RQAOACPU` applies exactly one merge per step through the reduction core, keeps
  the constraints, task versions, shift, and full Ising constant;
- the remainder is solved by `ExactCPU`; when the run stops on `max_reductions` or
  on the budget, the full `all_plus_reduced_fallback` solution is returned;
- the shared `ShotBudget` covers training and correlations, reserves the mandatory
  correlation shots of a step, and needs no shots for the exact remainder;
- `reduction_history`, `parameter_trajectory`, `local_optimizer_history`,
  `correlation_tables`, and `final_reduction_state` are collected by separate
  flags;
- Tools plots weights, energies, `cut_offset`/`ising_constant`, all `binomial(2p,2)`
  flat trajectories, a standalone 3D HTML view, and the profile
  `min_beta E(gamma,beta)` / `beta_star(gamma)`.

Limits of the RQAOA family:

- the statevector source uses the full vector and does not scale to 1000 vertices;
- the analytic source supports only `p=1`, Mean Cut, exact moments, and provides no
  distribution, shots, CVaR, Gibbs, Best-of-K, or Threshold Probability;
- there is no GPU, LS, QAOA-in-QAOA, or Laboratory path in this family;
- `num_treads` accepts only 1;
- checkpoint/resume for RQAOA is not implemented;
- RQAOA remains a heuristic: an exact remainder does not prove the global optimum.

## Analytic `p=1`

Implemented:

- `AnalyticP1Evaluator` with a dense `N×N` matrix, pair batches, and no `2**N`
  objects;
- the physical ZZ correlation formula for adjacent and non-adjacent pairs,
  including products without division by a cosine;
- coefficients `E0, A, B`, exact minimization in beta through `hypot` and `atan2`;
- the `gamma_grid_exact_beta` strategy with explicit bounds and deterministic
  tie-breaking;
- the `qaoa_analytic_p1_cpu` source with `edges`/`all_pairs` coverage, a gamma
  profile, and classical counters;
- connection to the same `RQAOACPU` without changing the reducer or the finisher;
- `shots_used=0`, `training_shots=0`, `correlation_shots=0`, `readout_shots=0`;
  `limits.shots_total=0` is accepted;
- notebook 05 and a numerical profile up to 1000 vertices.

Limits: no fields, no `p>1`, no other mixer or state, no full distribution, no
discrete readout, no shots, no GPU. The parallel mode belongs to LS-RQAOA, not to
the analytic evaluator itself.

## LS-RQAOA

Implemented:

- `RandomPartitionBuilder` with a separate RNG, exact coverage of every pass, and
  preservation of repeats between passes;
- `GreedyAbsWeightPartitionBuilder` growing by the largest sum of absolute link
  weights, tie-breaking by the smaller external label, and full vertex coverage
  including isolates and label 0;
- induced local tasks with `boundary="induced_no_fields"`, no fields, and no
  boundary edges; external labels and signed weights are preserved;
- `MeanCorrelationAggregator` dividing by the number of observations of each pair,
  with canonical pairs and true zeros;
- `ObservedMaxAbsCorrelationSelector` selecting only observed pairs, with a
  separate score and no fictitious zeros;
- coverage of `A_t`/`C_t`, `require_full`, an anchor task before the budget, and no
  automatic quadratic expansion of the plan;
- one global `GraphReducer.reduce` per step, `refresh_all`, and an exact remainder;
- `ProcessPoolExecutor` with `spawn`, one pool per run, a bounded future queue,
  `finally` closing, and no partial reduction on error;
- a shared step shot budget: the sum of mandatory correlations, then full
  `(4p+1)M_train` iterations in task-ID order; an analytic task costs 0;
- `num_treads` limits the number of worker processes and each worker receives
  `native_threads=1`; the request and the fact are stored separately;
- local angle histories, coverage, energy events, and a standalone 3D HTML view;
  notebook 06 and a P=1/2/4 profile.

Boundaries: only the base LS with `mean`, `refresh_all`, induced tasks without
fields, and an exact finisher. Experimental LS variants with fields, partial
updates, older scores, and Meta-LS are not declared ported. There is deliberately
no LRU cache of old correlations after the graph changes.

## Conditional fields and environment LS

Added in `0.1.0a13`:

- `ConditionalCutProblem` and exact fixed-spin conditioning of a weighted Max-Cut
  problem;
- a full CPU statevector for the active block and `ConditionalQAOACPU`;
- `EnvironmentLSRQAOACPU`, where local fields from fixed external spins enter only
  the local problem and never the global reducer;
- Laboratory routes, source cards, configurations, examples, tests,
  documentation, and notebook 14.

Limits: checkpoint/resume of the new LS recipe is not implemented, and GPU fields
are not part of this transfer.

## Block mean-field exchange

Added in `0.1.0a15`:

- `BlockMeanFieldQAOACPU` with a compact `m/q` bank;
- Gauss-Seidel and Jacobi schedules;
- angle and moment damping with the same convex rule for `m` and `q`;
- local surrogate objectives and a full sign-rounded output verified by a cut on
  the original graph;
- catalog cards, configurations, examples, tests, documentation, and notebook 15.

Checkpointing exists at full-sweep boundaries; a mid-sweep child codec is not
implemented.

## Distributed LS-RQAOA

Added in `0.1.0a15`:

- `DistributedLSRQAOACPU` with independent mean-field episodes;
- fresh final ZZ extraction and equal-episode mean aggregation;
- carry and cold transport between episodes;
- complete anchor repair and an outer reducer commit;
- a compact outer checkpoint at `initialized` and `reduction_committed`.

Limits: episodes run sequentially, and nested mid-episode state has no codec.

## Mutual-information diagnostics

Added in `0.1.0a16`:

- `tools.mutual_information`: partial trace, entropy, quantum mutual information,
  Z-MI from probabilities, counts, and moments, pair scope, mask and statistics,
  and hop distance;
- `tools.mi_plots`: offline MI heatmap, prefix profile, and hop profile;
- passive MI observers for QAOA and RQAOA runs.

MI is a diagnostic layer: it does not change the optimizer, the selection, or the
readout. `MI_QAOA.py` and `MI_LS_RQAOA.py` are classified as diagnostic assemblies,
not as faithful legacy trajectories. Not ported: MI-guided selection, tomography,
quantum-state lift, and GPU MI.

## Incremental LS and MK components

Added in `0.1.0a17`:

- `generators.incremental` with `single_densest_edge_count_v1` and
  `survivor_neighbor_cover_v1`;
- `IncrementalLSRQAOACPU` with an epoch correlation bank, freshness policies, four
  selection scores, and corrected component recomposition.

Not ported: batch `MK_3`, odds-fusion `fast_2`, round-max `LS_exp`,
`CorrelationExtractor` rollback, `MK_6/7/8`, and the GPU variants.

## Generators, structured weights, typed plans

Added in `0.1.0a18`:

- schema2 graph families and BenchGraph16/Benchmark297 resolvers;
- Wishart and Hebbian candidate generators;
- typed vertex, induced, and edge plans with versioned builder IDs;
- corrected builders and an independent coverage and topology analyzer;
- `generators.cli` and `tools.partition_analysis`.

Schema1 recipes and the induced LS defaults are preserved. Not ported:
`GraphDecomposer_light` truss and repair branches, the pattern-recovery oracle, and
the full edge-subset LS controller.

## Batch LS and quantized QAOA

Added in `0.1.0rc2`:

- `BatchLSRQAOACPU`: all local observations complete before the graph changes,
  proposals are frozen, and one of two explicit reconciliation policies is applied.
  The module is a compact CPU implementation of the batch contract and does not
  claim bitwise equivalence with the historical `MK_3` controller.
- `QuantizedQAOACPU`: continuous QAOA training followed by evaluation of a
  quantized-angle candidate. Quantization rounds to the nearest grid point with
  ties to even; the default step is `π/8`. The quantized state is arithmetic from
  the stored angles, and no additional measurement batch is created. The continuous
  result remains available separately.

## Basis for the migration

The source archive contains 89 Python files and 25 548 lines. The file and class
inventory is in `reference/source_inventory.json`; the closure ledger is
`SOURCE_CLOSURE.json`. Presence of a file in the inventory does not mean that the
corresponding algorithm has been ported.

## Remaining sequence

1. Add the missing checkpoint codecs and declare each boundary in the capability
   matrix.
3. Complete CE and updatable-environment variants as separate mathematical
   strategies.
4. Add independent GPU implementations only with a numerical comparison against the
   CPU path.

Missing capabilities are not replaced by stubs with a successful status.
