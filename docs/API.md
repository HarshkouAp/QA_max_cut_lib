# API reference — QA_max_cut_lib 0.1.0rc2

This page is the exact reference for the public Python API of
`QA_max_cut_lib` 0.1.0rc2: classes, functions, configuration fields,
defaults, constraints, exceptions, result fields, and data formats. Use it
when a signature or a default matters. Conceptual and mathematical
explanations are linked at the end of each part.

Every configuration is an ordinary JSON-compatible Python `dict`; a JSON
file is read with `tools.load_json(path)`. `solve()` returns a spin
dictionary `dict[int, int]`. `algorithm.result` returns a JSON-compatible
`dict`, never a formatted string. Textual JSON is produced with the
standard `json.dumps`.

## Minimal call

```python
from QA_max_cut_lib import ExactCPU, WeightedGraph

graph = WeightedGraph(
    edges=[(10, 30, 1.5), (10, 80, 2.0), (30, 80, -0.5)],
    nodes=[10, 30, 80, 101],
)

solver = ExactCPU()
solution = solver.solve(graph)       # {10: 1, 30: -1, 80: -1, 101: 1}
cut = graph.cut_value(solution)      # 3.5
result = solver.result               # JSON-compatible dict
last_run = solver.last_run           # independent RunRecord
```

`solution` contains one integer spin `+1` or `-1` for every original node
label, including isolated nodes. `result["cut_value"]` is `C(s)`;
`result["solution_energy"]` is `-C(s)`.

## Reading this reference

- A configuration object is a closed schema. Unknown fields raise
  `ValueError` unless a table explicitly says otherwise.
- Types are written as Python types. `mapping` means any `Mapping`;
  `array_like` means anything accepted by `numpy.asarray`.
- Integer fields reject `bool`, even though `bool` is an `int` subclass.
- The spelling `num_treads` is part of the public API and is used
  everywhere, including result fields.
- A value of `None` means "not set" or "not applicable", as stated by the
  field description. It is not a zero and not an empty table.

| Exception | Raised for |
|---|---|
| `TypeError` | A Python type is wrong: non-integer label, boolean weight, non-mapping, non-`WeightedGraph` input |
| `ValueError` | A value is syntactically valid but not allowed: unknown field, out-of-range number, duplicate label, spin not `±1`, unknown solver type |
| `ArithmeticError` | A newly computed constant, field, coupling, offset, or reduction metric is not finite |
| `RuntimeError` | The run state is wrong (an active run, a missing resume state) or a required backend is unavailable |
| `KeyError` | `WeightedGraph.remove_edge` names an absent edge |
| `FileNotFoundError`, `FileExistsError`, `NotADirectoryError` | Export destination problems |
| `CheckpointError` | Base class for checkpoint corruption, incompatibility, and lock errors |

## Public imports and run lifecycle

```python
from QA_max_cut_lib import (
    WeightedGraph, GraphFactory, SolverFactory, RunRecord, ExactCPU, QAOACPU,
    RQAOACPU, AnalyticP1Evaluator, AnalyticP1CorrelationSource,
    LSRQAOACPU, QAOAGPU, QAOAGPUCorrelationSource, RQAOAGPU, LSRQAOAGPU,
    QAOAInQAOACPU, QAOAInQAOAGPU, GoemansWilliamsonCPU, SimulatedAnnealingCPU,
    ExactBlockRefiner, RefinedSolver, BenchmarkSpec, ExperimentRunner,
    ParallelismProfiler, ProfileSpec, ReferencePolicy, RunControl, restore,
    inspect_checkpoint,
    GraphReducer, ReductionState, IsingProblem, SpinConstraint,
    QAOAStatevectorCorrelationSource, MaxAbsCorrelationSelector, ExactCPUFinisher,
)
from QA_max_cut_lib import tools
from QA_max_cut_lib.generators import (
    partition, RandomPartitionBuilder, GreedyAbsWeightPartitionBuilder,
)
from QA_max_cut_lib.solvers.ls_components import MeanCorrelationAggregator
from QA_max_cut_lib.solvers.objectives import validate_objective, objective_value
from QA_max_cut_lib._version import __version__
```

`SolverFactory.available()` returns the tuple

```text
("exact_cpu", "qaoa_cpu", "rqaoa_cpu", "ls_rqaoa_cpu", "qaoa_gpu",
 "rqaoa_gpu", "ls_rqaoa_gpu", "qaoa_in_qaoa_cpu", "qaoa_in_qaoa_gpu",
 "gw_cpu", "simulated_annealing_cpu", "refined",
 "laboratory.warm_start_qaoa_cpu", "laboratory.warm_start_rqaoa_cpu",
 "laboratory.conditional_qaoa_cpu",
 "laboratory.environment_ls_rqaoa_cpu",
 "laboratory.block_mean_field_qaoa_cpu",
 "laboratory.distributed_ls_rqaoa_cpu", "laboratory.qaoa_mi_cpu",
 "laboratory.incremental_ls_rqaoa_cpu", "laboratory.batch_ls_rqaoa_cpu",
 "laboratory.quantized_qaoa_cpu", "laboratory.rqaoa_mi_cpu")
```

`SolverFactory.from_config(config)` selects the class by `config["type"]`.
An unknown type raises `ValueError` that names the requested type, the
package version, and the available tuple. The factory performs no
calculation.

The common constructor is `ClassName(config=None)`. It copies the
configuration and validates it. If `config["type"]` is present, it must
match the class. The common run method is

```python
solve(graph, *, report=None, logs=None, num_treads=None, control=None)
```

It accepts a `WeightedGraph`, returns `dict[int, int]`, and does not modify
the input graph. `report`, `logs`, and `num_treads` override the stored
configuration for that run only. `control` must be a `RunControl`
(`TypeError` otherwise) and is accepted only by checkpoint-capable classes
(`ValueError` otherwise). One solver instance supports one active run;
starting a second concurrent call raises `RuntimeError`. Use separate
instances for concurrent calculations.

A repeated `solve` creates a fresh RNG from the configured `seed`, clears
the previous result in the solver, and does not reuse prior state. There
is no explicit warm start through `solve()`. `last_run` is an independent
snapshot; keep it in your own list if you need a run history. A later
graph change does not alter an existing snapshot. Do not mutate the input
graph concurrently while the snapshot is created.

Before the first run, `result` and `last_run` are `None`, and the optional
scientific attributes in `BaseSolver.optional_fields` read as `None`.
Once validation of a run starts, the previous record is discarded; a
validation error therefore leaves `result` as `None`. A computation error
produces an available `last_run` with `status="failed"`,
`termination_reason` equal to the exception class name, an `error`
message, and re-raises the exception. An export error after a successful
calculation leaves the scientific result available.

See also: [Quickstart](QUICKSTART.md); the repository-level
``CONFIGURATION.md`` describes the configuration workflow.

## WeightedGraph

Signature: `WeightedGraph(edges=(), nodes=())`.

| Argument or attribute | Type | Meaning |
|---|---|---|
| `edges` | iterable of `(u, v, w)` | Explicit finite real weights |
| `nodes` | iterable of `int` | Labels, including isolated nodes; insertion order is preserved |
| `.nodes` | `tuple[int]` | External labels in fixed order; gaps and negative labels are allowed |
| `.edges` | `tuple` of triples | Each undirected edge exactly once |
| `.version` | `int` | Number of mutations of this instance |
| `len(graph)` | `int` | Number of vertices |

Boolean labels, self-loops, duplicate labels in `nodes`, duplicate
undirected edges in the constructor, NaN/Inf weights, and boolean weights
are rejected. Negative and zero weights are allowed. A zero-weight edge is
stored as an edge and counted in structural statistics; an absent edge is a
different case.

| Method | Returns | Action |
|---|---|---|
| `add_node(node)` | `None` | Adds a missing vertex |
| `set_weight(u, v, weight)` | `None` | Adds or updates an edge and any missing endpoints |
| `remove_edge(u, v)` | `None` | Removes an edge; endpoints remain; an absent edge raises `KeyError` |
| `weight(u, v)` | `float` | Edge weight, or `0.0` for an absent pair |
| `cut_value(solution)` | `float` | `C(s)` by direct edge summation; checks the exact vertex set and integer spins `±1` |
| `subgraph(nodes)` | `WeightedGraph` | Induced subgraph with the requested original-label order |
| `snapshot()` | `dict` | Weighted snapshot, `version`, `recipe`, and SHA-256 `fingerprint` |
| `from_snapshot(data)` | `WeightedGraph` | Restores a snapshot and verifies `fingerprint` when present |
| `copy()` | `WeightedGraph` | Independent copy |

`cut_value` raises `ValueError` if the solution does not contain every node
exactly once, if a spin is not an integer `+1` or `-1`, or if the summed
cut is not finite.

The fingerprint covers task content (labels, pairs, weights), not insertion
order. Insertion order is recorded separately in `node_order`; use that
order when interpreting state-vector bits and correlation matrices.

See also: [Concepts](CONCEPTS.md).

## Reduction core: Max-Cut and Ising

The core does not choose a correlation or a pair. It accepts an explicit
constraint `s_removed = sign * s_retained` and performs the exact algebraic
transformation. The original `WeightedGraph` is never modified.

```python
from QA_max_cut_lib import GraphReducer, IsingProblem

reducer = GraphReducer()
state = reducer.start(graph)
event = reducer.reduce(state, removed=10, retained=30, sign=-1,
                       correlation=-0.6)
solution = reducer.restore(state, {30: 1, 80: -1, 101: 1})
snapshot = reducer.snapshot(state)
restored_state = reducer.from_snapshot(snapshot)
```

`GraphReducer` holds no state. All mutable data lives in `ReductionState`.
One `ReductionState` describes one working process; independent processes
need separate states.

### IsingProblem

Signature: `IsingProblem(constant=0.0, fields=None, couplings=None, nodes=())`.

The energy convention is

```text
E(s) = c + sum_i h_i s_i + sum_{i<j} J_ij s_i s_j.
```

| Argument or attribute | Type | Meaning |
|---|---|---|
| `constant` | `float` | Full constant `c`; not a Max-Cut offset |
| `fields` | mapping `int -> float` | Local fields `h_i`; a missing field is zero |
| `couplings` | mapping `(int, int) -> float` | Pairs `J_ij`; each unordered pair once |
| `nodes` | iterable of `int` | All vertices, including isolated ones; order is preserved |
| `.nodes` | `tuple[int]` | Vertex order |
| `.fields` | `dict[int, float]` | Nonzero fields only |
| `.couplings` | `dict[(int, int), float]` | Nonzero pairs only |
| `.edge_count` | `int` | Number of nonzero pairs |
| `.sum_abs_couplings` | `float` | Sum of absolute nonzero pairs |
| `.energy(solution)` | `float` | Direct polynomial sum for a complete assignment |

`IsingProblem.from_maxcut(graph)` sets `J_ij = w_ij / 2`, `h_i = 0`, and
`c = -sum(w) / 2`. `to_weighted_graph()` returns the graph `w = 2J` only
when every field is zero; otherwise it raises `ValueError`. Zero pairs are
not stored. Small nonzero pairs are preserved without a hidden threshold.

Additional members: `field(node)`, `coupling(u, v)`, `neighbors(node)`,
`set_field(node, value)`, `set_coupling(u, v, value)`, `add_node(node)`,
`copy()`, `snapshot()`, and `from_snapshot(data)`. Their snapshots carry
`schema_version=1` and `problem_type="ising"`.

### SpinConstraint

`SpinConstraint(removed, retained, sign)` is an immutable record of one
constraint. `sign` accepts only the integers `+1` and `-1`. `removed` and
`retained` are original integer labels and must differ. The record exposes
`to_dict()` and `from_dict(data)`.

### ReductionState

| Attribute or method | Meaning |
|---|---|
| `.original_nodes` | All original labels in insertion order |
| `.original_graph` | Independent copy of the original Max-Cut graph, or `None` for a general Ising origin |
| `.original_ising` | Independent copy of the original general Ising problem, or `None` for a graph origin |
| `.ising` | Current working problem; treat it as read-only |
| `.current_nodes` | Current labels |
| `.constraints` | Constraints in application order |
| `.trace` | Reduction events in application order |
| `.cut_offset` | Accumulated `b_t`; `None` for a general Ising origin |
| `.version` | Number of applied reductions |
| `.current_graph()` | Current graph `w = 2J` for a Max-Cut state; `ValueError` for an Ising origin or nonzero fields |
| `.current_cut_value(residual)` | `b_t + C_current(residual)`; `ValueError` when cut metrics are undefined |
| `.original_cut_value(solution)` | `C` on the original graph; `ValueError` if there is no original graph |
| `.original_energy(solution)` | `E = -C`, or the original Ising polynomial |
| `.current_energy(residual)` | Current Ising polynomial |
| `.apply_pair(removed, retained, sign, *, correlation=None, selection_score=None)` | Applies one reduction and returns its event |
| `.restore(residual)` | Restores all original spins in reverse constraint order |
| `.snapshot()` | JSON-compatible snapshot of the mathematical state |
| `ReductionState.from_snapshot(data)` | Restores a state |

`GraphReducer.start(graph)` and `start_from_ising(problem)` create a state.
`GraphReducer.reduce(state, removed, retained, sign, *, correlation=None,
selection_score=None)` delegates to `apply_pair`.
`GraphReducer.restore(state, residual_solution)` delegates to `restore`.
`GraphReducer.snapshot(state)` and `GraphReducer.from_snapshot(data)` save
and read a state.

### Reduction event

Every `reduce` call appends exactly one event to `state.trace`.

| Field | Meaning |
|---|---|
| `step`, `task_version_before`, `task_version_after` | Step number and working-task versions |
| `removed_node`, `retained_node`, `sign` | Original labels and the applied constraint |
| `selected_coupling` | `J_uv` before the transformation |
| `selected_weight` | `w_uv = 2 J_uv` for a Max-Cut origin; otherwise `None` |
| `physical_correlation`, `correlation_reason` | Supplied `c_uv`, or the reason it is absent |
| `selection_score`, `selection_score_reason` | Separate selection score, not a correlation |
| `fixed_energy`, `fixed_energy_reason` | `-w_uv (1 - sign) / 2` for Max-Cut; reason when not defined |
| `expected_fixed_energy`, `expected_fixed_energy_reason` | `-w_uv (1 - c_uv) / 2` when a correlation exists |
| `constraint_probability` | `(1 + sign * c_uv) / 2` when a correlation exists |
| `cut_offset`, `cut_offset_delta` | `b_t` and its change `Delta b` |
| `ising_constant`, `ising_constant_before`, `ising_constant_delta` | Full Ising constant and its change |
| `vertices_after`, `edges_after` | Structure after the step; edges are nonzero pairs |
| `sum_abs_weight`, `mean_abs_weight` | Nonzero-edge domain after the step |
| `sum_abs_coupling`, `mean_abs_coupling` | The same quantities for `J` |
| `mean_abs_weight_domain` | Literal domain marker `nonzero_edges_after` |
| `mean_abs_weight_reason` | Why `mean_abs_weight` is `None`; `empty_nonzero_edge_set` for an empty domain |
| `maxcut_metrics_applicable`, `maxcut_metrics_reason` | Whether graph cut metrics are defined |

For a general Ising problem with nonzero fields, `selected_weight`,
`fixed_energy`, `cut_offset`, and `mean_abs_weight` are `None`, and
`maxcut_metrics_reason` is `working_problem_has_fields` (or
`original_problem_is_general_ising`). Fields remain in the working problem
and are included in `ising_constant` and `.current_energy`.

### State snapshot

A snapshot contains `schema_version=1`, `state_type="reduction_state"`,
`energy_convention`, `version`, `maxcut_origin`, `original_nodes`,
`original_graph`, `original_ising`, `ising`, `cut_offset`, `constraints`,
and `trace`. It is JSON-compatible but is not an optimizer
checkpoint/resume record. After `json.dumps`/`json.loads`, the snapshot
still accepts reductions and restores a solution.

### Reduction errors

`TypeError`: a label is not an integer, a weight is not a real number, or a
boolean was supplied as a number. `ValueError`: self-pair, absent or
already removed node, `sign` not `±1`, correlation outside `[-1, 1]`,
incomplete residual. `ArithmeticError`: a new constant, field, or coupling
is not finite. A failed request does not change the state.

See also: [Mathematical conventions](MATHEMATICS.md).

## RQAOA on CPU

`RQAOACPU` is the full recursive CPU solver. It trains QAOA on the current
working graph, obtains physical correlations `⟨Z_u Z_v⟩`, selects one
allowed pair, applies the exact constraint `s_u = σ s_v` through the
reduction core, repeats the loop, and solves the residual with `ExactCPU`.
Original spins are restored in reverse order. It is a heuristic for the
original task, not an exact optimizer of it.

```python
from QA_max_cut_lib import SolverFactory, tools

config = tools.load_json("configs/solvers/rqaoa_cpu.json")
algorithm = SolverFactory.from_config(config)
solution = algorithm.solve(graph, report=1, logs=0)
result = algorithm.result
trajectory = algorithm.parameter_trajectory
history = algorithm.reduction_history
```

### Configuration

| Field | Value |
|---|---|
| `parameters.finish_at` | Integer `>= 1`, default `4`; a residual of this size is solved by the exact finisher |
| `parameters.max_reductions` | `null` or integer `>= 0`, default `null`; `0` forbids reductions |
| `correlation_source.type` | `qaoa_statevector_cpu`, `qaoa_analytic_p1_cpu`, `qaoa_statevector_gpu` (only in `RQAOAGPU`), or `warm_start_ensemble_cpu` |
| `correlation_source.parameters.layers` | Depth `p >= 1`, default `1`; angle order is γ₀…γₚ₋₁, β₀…βₚ₋₁ |
| `correlation_source.parameters.max_vertices` | Protective full-state limit, default `20`, at most `24` for the direct state-vector source |
| `correlation_source.objective` | One of the five QAOA criteria |
| `correlation_source.objective_reference` | `working_graph` (default) or `original_graph` |
| `correlation_source.estimation` | `exact` or `shots` for the training criterion |
| `correlation_source.optimizer` | ADAM with central differences |
| `correlation_source.initialization` | `fixed_angles` or `random_angles` |
| `correlation_source.step_initialization` | `restart` (default) or `previous_step` |
| `correlation_source.correlations` | `exact` or `shots`; `shots` requires `shots` |
| `selection.parameters.pairs` | `edges` (default) or `all_pairs` |
| `selection.parameters.zero_sign` | `+1` (default) or `-1` for an exactly zero correlation |
| `finisher.type` | `exact_cpu` |
| `finisher.parameters.max_vertices` | Default `16`; must be at least `finish_at` |
| `limits.shots_total` | `null` or a strict total shot limit |
| `collect.reduction_history` | Compact reduction events |
| `collect.parameter_trajectory` | Angles before each merge |
| `collect.local_optimizer_history` | Local ADAM histories |
| `collect.correlation_tables` | Full correlation tables |
| `collect.final_reduction_state` | JSON snapshot of the mathematical state |

The source has no `save` field, no public readout, and no separate
`shots_total`. `num_treads` accepts only `1` for `RQAOACPU`. Unknown fields
and incompatible combinations are rejected. `finish_at` and
`max_vertices` are checked before a full state vector is allocated.

`qaoa_statevector_gpu` is accepted only by `RQAOAGPU`; the other GPU
hierarchies enforce their own backend rules.

### Result fields

In addition to the common `RunRecord` fields:

| Field | Meaning |
|---|---|
| `reductions_completed` | Number of applied constraints |
| `remaining_node_order` | External labels of the final residual |
| `cut_offset` | Accumulated offset `b_t` |
| `ising_constant` | Full constant `c_t` of the working problem |
| `reduced_solution`, `reduced_cut_value` | Residual solution and residual cut |
| `solution_source` | `exact_finisher`, `zero_interaction_initial`, `zero_interaction_residual`, `empty_graph`, or `all_plus_reduced_fallback` |
| `termination_reason` | `empty_graph`, `no_interactions`, `finish_at`, `max_reductions`, or `shot_budget` |
| `reduction_history` | Events with weight, correlation, `σ`, score, and energies |
| `parameter_trajectory` | Angles before the corresponding merge |
| `local_optimizer_history` | Local optimizer histories |
| `correlation_tables` | Tables with coverage and task version |
| `final_reduction_state` | Snapshot of the mathematical state |
| `resources` | Shots, evaluations, steps, stages, `num_treads` |

For an absent edge, `selected_weight=0`. `fixed_energy = -w (1 - σ) / 2`
and `expected_fixed_energy = -w (1 - c) / 2`. Sums of these contributions
do not replace the full solution energy or the reduction constant.
`score = abs(c)` is stored separately from the signed correlation.

With `status="stopped"`, the solver returns a complete valid solution
through `all_plus_reduced_fallback`, not a partial dictionary. No global
optimality is claimed.

### RQAOA plotting tools

| Function | Purpose |
|---|---|
| `tools.plot_reduction_weights(source, config=None)` | Selected-pair weight and mean absolute nonzero weight |
| `tools.plot_reduction_energy(source, config=None)` | `fixed`, `expected`, accumulated sums, `cut_offset`, `ising_constant` |
| `tools.plot_parameter_trajectories(source, config=None)` | All `binomial(2p, 2)` flat trajectories; `layout.panels_per_page` produces deterministic pages named `filename_page_001` |
| `tools.plot_parameter_trajectories_3d(source, config=None)` | Standalone HTML; Z is the reduction index |
| `tools.plot_analytic_gamma_profile(source, config=None)` | `min_beta E(gamma, beta)` and `beta_star(gamma)` for a selected step |

`source` is a solver, a `RunRecord`, or a `result` dictionary. Plots read
the stored record and never run the solver, optimizer, or new
measurements. With `save.format="html"`, the 3D plot embeds Plotly.js and
the data, so the file opens without a server or network. An empty history
raises an explicit error that names the required `collect` flag.

## Analytic p=1 source

`AnalyticP1Evaluator` computes single-layer moments without a full state
vector. `AnalyticP1CorrelationSource`, with type `qaoa_analytic_p1_cpu`,
organizes the angle search and returns the common `CorrelationResult`.

```python
from QA_max_cut_lib import AnalyticP1Evaluator, SolverFactory, tools

evaluator = AnalyticP1Evaluator(graph)
correlations = evaluator.correlations(0.3, 0.2, pair_domain="all_pairs")
profile = evaluator.energy_min_beta(0.3)
print(profile["beta_star"], profile["energy_min"])

config = tools.load_json("configs/solvers/rqaoa_analytic_p1_cpu.json")
algorithm = SolverFactory.from_config(config)
solution = algorithm.solve(graph)
```

### Formulas and scope

For `E = -C`, `J = w / 2`, the initial state `|+>`, and one layer,

\[
c_{uv} = a_{uv}\sin(4\beta) + d_{uv}\sin^2(2\beta),
\]

where `a_{uv} = 0.5 sin(gamma w_uv)(P_{u\setminus v} + P_{v\setminus u})` and
`d_{uv} = 0.5(P^-_{uv} - P^+_{uv})`. Products run over all `k` except `u`
and `v`. A non-adjacent pair can have a nonzero correlator because of
shared neighbors.

The mean energy is

\[
E(\gamma,\beta) = E_0 + A\sin(4\beta) + B\sin^2(2\beta),
\]

\[
\min_\beta E = F_0 - \operatorname{hypot}(F_s, F_c), \qquad
\beta_* = \frac14\operatorname{atan2}(-F_s, -F_c) \bmod \frac{\pi}{2}.
\]

Here `E0 = -0.5 Σw`, `F0 = E0 + B/2`, `Fs = A`, and `Fc = -B/2`. When
`R = 0`, `beta_star = 0`. The minimum over gamma is the best point of an
explicit finite grid, not a proven continuous global optimum.

### Evaluator parameters

| Argument or field | Value |
|---|---|
| `AnalyticP1Evaluator(graph, *, pair_batch_size=2048, dtype="float64")` | Dense `N×N` matrix; pair batches use `O(pair_batch_size * N)` memory |
| `parameters.layers` | Only `1` |
| `parameters.dtype` | Only `float64` |
| `parameters.pair_batch_size` | Positive integer, default `2048` |
| `optimizer.type` | `gamma_grid_exact_beta` |
| `optimizer.parameters.gamma_bounds` | Two increasing finite bounds, inclusive |
| `optimizer.parameters.points` | Integer `>= 2` |
| `optimizer.parameters.gamma_batch_size` | Positive planning batch size, default `4` |
| `estimation.mode` | Only `exact` |
| `correlations.mode` | Only `exact` |
| `objective.type` | Only `mean_cut` |

`initialization`, `step_initialization`, ADAM, `shots`, a full
distribution, `readout`, CVaR/Gibbs/Best-of-K/Threshold Probability, local
fields, and `p > 1` are rejected. The source does not build `2**N` and does
not support a discrete readout. An analytic run consumes no shots, so
`limits.shots_total=0` is allowed.

### Source result and RQAOA

`CorrelationResult` contains `node_order`, the task version, coverage,
physical correlations, `[gamma, beta]`, `layers=1`, and `training`:

| `training` field | Meaning |
|---|---|
| `gamma_profile.type` | `analytic_gamma_grid` |
| `gamma_profile.gamma_values` | Full grid, including both bounds |
| `gamma_profile.profile_energy` | `min_beta E` on the grid |
| `gamma_profile.profile_energy_original` | The same profile shifted by `-b_t` when `objective_reference="original_graph"` |
| `gamma_profile.beta_star` | Exact beta for every gamma |
| `gamma_profile.A`, `B`, `E0` | Energy coefficients |
| `gamma_profile.best_index`, `best_gamma`, `best_beta` | Deterministic best grid point |
| `classical_counters` | `gamma_points`, `coefficient_passes`, `pair_correlators`, `classical_seconds` |

The root `RQAOA` record keeps `shots_used=0`, `training_shots=0`,
`correlation_shots=0`, `readout_shots=0`, and also exposes `gamma_points`,
`coefficient_passes`, `pair_correlators`, and `classical_seconds`. The
profile enters `local_optimizer_history` with
`profile_type="analytic_gamma_grid"`.

The plotting tool `tools.plot_analytic_gamma_profile(source, config)` draws
`min_beta E(gamma, beta)` and `beta_star(gamma)` for a selected reduction
step. It reads `local_optimizer_history` and starts no new search. When no
profile exists, the error names `collect.local_optimizer_history`.

See also: [Mathematical conventions](MATHEMATICS.md).

## Generators

`GraphFactory.from_recipe(recipe)` accepts a schema-version-1 recipe:

| Field | Value or default |
|---|---|
| `schema_version` | `1` |
| `type` | `complete`, `gnm`, `erdos_renyi` |
| `parameters.n_vertices` | Required integer `0…1000` |
| `parameters.nodes` | List of `n` unique integers; defaults to `0…n−1` |
| `parameters.n_edges` | `gnm` only; integer `0…n(n−1)/2` |
| `parameters.probability` | `erdos_renyi` only; number in `[0, 1]` |
| `seed` | Topology seed; default `null` |
| `weights.type` | `constant` (default), `signed_discrete`, or `uniform` |
| `weights.parameters.value` | `constant` only; default `1.0` |
| `weights.parameters.values` | `signed_discrete` only; nonempty list of finite weights, sampled uniformly with replacement |
| `weights.parameters.low`, `.high` | `uniform` only; finite `low < high`; interval `[low, high)` |
| `weights.seed` | Separate weight RNG; defaults to the topology seed |

A recipe with `schema_version=2` selects the extended families
`complete`, `gnm`, `erdos_renyi`, `wishart_projected_v1`,
`hebbian_patterns_v1`, `gnm_density_v1`, `connected_path_fill_v1`,
`regular_v1`, `barabasi_albert_v1`, `watts_strogatz_v1`, `sbm_v1`,
`geometric_v1`, and `periodic_grid_v1`. Extended recipes also accept
`limits.max_vertices` and `limits.max_edges`, and store a resolved recipe
in the graph snapshot.

Equal fixed seeds produce equal graphs in the same software environment.
For long-term reproducibility, keep the actual snapshot: one recipe
without dependency versions does not promise bit-identical results in
every future NumPy release.

`partition(graph, max_vertices, *, seed=0)` returns a list of disjoint
induced subgraphs that cover every vertex. `max_vertices` is a positive
integer. Edges between parts are not included in the subgraphs; this is
not an exact decomposition of the original objective. An empty graph maps
to an empty list.

See also: [Generators and typed plans](GENERATORS_PLANS.md) for worked
examples.

## Common solver configuration

Unknown root fields are rejected. Algorithm-specific nested fields are
listed in the corresponding section.

| Field | Type or default | Purpose |
|---|---|---|
| `schema_version` | `int`, `1` | Schema version; any other value raises `ValueError` |
| `type` | `str` | Class type; may be omitted for a direct constructor, but must match when present |
| `seed` | `int` or `null`, `null` | Local NumPy RNG, recreated on every run |
| `num_treads` | `int`, `1` | CPU budget; values `> 1` require a solver with `supports_parallel=True`, otherwise `ValueError` |
| `parameters` | `dict` | Physical and computational parameters of the solver |
| `output.report` | `0/1/2`, `0` | No / brief / detailed terminal report |
| `output.logs` | `0/1/2`, `0` | No / start-and-end / additional step messages |
| `output.color` | `auto/always/never`, `auto` | ANSI color; `auto` enables it on a TTY |
| `collect` | `dict` | Optional arrays and histories; disabled by default |
| `save` | `dict`, absent | Common export after computation |
| `checkpoint.enabled` | `bool`, `false` | `true` raises `ValueError` for classes without a resume codec |

`report` and `logs` are independent. The JSON report does not depend on
the terminal level. Importing the package prints nothing and runs no
examples. Colors appear only in ANSI-capable terminals; JSON never
contains color codes.

## ExactCPU

| `parameters` field | Default | Constraint |
|---|---|---|
| `max_vertices` | `24` | Positive integer `<= 30`; a protective limit, not a speed guarantee |
| `batch_size` | `65536` | Positive integer; representatives per working block |
| `tie_tolerance` | `0.0` | Finite nonnegative absolute deviation of `C` for counting degenerate optima |

Training, readout, and a nonempty `collect` are rejected for this class
with `ValueError`. `limits.shots_total` is accepted for interface
uniformity; classical enumeration consumes no shots. The number of
examined representatives across both passes is `states_evaluated`. The
number of stages is the number of blocks in both passes.

Additional attributes: `worst_cut_value` (minimum `C`), `optimal_count`
(number of full spin solutions within `tie_tolerance`, counting the global
flip), and `states_evaluated`. Probabilities, state vector, and optimizer
history are not applicable and read as `None`. The termination reason is
`exhaustive_search`.

## QAOACPU

### Circuit and criterion

| Field | Default | Constraint |
|---|---|---|
| `parameters.layers` | `1` | Positive integer `p` |
| `parameters.max_vertices` | `20` | Positive integer; the simulator also enforces a hard cap of `24` |
| `parameters.dtype` | `complex128` | `complex128` or `complex64`; state shape `(2**N,)` |
| `objective.type` | `mean_cut` | `mean_cut`, `cvar`, `gibbs`, `best_of_k`, `threshold_probability` |
| `objective.parameters` | `{}` for `mean_cut` | `cvar`: `alpha ∈ (0, 1]`; `gibbs`: `eta > 0`; `best_of_k`: integer `k >= 1`; `threshold_probability`: finite `threshold` |
| `estimation.mode` | `exact` | `exact` or `shots` |
| `estimation.shots_per_evaluation` | `256` | Positive integer; used only in `shots` mode |
| `initialization.type` | `fixed_angles` | `fixed_angles` or `random_angles` |
| `initialization.parameters.values` | `p` values `0.5`, then `p` values `0.3` | `fixed_angles` requires exactly `2p` finite angles in radians |

`random_angles` takes no parameters: each γ is drawn independently from
`[−π, π)` and each β from `[−π/2, π/2)`. This is a heuristic
initialization window, not a claim about the period of a weighted problem.
Training does not restrict later angles to this window.

### Optimizer

| Field | Default | Purpose |
|---|---|---|
| `optimizer.type` | `adam` | Only `adam` in 0.1.0rc2 |
| `optimizer.parameters.max_steps` | `20` | Integer `>= 0`; `0` performs only the readout of the initial state |
| `.learning_rate` | `0.05` | Finite positive number |
| `.beta1`, `.beta2` | `0.9`, `0.999` | Moment coefficients in `[0, 1)` |
| `.epsilon` | `1e-8` | Positive stabilizing constant |
| `.gradient_tolerance` | `0.0` | Nonnegative gradient-norm threshold; `0` disables this stop |
| `optimizer.gradient.type` | `central_difference` | Central differences only |
| `optimizer.gradient.parameters.step` | `0.01` | Finite positive step `h` |

### Readout, budget, and collection

| Field | Default | Purpose |
|---|---|---|
| `readout.mode` | `most_probable` | Most probable basis state, or `best_of_shots` |
| `readout.shots` | Required for `best_of_shots` | Positive integer `M`; not allowed for `most_probable` |
| `limits.shots_total` | `null` | Total integer budget `>= 0`, or no limit |
| `collect.optimizer_history` | `false` | Observations at step centers |
| `collect.objective_history` | `false` | Every center evaluation and gradient probe |
| `collect.probabilities` | `false` | Full probability vector, `float64` |
| `collect.statevector` | `false` | Full complex vector of the selected `dtype` |
| `collect.correlations` | `false` | Physical `⟨Z_i Z_j⟩` matrix of shape `(N, N)`, `float64` |

Shots are samples of the classical simulator, without a hardware-noise
model. The final readout is reserved before training starts. A `stopped`
status with reason `shot_budget` is a correct early stop with a valid
solution. `objective_value=None` is valid when training performed no
evaluation.

### Results and attributes

Every field is readable directly, for example `algorithm.resources`, and
through `algorithm.result["resources"]`. Direct access returns independent
copies of arrays and dictionaries; fetch large data once. Missing optional
data is not computed on demand.

| Field | Type and meaning |
|---|---|
| `solution` | `dict[int, int]`, one `±1` per original vertex |
| `cut_value`, `solution_energy` | `float`: `C` and `-C` of the selected discrete solution |
| `expected_cut`, `expected_energy` | `float`: exact expectations of the final state; `None` for `ExactCPU` |
| `expected_cut_source` | For QAOA: `exact_final_statevector` |
| `objective_value`, `optimizer_loss` | Best evaluated `F` and `-F`; `None` if no evaluation occurred |
| `objective_source` | `best_training_evaluation` or `not_evaluated` |
| `optimal_angles` | `ndarray (2p,)`: best already-evaluated vector, or the initial vector when training is absent |
| `optimizer_history` | `list[dict]` or `None`; keys `optimizer_step`, `objective_evaluations`, `objective_value`, `optimizer_loss`, `gradient_norm`, `angles`, `shots_used` |
| `objective_history` | `list[dict]` or `None`; keys `evaluation_id`, `optimizer_step`, `kind`, `objective_value`, `optimizer_loss`, `angles`, `shots_used` |
| `probabilities`, `statevector`, `correlations` | Arrays of the shapes above, or `None` |
| `data_status` | `collected`, `not_collected`, or `not_applicable` for each additional data field |
| `node_order` | `list[int]` mapping matrix indices to external vertices |
| `graph` | Weighted snapshot and recipe of the graph actually solved |
| `config` | Fully resolved configuration, including defaults that were used |
| `status`, `termination_reason` | `completed`, `stopped`, `paused`, or `failed`, plus the specific reason |
| `resources` | Counters listed below |
| `timing.compute_seconds` | Snapshot creation and computation time up to the final record; excludes printing and file export |
| `report` | Structured quality, status, time, and resource summary |
| `started_at`, `finished_at` | ISO 8601, UTC |
| `run_id` | UUID of this run |
| `algorithm_type`, `package_version`, `schema_version`, `seed`, `num_treads` | Run provenance and settings |

QAOA `resources` includes `shots_used`, `training_shots`,
`readout_shots`, `objective_evaluations`, `optimizer_steps`,
`num_stages`, and `num_treads`. `num_stages` is the number of completed
optimization steps plus the final readout. Shots in `optimizer_history`
belong to training; the final `shots_used` also includes the readout.

The 0.1.0rc2 QAOA record additionally exposes budget-journal keys
`known_spent`, `uncertain_upper_charge`, `total_charged`,
`logical_training_shots`, `readout_reserved`, `shots_available`, and
`checkpoint_writes`, plus `best_evaluation_id`, `readout_completed`,
`resume_mode`, and `checkpoint_info`. A paused run reports
`status="paused"` and `termination_reason="checkpoint_pause"`.

## RunRecord and JSON

`RunRecord(fields)` copies the input data. `.get(name, default=None)` and
`record.attribute` return independent native values. `.field_names`
contains the available names. `.result` builds a complete JSON-compatible
snapshot. A record describes one run; it is not a log of every solver run.

An array in JSON is encoded as

```json
{"__type__": "ndarray", "dtype": "<f8", "shape": [2, 2], "data": [[1.0, 0.0], [0.0, 1.0]]}
```

For a complex array, `real` and `imag` replace `data`; `dtype` and `shape`
are retained. A complex scalar is
`{"__type__": "complex", "real": ..., "imag": ...}`. Spin-solution keys
become strings by the normal JSON rule and are restored as integers when
loaded. NaN/Inf and unknown types are never serialized silently: a
non-finite float raises `ValueError`, and an unknown object raises
`TypeError`.

`tools.load_result(path, *, load_original_data=False)` reads either a full
result or a selected export and returns a `RunRecord`. With `True` it also
loads the referenced `.npy` files from the same folder without pickle. If
data needed by a plotting tool was not exported or collected, the tool
reports its absence.

## Common export

`tools.save_result(source, config)` returns `SavedFiles` with `.primary`
(the JSON `Path`) and `.files` (a tuple of every published path). `source`
is a solver, a `RunRecord`, or a full JSON result dictionary.

| `save` field | Default | Rule |
|---|---|---|
| `path` | Required | Destination folder |
| `filename` | Required | Base name without a path or reserved characters |
| `atts` | All record fields | Unique attribute names for JSON; `[]` keeps metadata and report only |
| `save_original_data` | `[]` | Names of collected numeric `ndarray` fields to write as `filename_attribute.npy` |
| `makedir` | `false` | Create a missing parent path |
| `savedir` | `false` | Create a new `path/filename` folder and place the whole set there |

An array named in both lists is stored in both representations; the lists
are independent. Native export of other special types will be added with
the algorithms that produce them; an unsupported type raises an error.

The main JSON document contains `schema_version`, `export_type`
(`"selected_result"`), `saved_at`, `metadata`, `attributes`,
`original_data`, and `export_complete`. Metadata and the report are always
included when the record carries them, regardless of `atts` and output
levels. Every NPY reference is relative and records `file`, `format`,
`dtype`, and `shape`.

The complete set can be moved as one directory. The JSON file is published
last; simultaneous appearance of all files is not guaranteed. A normal
error removes the files published by the current attempt. An abrupt
process termination can leave extra files without the main JSON; they are
not a completed export.

An existing `path/filename` folder with `savedir=True`, even an empty one,
raises `FileExistsError`. With `savedir=False`, a conflict on any final
file is also rejected. A missing `path` with `makedir=False` raises
`FileNotFoundError`; a `path` that is not a directory raises
`NotADirectoryError`. Export never overwrites existing data, including
simultaneous attempts with the same name. The entire prepared set is
checked before publication. Pre-serialization can require extra memory
for large arrays; streaming export is not implemented in 0.1.0rc2.

See also: the repository-level ``RESULTS_AND_SAVE.md`` for the full export
workflow.

## Numerical tools

| Function | Arguments | Result |
|---|---|---|
| `correlation_matrix(probabilities)` | Normalized finite nonnegative vector of length `2**N` | Physical `⟨Z_i Z_j⟩`, `ndarray (N, N)`; the order is the bit order |
| `quality_score(cut_value, optimum, worst)` | Finite values for one task; `optimum >= worst` | `(cut_value − worst) / (optimum − worst)`, or `None` for a zero range |
| `load_json(path)` | UTF-8 JSON file | Ordinary parsed JSON |

`quality_score` does not check that the graph identities match; that is the
caller's responsibility. In 0.1.0rc2 the exact reference is produced by
`ExactCPU`; a GW reference and a best-known database are also supported by
the experiment layer.

## Parameter maps and plots

`scan_landscape(graph, config)` does not modify the graph and returns a
`RunRecord`. Configuration fields: `gamma` and `beta` are strictly
increasing finite lists of length `2…256`; `objective` is the criterion
described above; `quantity` is `optimizer_loss` (default) or
`objective_value`; an optional common `save` block is accepted. The graph
may have `1…16` vertices and `p=1` only. Record attributes: `gamma` `(G,)`,
`beta` `(B,)`, `values` `(B, G)`, `graph`, `config`, timestamps, `timing`,
and `resources`. `resources.grid_points` equals `B·G`; no shots are used.
The record type is `landscape_scanner_cpu` with termination reason
`grid_complete`.

All plotting functions return a Matplotlib `Figure`. `figure.qa_record` is
the tool result with configuration, provenance, `plot_data`, and a
timestamp; `figure.qa_saved_files` appears when a file is saved. Keep
`.qa_record` for your own tool history. Plotting never retrains a model
and never collects missing data.

| Function | Input and additional `config` fields |
|---|---|
| `plot_graph(source, config=None)` | `WeightedGraph` or a result; draws the input snapshot |
| `plot_graph_solution(source, solution=None, config=None)` | For a graph, pass the solution dictionary explicitly; for a solver or record, `solution` may be omitted |
| `plot_graph_info(source, config=None)` | Adds `N`, edge count, density, components, isolates, weight min/max, and mean absolute weight |
| `plot_probabilities(source, config=None)` | Requires `probabilities`; `selection.top_k=16`, a positive integer |
| `plot_optimization(source, config=None)` | Requires a nonempty `optimizer_history`; `axes.x`: `optimizer_step`/`objective_evaluations`/`shots_used`; `axes.y`: `optimizer_loss` (default)/`objective_value`/`gradient_norm` |
| `plot_landscape(source, config=None)` | A `scan_landscape` record; γ on X, β on Y, color is the stored quantity |
| `plot_ls_coverage(source, config=None)` | An LS-RQAOA record; two panels: coverage fraction and allowed/observed pair counts |

Common visualization fields:

| `config` field | Default or meaning |
|---|---|
| `title` | Tool-specific title |
| `style.text_engine` | `mathtext` or `latex`; default `mathtext` |
| `style.figsize` | `[10, 6]` inches; metadata adds caption space below |
| `style.label_size`, `.tick_size`, `.legend_size` | `19`, `16`, `15` |
| `style.latex_preamble` | Custom TeX preamble when `latex` is selected |
| `output.show` | `false`; `true` also calls `plt.show()` |
| `graph.layout` | `spring` (default) or `circular` |
| `graph.seed` | `0` for the spring layout |
| `graph.weight_labels` | `true` |

The probability chart preserves the original masses: top-K is not
renormalized, and the title reports the shown mass and the remainder. Blue
and orange vertices mean `+1` and `-1`; cut edges are emphasized and
negative edges are dashed. For a large dense graph, disable weight labels;
an edge filter and a specialized large-distribution chart are not part of
0.1.0rc2.

For a saved plot, `save` uses the common `path`, `filename`, `makedir`, and
`savedir` fields plus `format` (`png` by default; `jpeg`, `pdf`, `svg`),
`DPI` (`250`), and `metadata` (`false`). With `metadata=True`, the caption
below the figure shows the result timestamp, algorithm type and parameters,
graph parameters, and fingerprint. The full weighted graph remains in the
record; the visible caption does not print an edge table. For a run without
a recipe, the caption shows size and fingerprint, and the weights remain in
the stored snapshot.

Specifying `atts` or `save_original_data` in a plot `save` block adds a
sidecar JSON document for the tool result. In this version numeric series
live inside `plot_data`, so native export from that record does not resolve
nested fields by dotted paths. For a native landscape export, pass the
scanner `RunRecord` to `save_result` and select `values`.

Interactive HTML, 3D surfaces, and reduction trajectories are available for
stored `RQAOA`/`LS-RQAOA` trajectories through
`plot_parameter_trajectories_3d`; Plotly.js is embedded in the standalone
HTML.

## LS-RQAOA (`ls_rqaoa_cpu`)

`LSRQAOACPU.solve(graph, *, report=None, logs=None, num_treads=None,
control=None)` executes the same local-task plan sequentially when
`num_treads=1` and in a `spawn` process pool for a larger value. The explicit
`num_treads` argument overrides JSON. Requested and applied values appear
in `resources.num_treads_requested` and `resources.num_treads_applied`.

### Configuration sections

| Section | Required fields | Meaning |
|---|---|---|
| `parameters` | `finish_at`, `max_reductions` | Exact-residual size, default `6`, and optional global reduction limit |
| `subgraph_builder` | `type`, `parameters.max_size`, `parameters.rounds` | `random_partition` or `greedy_abs_weight_partition`; `boundary` only `induced_no_fields` |
| `correlation_source` | `type`, `objective_reference` | `qaoa_statevector_cpu` with `p >= 1`, `qaoa_analytic_p1_cpu`, `qaoa_ising_statevector_cpu`, or `qaoa_statevector_gpu` (only `LSRQAOAGPU`); the local objective is `working_graph` except for the ising source |
| `aggregation` | `type` | Only `mean`; average over the number of observations of each pair |
| `selection` | `type`, `parameters.pairs` | `max_abs_correlation`, `edges` or `all_pairs`, `zero_sign` |
| `coverage` | `require_full`, `empty_policy` | `require_full=false` and `empty_policy="anchor_pair"` by default; `anchor_pair` is the explicit repair task for an empty coverage |
| `update` | `type` | Only `refresh_all`: a new plan is built at the next step |
| `execution` | `backend`, `start_method`, `native_threads` | `process`, `spawn`, `1` |
| `budget_policy` | `type` | Only `step_round_robin` |
| `finisher` | `type`, `parameters.max_vertices` | The existing `ExactCPUFinisher` |
| `limits` | `shots_total`, `max_local_tasks_total` | Total shot limit and a cap on local calls per step |
| `collect` | flags | `reduction_history`, `parameter_trajectory`, `local_task_history`, `local_optimizer_history`, `correlation_tables`, `final_reduction_state` |

`full_graph_when_fits=true` on the builder creates one local task over the
whole residual when it fits within `max_size`; normal repeat passes are not
created in that mode. Duplicate sets in different rounds are separate tasks
with distinct IDs and seeds.

### Coverage and budgets

The allowed domain `A_t` is defined by `selection.parameters.pairs`. The
observed domain is the union of allowed pairs whose two endpoints appeared
in planned tasks. With `require_full=true`, incomplete coverage is detected
before training and raises `ValueError`. With `false`, selection uses only
observed pairs; a missing estimate is never treated as zero.

`budget_policy=step_round_robin` reserves the sum of mandatory correlation
shots for all tasks of a step, then grants full iterations one at a time to
each task in canonical order. For a state-vector shot estimate, one
iteration costs `(4p + 1) * M_train`; exact and analytic iterations cost
zero. Quotas are fixed before worker processes start. When the minimum is
unavailable, the step does not start, the graph does not change, a complete
solution is returned, and `status="stopped"`.

### LS results

Besides the common fields: `cut_offset`, `ising_constant`,
`reduced_solution`, `coverage_history`, `reduction_history`,
`parameter_trajectory` (rows of local tasks), `local_task_history`,
`local_optimizer_history`, `correlation_tables`, `final_reduction_state`,
and `budget_history`. A reduction event adds
`physical_correlation`, `correlation_origin="local_subgraph_mixture"`,
`expected_fixed_energy_local`, `local_task_ids`,
`aggregation_observations`, `coverage`, `global_expected_energy=None`, and
`global_expected_energy_reason="local_subgraph_mixture_not_global"`. A
local average is never presented as the global expectation of the full
graph.

Builders:

```python
from QA_max_cut_lib.generators import (
    RandomPartitionBuilder, GreedyAbsWeightPartitionBuilder,
)
builder = GreedyAbsWeightPartitionBuilder({
    "type": "greedy_abs_weight_partition",
    "parameters": {"max_size": 10, "rounds": 2, "full_graph_when_fits": True},
}, seed=42)
tasks = builder.build(graph, version=0)
```

`build()` returns `round`, `block`, `nodes`, `task_id`, and `boundary`; it
does not modify the graph and does not call QAOA. Both builders default to
`max_size=10`, `rounds=1`, `full_graph_when_fits=True`, and
`boundary="induced_no_fields"`. `max_size >= 2` is required.

See also: [Mathematical conventions](MATHEMATICS.md).

## Classical methods (`gw_cpu`, `simulated_annealing_cpu`, `refined`)

`GoemansWilliamsonCPU` (`gw_cpu`) solves the SDP with CVXPY/SCS and rounds
it with independent hyperplanes. `parameters` accepts `num_hyperplanes`
(default `10`), `solver` (only `"SCS"`), `eps_abs` (`1e-6`), `eps_rel`
(`1e-6`), `max_iters` (`10000`), `allow_inaccurate` (`false`),
`psd_tolerance` (`1e-6`), `diag_tolerance` (`1e-6`), `repair_tolerance`
(`1e-6`), and `time_limit_secs` (`null`). `limits.sdp_time_limit` is
`null` by default. `collect` accepts `candidate_history`, `sdp_matrix`,
`sdp_factor`, and `roundings`, all `false` by default.

`sdp_upper_bound` is `null`, and `sdp_upper_bound_reason` states that no
strict dual certificate is built; the numerical relaxation value is never
presented as a certified bound. `positive_weight_upper_bound` and
`bound_kind` are also reported. Nonpositive tolerance values, a non-`SCS`
solver, and unknown fields raise `ValueError`; a non-boolean
`allow_inaccurate` also raises `ValueError`.

`SimulatedAnnealingCPU` (`simulated_annealing_cpu`) builds a spin BQM with
`h=0`, `J=w/2`, `offset=-sum(w)/2` and calls the classical
`dwave.samplers.SimulatedAnnealingSampler`, one logical read per call.
`parameters`: `num_reads` (`10`), `num_sweeps` (`64`), `beta_schedule_type`
(`linear` or `geometric`), `beta_range` (`null` or an increasing finite
pair). `limits`: `anneal_reads_total`, `anneal_sweeps_total` (both `null`
by default). `collect.candidate_history` defaults to `false`. These are
annealing reads, not quantum shots; `resources.shots_used` is `0`.

`ExactBlockRefiner` (`exact_block`) exposes
`refine(graph, spins, *, context=None)` and returns an improved independent
copy. `parameters`: `max_size` (`16`, integer `1…22`), `batch_size`
(`4096`), `max_passes` (`1`), `max_blocks_total` (`null`),
`improvement_atol` (`1e-10`), `improvement_rtol` (`1e-12`), and
`max_states_per_block` (`2**20`). `selector` is `explicit_blocks` or
`pair_greedy_abs_weight` with `max_seed_pairs` and `ranked_pairs`
(`null` by default); `pair_greedy_abs_weight` requires `max_size >= 2`.
Blocks are applied sequentially; fields are recomputed after each accepted
block. `limits.refinement_states_total` reserves the full `2**|S|`
enumeration of a block before the block starts. The returned solution is
never worse than the input within the configured acceptance tolerance.

`RefinedSolver` (`refined`) requires `base_solver` and `refiner`;
`base_solver` must be a mapping and must not itself be `refined`
(`ValueError`). `base_run` and `refinement_run` are independent records.
Final `cut_value` and `solution` refer to the improved assignment; quantum
angles and `expected_energy` remain in `base_run`. `limits` accepts
`anneal_reads_total`, `anneal_sweeps_total`, and
`refinement_states_total`. `collect` accepts `base_run`,
`candidate_history`, and `refinement_history` (all `true` by default). The
result also exposes `base_cut_value`, `base_energy`, `gain`,
`base_solution`, and `solution_source` (`refined` or
`refined_negative`).

## QAOA-in-QAOA (`qaoa_in_qaoa_cpu`, `qaoa_in_qaoa_gpu`)

`QAOAInQAOACPU` and `QAOAInQAOAGPU` accept `parameters.finish_at`
(default `4`), `parameters.max_levels` (`null` by default), `partitioner`,
`local_solver`, `execution`, `budget_policy`, `finisher`, `limits`, and
`collect`. Blocks of the current level are disjoint and cover all of its
vertices. A spin pattern is fixed inside each block; each block anchor
becomes a vertex of the next meta-graph. `GraphReducer` applies exact
constraints `s_u = r_u s_anchor` to the whole graph.

`local_solver` is one of the explicit adapters:

- `qaoa_cpu` (`QAOACPU`), with `most_probable` or `best_of_shots`;
- `qaoa_gpu` (`QAOAGPU`), only in `qaoa_in_qaoa_gpu`;
- `exact_cpu`, an explicit test component, not a quantum run.

The local solver has no root `limits`, `save`, `output`, or `seed`. The
coordinator passes a stable seed and a fixed quota. `execution.backend`
must be `"process"` for the CPU hierarchy and `"gpu_owner"` for the GPU
hierarchy; `start_method` is `"spawn"` and `native_threads` is `1`.
`budget_policy.type` is `"level_round_robin"`: it first covers every
level minimum, then distributes full iterations. `best_of_shots`
reserves `M` for each readout.

`hierarchy_history` contains the level, blocks, anchors, task IDs,
`raw_pattern`/`normalized_pattern`, `delta_cut_offset`,
`cut_offset_before`, `cut_offset_after`, quotas, and timings.
`hierarchy_tree` uses composite IDs `L{level}:A{anchor}`; leaves of
original vertices are `leaf:{node}`. Angles are stored per local task in
`parameter_trajectory`; they are never averaged into a global vector.

`limits.max_local_tasks_total` and `parameters.max_levels` produce a
complete `hierarchy_all_plus_fallback` with `status="stopped"`. Partition,
projection, worker, and GPU errors are never masked as a limit. `finish_at`
is solved by `ExactCPUFinisher`; with `finish_at=0` a small residual is not
promised.

## NVIDIA GPU QAOA (`qaoa_gpu`)

`QAOAGPU` accepts `parameters.layers` (`1`), `parameters.max_vertices`
(`30`), `parameters.dtype` (`complex64`), and `parameters.representation`
(`spin_flip_pairs` or `full`). `gpu` contains `device_id` (`0`),
`diagonal_dtype` (`float64`), `accumulation_dtype` (`float64`),
`memory.fraction_of_available` (`0.8`), `memory.reserve_bytes`
(`268435456`), `kernels.mixer="pairwise"`, `kernels.cuda_graphs=false`, and
`state_batch_size` (`1`).

The GPU path does not import CuPy when the package is imported. Without
CuPy or a device, a call fails with `RuntimeError`; CPU solvers never switch
to NumPy silently. The memory preflight runs before the diagonal and the
state are created. `state_representation` contains `type`, `n_vertices`,
`node_order`, `dtype`, `amplitude_kind`, `probability_kind`,
`representative_rule`, `normalization`, and device information.

Collection in pair mode:

- `collect.compressed_statevector=true` stores `chi`;
- `collect.pair_probabilities=true` stores `q`;
- `collect.statevector=true` or `collect.probabilities=true` is rejected
  until the caller selects the full representation. This prevents `chi`
  from being returned silently as a physical `psi`.

`gradient.type="central_difference"` supports every criterion and
`exact`/`shots`. `gradient.type="adjoint"` supports exact Mean Cut only;
any other criterion or `shots` mode is rejected with `ValueError`. The
adjoint computes a reverse pass over the state and does not pretend to be
`4p+1` measurements.

`QAOAGPUCorrelationSource` (`qaoa_statevector_gpu`) uses its own `gpu`
section and returns a normal `CorrelationResult` with original labels and
the task version. `RQAOAGPU` uses the same reducer and finisher;
`LSRQAOAGPU` requires `execution.backend="gpu_owner"` and serves local
tasks sequentially on one device.

See also: [NVIDIA GPU guide](GPU.md).

## Module execution

Every implemented solver module provides `main(argv=None)` and an
`if __name__ == "__main__":` entry point. CLI arguments: `--graph` (graph
recipe JSON), `--config` (solver JSON), `--report` and `--logs` (`0/1/2`).
Without files, the module runs a small reproducible example: a complete
six-vertex graph with signed weights, topology seed `10`, weight seed `11`,
and solver seed `42`. These demonstration calculations never run at import
time.

## Experiments and profiling

```python
from QA_max_cut_lib import BenchmarkSpec, ExperimentRunner, ParallelismProfiler
from QA_max_cut_lib import tools

spec = BenchmarkSpec(tools.load_json("experiments/benchmark_small.json"))
plan = spec.plan()
print(plan.summary)
series = ExperimentRunner(spec).run()
profile = ParallelismProfiler(spec, tools.load_json("tools/parallelism_sweep.json")).run()
```

`BenchmarkSpec` accepts a JSON-compatible dictionary and validates a closed
schema. Top-level fields are `schema_version` (only `1`), `plan_seed`,
`trial_seeds`, `graph_cases`, `solver_templates`, `solver_cases`,
`references`, `execution`, `measurement`, `series_limits`, `metrics`,
`failure_policy`, `collect`, `save`, `planner_limits`, and `profile`.
`planner_limits.max_cases` defaults to `50000` and
`planner_limits.max_trials` to `200000`. `metrics.success.type` is
`fraction_of_exact_max` or `absolute_cut`. `failure_policy.on_trial_error`
is `continue` (default) or `fail_fast`.

`plan()` does not call a solver. `ExperimentPlan.result` contains
`counts`, `graph_instances`, `cases`, `trials`, `reference_jobs`,
`rejected`, `seeds`, `execution`, `measurement`, `series_limits`,
`summary`, and `profile`. `ExperimentPlan.plan_hash` is a separate hash
property. `counts` includes
`graph_instances`, `cases`, `raw_solver_configs`,
`accepted_solver_configs`, `rejected_solver_configs`, `trials`,
`reference_jobs`, and `planned_top_level_runs`.

`ExperimentRunner.run()` returns `ExperimentSeries` with `result`, `trials`,
`references`, `analysis`, `budget`, `environment`, and `save_status`. The
result document has `record_type="experiment_series"`, `mode`, `summary`,
`plan`, `references`, `trials`, `analysis`, `errors`, `environment`,
`budget`, `timing` (`wall_seconds`, `makespan_seconds`), and `save`.

`ParallelismProfiler.run()` returns `ParallelismProfile` with `result`,
`measurements`, `aggregate`, `per_solver`, `recommended_num_treads`,
`analysis["fit_amdahl"]`, and `analysis["uncertainty"]`. Its
`ProfileSpec` requires `num_treads_values` to be a nonempty list that
includes `1`; defaults include `warmup_runs=0`, `repetitions=1`,
`execution_mode="isolated"`, `timing_scope="benchmark"`,
`measurement_order="shuffle_within_case_repetition"`,
`measurement_order_seed=2027`, `models=["amdahl", "karp_flatt"]`, and
`recommendation.criterion="total_time"`.

`ReferencePolicy` accepts `type` (`tiered` by default; also `exact`, `gw`,
`best_known`, `none`), `exact`, `gw`, and `best_known`. The policy returns
`ReferenceJob` objects; `run_job` creates an independent reference record.
`exact.max_vertices` is `0…30` with `include_minimum=false`; `gw` uses
`num_hyperplanes=10` by default. The exact minimum is computed by a
separate run on the graph with weights `-w`. A GW record has
`proof="heuristic"`; `sdp_upper_bound` is not invented.

CLI:

```bash
python -m QA_max_cut_lib.tools.experiments plan --config experiments/benchmark_small.json
python -m QA_max_cut_lib.tools.experiments run --config experiments/benchmark_small.json
python -m QA_max_cut_lib.tools.parallelism --benchmark experiments/benchmark_small.json --config tools/parallelism_sweep.json
```

`plan` also accepts `--json`; `run` accepts `--reuse-completed`. The
parallelism CLI accepts `--plan-only` and `--json`.

See also: [Experiment series and parallelism](EXPERIMENTS.md).

## Checkpoints

`QAOACPU.solve(..., control=RunControl(pause_after={"optimizer_iterations": 2}))`
returns a complete anytime assignment and sets `last_checkpoint`.
`tools.inspect_checkpoint(path)` reads a manifest without constructing a
solver. `restore(path, *, overrides=None)` creates the solver object;
`resume()` continues the same canonical run. A new `solve()` on a restored
object starts a new run. Checkpoint events supported by `QAOACPU` are
limited to `optimizer_iterations`.

```python
from QA_max_cut_lib import RunControl, restore, tools

solver = restore("checkpoints/run/checkpoint.json")
print(tools.inspect_checkpoint("checkpoints/run/checkpoint.json"))
solution = solver.resume()
```

`RunControl(pause_after=None, *, request_pause=False)` stores a mapping
from event name to an absolute completed count; counts must be nonnegative
integers, otherwise `ValueError`, and a non-mapping raises `TypeError`.
`request_pause()` and `request_cancel()` set a flag for the next supported
boundary. `should_pause(event, completed)` returns `True` once per reached
trigger; a consumed trigger does not fire again after resume.
`snapshot()` and `from_snapshot(data)` serialize the control.

`CheckpointManager` exposes `write_generation`, `load`, and
`list_generations`. `ResourceJournal` exposes `begin`, `commit`, `resolve`,
`read`, and `reconcile`. `inspect_checkpoint` returns `schema_version`,
`checkpoint_id`, `run_id`, `generation`, `phase`, `component_id`,
`package_version`, `next_operation`, `array_payloads`, `journal_cutoff`,
and `path`.

`restore` overrides may contain only `output` and `checkpoint`; any other
key raises `CheckpointIncompatible`. A missing component codec, a changed
config or graph fingerprint, or an unsupported checkpoint format raises
one of `CheckpointCorrupt`, `CheckpointIncompatible`, or `CheckpointLocked`,
all subclasses of `CheckpointError`.

In 0.1.0rc2, resume codecs are registered for `qaoa_cpu`,
`laboratory.warm_start_qaoa_cpu`, `laboratory.block_mean_field_qaoa_cpu`,
`laboratory.distributed_ls_rqaoa_cpu`, and `laboratory.qaoa_mi_cpu`.
Other components report `supports_checkpoint=False`; enabling
`checkpoint.enabled` for them raises `ValueError`.

Checkpoint configuration fields when enabled: `path` (required), `format`
(only `json_npy_v1`), `every` (subset of `optimizer_iterations`,
`reductions`, `levels`, `local_tasks`, positive integers),
`at_phase_boundaries` (`true`), `on_pause` (`true`),
`durable_resource_journal` (`true`), `durability` (`local_fsync` or
`best_effort`), `on_write_error` (`stop` or `raise`), and `retention`
(only `{"type": "keep_all"}`).

CLI:

```bash
python -m QA_max_cut_lib.tools.checkpoints inspect --checkpoint PATH
python -m QA_max_cut_lib.tools.checkpoints resume --checkpoint PATH --report 1
```

See also: [Checkpoints](CHECKPOINTS.md).

## Laboratory and warm start

```python
from QA_max_cut_lib import GraphFactory, tools
from QA_max_cut_lib.laboratory import WarmStartQAOACPU

graph = GraphFactory.from_recipe(tools.load_json("configs/graphs/warm_start_4.json"))
config = tools.load_json("configs/solvers/warm_start_qaoa_cpu_provided.json")
solver = WarmStartQAOACPU(config)
solution = solver.solve(graph)
```

`type="laboratory.warm_start_qaoa_cpu"`. The `warm_start` section is
required:

```json
{
  "epsilon": 0.25,
  "source": {"type": "provided_spins", "parameters": {"entries": [{"node": 10, "spin": -1}]}},
  "mixer": {"type": "rounded_modified"}
}
```

`epsilon` defaults to `0.25` and must lie in `[0, 0.5]`; it is frozen for
the run. Sources: `provided_spins`, `provided_probabilities`,
`classical_solver` (`gw_cpu`, `simulated_annealing_cpu`, `exact_cpu`).
Mixers: `x`, `warm_aligned`, `rounded_modified`. Source entries must cover
every label of the current graph exactly once, without duplicates
(`WarmStartError`, a `ValueError` subclass, otherwise). A classical child
is not run in the constructor; its `source_cost` is reported separately.

`result` additionally contains `variant_id`, `warm_start_data` (`q_raw`,
`q_clipped`, `warm_start_angles`, `node_order`), `mixer_id`, `source_type`,
`source_cut`, `source_run`, `source_cost`, `source_cost_status`,
`initial_expected_cut`, and `initial_expected_energy`. `source_cut` is a
discrete seed cut, not the expectation of the product state.
`warm_start_angles` are not part of `optimal_angles` and are not trainable
parameters.

Catalog:

```bash
python -m QA_max_cut_lib.laboratory list
python -m QA_max_cut_lib.laboratory show --id warm_start_product_qaoa_cpu_v1
```

`list` and `show` do not require CuPy or CVXPY. `list` also accepts
`--implemented-only` and `--json`; `show` accepts `--json`.

See also: [Laboratory catalog and warm start](LABORATORY.md).

## Warm-start ensemble RQAOA

`WarmStartRQAOACPU` (`type="laboratory.warm_start_rqaoa_cpu"`) uses
`correlation_source.type="warm_start_ensemble_cpu"`. The source accepts
`candidate_source` (`gw_rounds_cpu` or `provided_by_graph`), discrete
starts, `warm_start`, objective/estimation/optimizer/initialization, and
`correlations` with `aggregation="uniform_components"`.

```python
from QA_max_cut_lib.laboratory import WarmStartRQAOACPU
solver = WarmStartRQAOACPU(config)
solution = solver.solve(graph)
print(solver.warm_start_history)
```

The result additionally contains `warm_start_history`, `variant_id`,
`aggregate_statevector=None`, and
`aggregate_statevector_status="not_applicable_statistical_mixture"`.
Classical preprocessing cost is in `resources.source_cost`, with
`resources.source_cost_status`. Checkpoint/resume is not available for this
component in 0.1.0rc2.

## Incremental LS-RQAOA

`QA_max_cut_lib.laboratory.IncrementalLSRQAOACPU` /
`type="laboratory.incremental_ls_rqaoa_cpu"` accepts a compact recipe:
`initial_builder`, `update`, `bank`, `selection`, `correlation_source`,
`local_problem`, `finisher`, `execution`, `limits`, top-level `finish_at`
(default `min(6, max(1, len(graph)))`), `pairs` (`edges` by default),
`zero_sign` (`1`), and `require_full_coverage` (`false`). It keeps
`solve(graph, *, report=None, logs=None, num_treads=None, control=None)`,
returns a complete dictionary from original labels to `±1`, and provides an
independent `RunRecord`.

Main result fields: `epoch_history`, `bank_history`, `selection_history`,
`coverage_history`, `budget_history`, `local_task_history`,
`final_bank`, `reductions_completed`, `remaining_node_order`, `cut_offset`,
`ising_constant`, `reduced_solution`, `reduced_cut_value`,
`solution_source`, and `resources`. `selection.physical_correlation` and
`selection.score` are different quantities.

See also: [Incremental LS-RQAOA](INCREMENTAL_LS.md).
