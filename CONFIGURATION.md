# Configuration: from a JSON file to a running solver

Every solver and graph in QA_max_cut_lib `0.1.0rc2` is described by a
JSON-compatible dictionary. This page shows how a configuration is read,
validated, and executed, which fields are shared by all solvers, and which
mistakes are rejected before any expensive work starts.

By the end you can run `configs/solvers/qaoa_cpu.json` on
`configs/graphs/complete_signed_7.json` and check the cut it returns.

## Run a configuration file end to end

```python
from QA_max_cut_lib import GraphFactory, SolverFactory, tools

recipe = tools.load_json("configs/graphs/complete_signed_7.json")
config = tools.load_json("configs/solvers/qaoa_cpu.json")

graph = GraphFactory.from_recipe(recipe)
solver = SolverFactory.from_config(config)
solution = solver.solve(graph)

print(graph.nodes)
print(solver.cut_value)
print(round(solver.expected_cut, 6))
```

Expected output with the committed files and `seed=42`:

```text
(10, 20, 30, 50, 80, 130, 210)
6.0
3.857348
```

`cut_value` is the weight of the returned partition; `expected_cut` is the mean
cut of the final QAOA state and is not required to equal `cut_value`. The
30 optimizer steps come from the file, not from a default.

## The two configuration families

| Family | Used by | Schema |
|---|---|---|
| Solver config | `SolverFactory.from_config`, every `*CPU`/`*GPU` class | `schema_version: 1` |
| Graph recipe | `GraphFactory.from_recipe` | `schema_version: 1` or `2` |

Solver config schema 1 is the historical family. Graph schema 1 supports
`complete`, `gnm`, and `erdos_renyi`. Graph schema 2 adds extended generators
and an explicit `limits` block; the resolved metadata is stored on the graph.

`SolverFactory.from_config` accepts a mapping only. Read a file with
`tools.load_json(path)` first. The factory never changes the algorithm to make a
configuration valid; an unknown `type` is an error.

## Fields shared by solver configurations

| Field | Meaning | Notes |
|---|---|---|
| `schema_version` | Configuration schema | must be `1` |
| `type` | Registry name | must match the class or the factory entry |
| `seed` | Initialization seed | `None` is allowed; a fresh run resets it |
| `num_treads` | CPU worker budget | `1` unless the solver is parallel |
| `parameters` | Algorithm parameters | e.g. `layers`, `dtype`, `max_vertices` |
| `objective` | Training criterion | `mean_cut`, `cvar`, `gibbs`, `best_of_k`, `threshold_probability` |
| `estimation` | Exact or shot-based evaluation | `mode` is `exact` or `shots` |
| `optimizer` | Optimizer and gradient | names and defaults are in `API.md` |
| `initialization` | Starting angles | e.g. `fixed_angles` |
| `readout` | How a spin assignment is selected | e.g. `best_of_shots` |
| `limits` | Hard resource ceilings | e.g. `shots_total` |
| `collect` | Which optional fields to retain | arrays and histories |
| `output` | `report`, `logs`, `color` levels | `0`, `1`, or `2`; `color` is `auto`, `always`, or `never` |
| `save` | Export configuration | opt-in; see `RESULTS_AND_SAVE.md` |
| `checkpoint` | Pause/resume configuration | opt-in; see `CHECKPOINTS.md` |

Concrete classes may add a small, closed set of extra fields. For example
`QAOACPU` accepts `checkpoint`, `WarmStartQAOACPU` accepts `warm_start`, and
`BatchLSRQAOACPU` accepts `batch`, `messages`, `local_problem`, `source`,
`finisher`, and `max_reductions`. A field outside the allowed set raises
`ValueError: Unknown solver fields: {...}`.

## Validation order

The base class validates in this order before the solver builds a state:

1. The configured `type` matches the class.
2. No unknown top-level fields are present.
3. `schema_version` is `1`.
4. A checkpoint section is valid and the component supports checkpointing.
5. `output` contains only `report`, `logs`, `color`, each with an accepted value.
6. `num_treads` is a positive integer and is supported by this solver.
7. The algorithm-specific `_validate` accepts its parameters and the graph.

A rejected configuration leaves no partial run and no output files.

## Graph recipes and labels

```json
{
  "schema_version": 1,
  "type": "complete",
  "seed": 1200,
  "parameters": {"n_vertices": 4, "nodes": [-7, 10, 80, 101]},
  "weights": {
    "type": "signed_discrete",
    "seed": 1201,
    "parameters": {"values": [-2, -1, 1, 2]}
  }
}
```

- `nodes` are external integer labels. They are never renumbered silently.
- Insertion order is state order: the first node is the least-significant bit in
  `probabilities` and in the statevector.
- Topology and weights have separate seeds; changing the weight seed does not
  change the edge set.
- `n_vertices` must be an integer in `[0, 1000]` for schema 1. Schema 2 recipes
  add limits and extended generators such as `wishart_projected_v1` and
  `hebbian_patterns_v1`.

An unknown generator, an unknown recipe field, or a `nodes` list whose length
differs from `n_vertices` raises `ValueError` before a graph is built.

## Conventions that affect results

- Edge weights are signed. Max-Cut uses
  `C(s) = sum w_ij (1 - s_i s_j) / 2`; a negative edge rewards keeping its
  endpoints on the same side.
- `solution_energy = -cut_value`. The constant term is retained where exact
  comparisons or reductions need it.
- Source correlations are `c_ij = <Z_i Z_j>`. Decision scores, such as the
  output of `odds_fusion`, are a separate quantity and are not physical
  correlations.
- `num_treads` is a CPU budget, never a CUDA launch dimension.
- A checkpoint is resumable state; a result record is a scientific artifact.
  They are not interchangeable.

## Override output and threads at run time

`solve()` accepts explicit `report`, `logs`, and `num_treads` arguments. They
override the values stored in `output` for that call:

```python
solution = solver.solve(graph, report=1, logs=0, num_treads=1)
```

The solver copies its configuration at construction, so mutating the dictionary
you passed in does not change an existing solver.

## Where configurations live

- `configs/solvers/` — solver configurations, one file per algorithm variant;
- `configs/graphs/` — graph recipes used by examples and tests;
- `configs/plots/` — plotting style and figure configuration;
- `experiments/` — benchmark series with graph and solver cases.

## Common failures

| Message | Cause | Fix |
|---|---|---|
| `Solver '...' is not implemented in 0.1.0rc2` | unknown `type` | call `SolverFactory.available()` and use a listed name |
| `Unknown solver fields: {...}` | misspelled or unsupported key | remove it or use the class-specific field |
| `Unsupported configuration schema.` | `schema_version` is not `1` | solver configs use schema 1 |
| `Checkpoint/resume is not implemented for ...` | `checkpoint.enabled` on an unsupported component | see the capability table in `CHECKPOINTS.md` |
| `Parallel CPU execution is not implemented for this solver` | `num_treads > 1` on a sequential solver | use `num_treads=1` |

## Limitations

- Solver configurations are closed: unknown fields are rejected rather than
  ignored.
- Only one solver registry is loaded; a plugin without a registered type cannot
  be selected by name.
- Configuration validation does not prove that a requested shot budget is
  sufficient for convergence; it only enforces declared limits.
