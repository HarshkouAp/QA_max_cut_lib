# Warm-start RQAOA: an ensemble of starting points

`WarmStartRQAOACPU` (`type="laboratory.warm_start_rqaoa_cpu"`) runs one RQAOA
reduction per step, but replaces the single measured state of the step with an
ensemble of warm-started children. Given a graph and a source of candidate
partitions, it returns the original-label assignment, the per-step history, and the
classical cost of preparing the candidates. This page explains what each child
contributes, how children are merged, how the shared shot budget is divided, and
which numbers you can check on the four-vertex example.

The assembly is a heuristic. It is not a decomposition of the original objective:
the mixture is a statistical average, and a child has no discrete readout of its
own.

## What one reduction step does

On the current graph `G_t`:

1. `candidate_source` produces one finite batch of starting partitions:
   `gw_rounds_cpu` returns Goemans-Williamson roundings, `provided_by_graph`
   returns explicit assignments recorded for this snapshot.
2. Candidates are validated, canonicalized with global-flip deduplication, ranked
   by their cut on `G_t`, and truncated to `max_starts`.
3. Each selected start gets its own product state and its own independent warm
   training task. A child has no discrete readout.
4. Physical `ZZ` correlations of the children are averaged with equal weights.
5. One shared selector and one reduction are applied to `G_t`.
6. On the next graph the source batch is built again; old `q`, probabilities, and
   reduction factors are not reused.

The class supports CPU full statevector execution and an exact terminal finisher.

## Candidate sources

Two source types exist, and they differ in what they promise.

```json
{"type": "gw_rounds_cpu",
 "parameters": {"num_hyperplanes": 16, "solver": "SCS",
                "allow_inaccurate": false}}
```

`gw_rounds_cpu` solves one SDP per graph and performs `L` hyperplane roundings,
each carrying a `round_id`/`source_id`. It returns the roundings that actually
occurred, duplicates included: asking for `L` roundings does not promise `L`
distinct cuts. `candidate_roundings` is an extra field of the GW result that only
an ensemble source requests. This is a classical source; its primal value is not a
certificate of the global optimum, and the GW guarantee for non-negative Max-Cut
does not transfer once reductions make weights negative.

```json
{"type": "provided_by_graph",
 "parameters": {"records": [
   {"graph": {"nodes": [10, 30, 80, 101],
              "edges": [[10,30,2],[10,80,3],[30,80,5]],
              "fingerprint": "..."},
    "assignments": [
      {"candidate_id": "A", "spins": {"10": 1, "30": 1, "80": -1, "101": 1}}
    ]}
 ]}}
```

A record is used only when both the node order and the fingerprint match the
current graph. A missing record is an error: no hidden GW call and no stale seed is
substituted, and the run raises `WarmStartEnsembleError`. The same error type is
raised when `on_shortage="error"` and the source finds fewer unique starts than
`max_starts`. This is the source used by the checked example below.

## Deduplication, ranking, and the mixture

For zero-field Max-Cut the canonical anchor is the smallest external label. If that
spin is `-1`, the whole assignment is inverted. The uniqueness key is the tuple of
spins in canonical order, so `s` and `-s` are one start: multiplicity is recorded
but does not increase the weight of the child. Ranking is `-C_t`, then the canonical
tuple, then the source ID — a total order, not an `isclose` comparison that could
be non-transitive.

For `K_t` children the mixture is defined by

```text
P_bar(b) = (1/K_t) sum_l |<b|psi_l>|^2
c_bar_uv = (1/K_t) sum_l <Z_u Z_v>_l
```

The density matrix `rho_t = (1/K_t) sum_l |psi_l><psi_l|` is never materialized, and
no single pure statevector is created for the mixture: the result reports
`aggregate_statevector=null` with status `not_applicable_statistical_mixture`.
Correlations are therefore an average of physical correlators, not correlators of
an averaged state.

## Shot budget

There is one root `ShotBudget`. With `K_t` children, `M_corr` correlation shots per
child, `p` layers, and `M_train` shots per training evaluation:

```text
R_corr = K_t * M_corr
a      = (4p+1) * M_train
```

The correlation reserve is checked before any quantum child starts. Training
iterations are then distributed round-robin in canonical child order. `K_t` is
never reduced to make training fit; if the budget covers only the reserve, every
child gets zero training iterations. If even the reserve does not fit,
`ShotBudgetExceeded` is raised and no child starts. A completed run satisfies
`shots_used = training_shots + correlation_shots`.

## Running example

The checked case uses four vertices `10, 30, 80, 101` with edges
`(10,30,2)`, `(10,80,3)`, `(30,80,5)` and two distinct recorded starts:

```text
A = (+1, +1, -1, +1)   C(A) = 8
B = (+1, -1, +1, +1)   C(B) = 7
```

The record also contains `-A` and a duplicate of `A`. After global-flip
canonicalization the source reports `unique=2`, and both starts are selected with
multiplicities `3` and `1`. With exact correlations per child, the uniform mixture
gives

```text
c(10,30) = 0,  c(10,80) = 0,  c(30,80) = -1
```

`max_abs_correlation` selects the pair `(30,80)` with `sign=-1`. The reduction adds
the offset `b=8` and rewrites `w'(10,30) = 2 + (-1)*3 = -1`. The exact finisher
then returns the assignment `{10: 1, 30: 1, 80: -1, 101: 1}` with

```text
cut_value = 8.0     cut_offset = 8.0     ising_constant = -7.5
```

These values are asserted by `tests/test_warm_start_rqaoa.py`; the correlation
table is available as `result["correlation_tables"][0]`.

To run the shipped configuration instead of the recorded one:

```bash
python -m QA_max_cut_lib.laboratory.warm_start_rqaoa_cpu \
  --graph configs/graphs/warm_start_4.json \
  --config configs/solvers/warm_start_rqaoa_cpu_gw.json
```

With `seed=20261102`, `finish_at=2`, and `max_reductions=2`, that run reports
`Cut: 2`, `Energy: -2`, two completed reductions, `cut_offset=-4.0`,
`ising_constant=1.0`, and `shots_used=1536` (1440 training, 96 correlation). Its
`resources.source_cost` records one SDP call over 16 hyperplanes. The two examples
are different graphs, so their cut values are not comparable; the first is a
contract check, the second an end-to-end run.

## Series

`experiments/benchmark_warm_start_rqaoa.json` runs `2` graph instances × `2` trial
seeds × `3` recipes — `rqaoa_cpu`, `ws_rqaoa_k1`, `ws_rqaoa_k2` — for `12` trials,
all with `finish_at=2`, `p=1`, `max_steps=3`, and 32 shots per evaluation. This is
not a fixed-total-budget comparison: `k2` performs strictly more work per step than
`k1`. The exact reference is computed as a separate reference job.

## Limits

- No checkpoint/resume. `supports_checkpoint` is not declared for
  `laboratory.warm_start_rqaoa_cpu`, and a configuration with `checkpoint.enabled`
  is rejected rather than silently ignored.
- CPU full statevector only. The GPU, `chi`, and analytic `p=1` paths are not
  available through this assembly.
- No adaptive weights, no transfer of starts between reductions, no simulated
  annealing pools, and no fractional ensemble members.
- `gw_rounds_cpu` needs CVXPY/SCS, and its primal value is not a global-optimum
  certificate.
- Graph reductions can turn signed weights negative, so a guarantee stated for
  non-negative Max-Cut does not carry over.

## Where next

- [API reference](API.md) — result fields, source parameters, and exceptions.
- [Laboratory catalog](LABORATORY.md) — the other warm-start assemblies.
- [Choosing an algorithm](CHOOSE_ALGORITHM.md) — whether a recursive reduction is
  the right tool for the graph at hand.
