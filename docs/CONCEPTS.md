# Concepts: cut, energy, and solver results

Use this page after the first example has run but `cut_value`, negative energy,
`probabilities`, and original integer labels are still unclear. After reading,
you can check any result against the graph and distinguish an expected cut from
the single cut selected by readout.

## Graph and spins

The input is an undirected graph `G=(V,E)` with real weights `w_ij`. Vertices may
have arbitrary integer labels such as `10`, `-7`, or `101`; the library does not
reindex them to `0..n-1`. A spin `s_i ∈ {-1,+1}` says which side of the cut
contains vertex `i`. An edge crosses the cut when `s_i ≠ s_j`.

The cut indicator is `(1-s_i s_j)/2`: it is `1` for opposite signs and `0` for
equal signs. Multiplying by the weight and summing gives the cut weight:

```text
C(s) = Σ_(i,j) w_ij (1 - s_i s_j) / 2.
```

The library defines energy as `E(s) = -C(s)`. The best cut therefore has the
lowest energy. For one edge of weight `2`:

| Spins | `C` | `E` |
|---|---:|---:|
| opposite | 2 | -2 |
| equal | 0 | 0 |

The constant energy term is retained in `IsingProblem.constant` so reductions
and checks do not lose the offset. If you compare only `cut_value`, the constant
is irrelevant; if you check energy after a reduction, it matters.

## From spins to bits

QAOA uses the computational basis. Bit `b_i=0` corresponds to spin `+1`, and bit
`b_i=1` to spin `-1`:

```text
s_i = 1 - 2 b_i.
```

The full statevector has length `2^n`, where `n = len(graph.nodes)`. The index
`k` encodes bits in `node_order`: the least-significant bit belongs to the first
vertex in that order. If `node_order = [10, 30, 80, 101]`, index `k=1` means
`b=(1,0,0,0)`, that is `{10:-1, 30:+1, 80:+1, 101:+1}`.

The same rule is used by `QAOACPU.probabilities`, `QAOACPU.node_order`, and any
manual reconstruction of the cut-value distribution. In GPU mode
`spin_flip_pairs`, the representative fixes the last bit of `node_order` to zero;
the physical probability of the pair is `q_r/2`, not `q_r`.

## What the solver returns

The public call:

```python
solution = solver.solve(graph)
```

returns `{original_label: +1 or -1}`. For example:

```python
{10: 1, 30: -1, 80: -1, 101: 1}
```

`solver.cut_value` is the weight of that particular cut. `solver.solution_energy`
equals `-cut_value` when the constant is retained. `solver.expected_cut` is
`E_ψ[C]` under the final distribution, not the weight of the selected cut. With
stochastic readout these values can differ.

Main fields:

| Field | Meaning |
|---|---|
| `solution` | selected partition, original labels |
| `cut_value` | `C(solution)` |
| `solution_energy` | `-cut_value` |
| `expected_cut` | mean `C` under the final state |
| `optimal_angles` | parameters from the last training run |
| `resources` | shots, calls, and budget used |
| `last_run` | independent snapshot of the latest run |
| `result` | complete JSON-compatible result |

A solver instance can be called on different graphs. Each `solve()` starts a new
run with a reset seed; the previous `last_run` is not mutated.

## Expectation, readout, and optimum

Three quantities are easy to confuse:

1. **Optimum** `C*` — maximum over all `2^n` partitions; only exact enumeration or
   a certified method knows it.
2. **Expectation** `E_ψ[C]` — average over the final state distribution; it can be
   computed from the statevector without measurements.
3. **Readout** — one partition selected by `most_probable`, `best_of_shots`, or
   another rule. This is what `solution` contains.

`best_of_shots` maximizes `C` among measured outcomes; it does not prove a global
optimum. `expected_cut` is independent of the random choice of one outcome but
depends on the state and angles.

## Check on one edge

For a two-vertex graph with weight `w`, the analytic depth-one result is:

```text
E[C] = (w/2) [1 - sin(γw) sin(4β)].
```

With `w=1`, `γ=-π/2`, and `β=π/8`, this gives `E[C]=1`, `⟨H_E⟩=-1`, and
`⟨Z_1Z_2⟩=-1`. It is a compact check of signs: a wrong phase-operator or mixer
sign changes the correlator immediately.

```python
from QA_max_cut_lib import AnalyticP1Evaluator, WeightedGraph

graph = WeightedGraph(edges=[(0, 1, 1.0)])
evaluator = AnalyticP1Evaluator(graph)
print(evaluator.correlations(-3.141592653589793 / 2, 3.141592653589793 / 8))
print(evaluator.energy_min_beta(1.0))
```

Exact values and assumptions are in [Mathematical conventions](MATHEMATICS.md);
the one-edge derivation is in the [scientific chapter](weighted_max_cut.rst).

## What counts as a check

A minimal check:

```python
assert set(solution) == set(graph.nodes)
assert graph.cut_value(solution) == solver.cut_value
assert solver.solution_energy == -solver.cut_value
assert solver.resources["shots_used"] <= solver.resources["budget"]["total"]
```

For an exact reference, add `ExactCPU` and compare `cut_value`. For QAOA, verify
that `expected_cut` can be reconstructed from `probabilities` and `node_order`.
For a reduction, check original-label restoration and constant-energy
preservation, not only the final cut.

## Where next

- [Quickstart](QUICKSTART.md) — run the example.
- [Choosing an algorithm](CHOOSE_ALGORITHM.md) — which method to use when.
- [API](API.md) — exact field types and exceptions.
