# Conditional QAOA and LS-RQAOA with fixed outside spins

`ConditionalQAOACPU` (`type="laboratory.conditional_qaoa_cpu"`) solves one explicit
local problem: the spins outside an active block `S` are **fixed** to a reference
`r_T`, and the block itself is trained with a full CPU statevector. Given a graph, a
block, and a reference assignment, it returns the block partition plus the fields
that encode the fixed neighbourhood. This page defines that local problem, shows
the arithmetic on a four-vertex graph, and separates conditioning from the two
neighbouring ideas it is often confused with.

This is conditioning, not marginalization of a larger QAOA, and not the mean-field
exchange of [block mean-field QAOA](BLOCK_MEAN_FIELD.md): outside spins are
numbers here, not distributions.

## The local problem

For the current graph `G_t`, an active block `S`, and a fixed reference `r_T` on the
remaining vertices `T`:

\[
A_S(x;r_T)=C_{SS}(x)+C_{ST}(x,r_T),
\]

\[
E_{\mathrm{aff}}(x;r_T)=-A_S
=c_{\mathrm{aff}}+\sum_{i\in S}h_i x_i+
\sum_{\{i,k\}\subset S}J_{ik}x_ix_k,
\quad J_{ik}=\frac{w_{ik}}2,\quad
h_i=\frac12\sum_{j\in T}w_{ij}r_j.
\]

`A_S` is the cut of every edge that touches `S`; `E_aff` is its negative, so
minimizing `E_aff` maximizes the affected cut. The constant `c_aff`, the cut
`C_TT` inside `T`, and the accumulated offset `b_t` are stored separately. For the
original graph the active row is reconstructed as `b_t + C_TT + A_S`; dropping any
of the three would silently change the reported cut.

The field `h` is not encoded as a self-loop and is not an attribute of
`WeightedGraph`. Local fields are not passed to the global reducer either: they
arise from the chosen environment, and the global graph stays a zero-field Max-Cut
graph. A field that leaked into the working graph would change the meaning of later
reductions, so the separation is deliberate.

## Running example

Take the four-vertex graph with edges `(10,30,2)`, `(10,80,3)`, `(30,80,5)` and an
isolated vertex `101`. Fix the reference `r = {10: +1, 30: +1, 80: +1, 101: +1}`
and let `S = [10]`. Every edge incident to `10` leaves the block, so

```text
h_10    = (2*1 + 3*1)/2 = 2.5
c_aff   = -(2 + 3)/2     = -2.5
A_S(+1) = -c_aff - h*x  = 2.5 - 2.5  = 0
A_S(-1) = -c_aff - h*x  = 2.5 + 2.5  = 5
```

The two values match the graph: with `30` and `80` at `+1`, placing `10` at `+1`
cuts nothing, and placing it at `-1` cuts both edges for `5`. The field row is
asserted by
`tests/test_conditional.py::test_retain_survivor_rebuilds_fields_and_original_offset`
as `before.fields == {10: 2.5}`.

Now remove `80` and keep `30` with `sigma=-1`. The reducer rewrites the edge to
`w'(10,30) = 2 + (-1)*3 = -1`, and the same block is rebuilt against the shorter
reference:

```text
h_10     = (-1*1)/2 = -0.5
constant = -(-1)/2  =  0.5
```

The sign of the field follows the sign of the surviving edge, not the sign of the
original cross edge. `tests/test_conditional.py` checks both rows, together with
`original_cuts == [8.0, 7.0]` — the original-graph cut for the two assignments of
the active node — and the restored assignment `{10: 1, 30: 1, 80: -1, 101: 1}`.

## Public routes

```python
from QA_max_cut_lib.laboratory import ConditionalQAOACPU, EnvironmentLSRQAOACPU
```

- `laboratory.conditional_qaoa_cpu` — one block, completed by the fixed outside
  spins. There is no reduction and no recursion.
- `laboratory.environment_ls_rqaoa_cpu` — the LS controller that freezes a
  reference for one step and then reduces. Its environment section carries
  `update_policy` (`retain_survivor`) and `epoch_policy`
  (`frozen_per_reduction`).

In the LS route, `update="refresh_all"` means local problems are rebuilt after a
reduction. Old correlations are not relabelled as a cache hit: a rebuilt block is a
new problem even when its node set happens to match the previous step.

## Memory and cost

The active block is simulated as a full statevector of size `2^{|S|}`, so the cost
grows exponentially in the block size and not in the graph size. The default
`max_active_vertices` is `18` and the hard cap is `24`; a larger block is rejected
before any state is allocated. The parent graph may be much larger than a block —
each quantum task is bounded by `|S|` only.

## Limits

- No iterative `<Z>` exchange, no soft means, no correlation-weighted fields, no
  nonlinear F2 rules, no GPU/`chi`/analytic `p=1` path, and no warm start. Those
  remain separate assemblies.
- No checkpoint/resume is offered by either route: `supports_checkpoint` is not
  declared for `laboratory.conditional_qaoa_cpu` or
  `laboratory.environment_ls_rqaoa_cpu`, and a configuration with
  `checkpoint.enabled` is rejected. That is a visible refusal rather than a quiet
  restart from an earlier state.
- The reference must cover every label of the current graph, active labels included;
  it is a full assignment, not just the boundary. A reference that misses a label or
  adds an unknown one raises `ConditionalProblemError` with `missing` and `extra`
  lists, and the block is not built.

## Where next

- [API reference](API.md) — `active_vertices`, `environment`, `boundary`, and the
  result fields of both routes.
- [Block mean-field QAOA](BLOCK_MEAN_FIELD.md) — the averaging counterpart.
- [Choosing an algorithm](CHOOSE_ALGORITHM.md) — when fixed environments are
  cheaper than a global statevector.
