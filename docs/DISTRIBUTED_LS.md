# Distributed LS-RQAOA on CPU

`DistributedLSRQAOACPU` (`type="laboratory.distributed_ls_rqaoa_cpu"`) runs one
outer reduction step in which several independent mean-field episodes start from the
same message vector, and only their committed means are carried forward. Given a
graph, a partition policy, and an initial message, it returns the original-label
assignment together with the episode history and the fresh correlation table that
justified each reduction. This page states the order of one outer step, the two
transport policies, and the difference between a fresh correlation and a damped
model value.

The method is a local-search heuristic. A reduction is chosen from measured local
correlations; it is not an exact decomposition of the original objective.

## One outer step

1. Freeze the parent graph `G_t`, the accumulated offset `b_t`, and the message
   vector `mu_t`.
2. Run several independent mean-field episodes, all starting from the same `mu_t`.
3. Extract fresh final correlations at the frozen final means and final angles.
4. Average over the records that actually observed a pair.
5. Commit one reduction of the full `G_t`.
6. Apply the transport policy, reset or repartition, and advance.

Step 3 is separate from step 2 on purpose. The episodes produce a damped model pair
moment `q`; that number is part of a surrogate and is not a measurement of the
committed state. The reduction is selected from the fresh `c`, so a stale model
value cannot be mistaken for evidence.

## Episodes and partitions

The partition builder creates complete, disjoint partitions of the current graph.
Each ordinary round is one episode. If edge coverage comes out empty, anchor repair
adds a **separate complete** partition instead of inserting an overlapping block
into an existing model — an overlapping block would double-count edges and break the
moment bookkeeping.

Two transport policies carry means to the next graph:

- `carry_survivor_means` copies the average episode means onto the labels that
  remain after the reduction.
- `cold_restart` re-applies the original initializer.

Both reset `q = m_i m_j` for the new product state and restart the angles. This is
heuristic reinitialization, not an exact conditional projection: the surviving
means are kept, but the pair structure of the next state is rebuilt from them.

## Fresh extraction

After the episode engine finishes, the new fields are built from the final committed
means. At the final angles the assembly then performs a separate exact or shot-based
correlation operation on the local blocks. The damped model `q` and the last update
receipt are not reused as if they were that result.

## Running example

The contract fixture is a two-vertex graph with one edge of weight `2`
(`tests/test_distributed_ls.py::test_fixture_c_fresh_correlation_beats_bank_q`).
It starts from messages `(1, 1)` with zero mean-field sweeps, so the model moment is
`q = m_10*m_30 = +1`; the fresh extraction at the final angles returns

```text
final_correlation_tables[0]["values"] = [[10, 30, -1.0]]
reduction_history[0]["correlation"]    = -1.0
reduction_history[0]["sign"]           = -1
```

The selector keeps the pair with the largest absolute correlation — here the only
pair — with `sign=-1`; the finisher returns `{10: 1, 30: -1}` and
`cut_value = 2.0`. The point of the fixture is the first line: the fresh measurement
wins over the bank value. Using the model moment instead would flip the selected
sign to `+1` and produce a different reduction.

To run the shipped configuration on the six-vertex graph:

```bash
python -m QA_max_cut_lib.laboratory.distributed_ls_rqaoa_cpu \
  --graph configs/graphs/block_mean_field_6.json \
  --config configs/solvers/distributed_ls_jacobi_cpu.json
```

With `seed=20261102` the run completes two reductions, two episodes, and four logical
inner updates, spending 256 of 512 shots. It reports `Cut: -2`. That graph has
negative weights, so a negative cut is possible; the exact optimum is `7`, and this
short run is an interface check rather than a quality claim. Read the cut together
with `reductions_completed` and the fresh tables, not as a bound on the optimum.

## Limits

- Episodes execute sequentially. Episode-level parallelism is not implemented.
- Checkpoint and resume exist at the outer boundaries `initialized` and
  `reduction_committed`, with a codec over the reduction state, parent means,
  budget, completed episodes, and history. Mid-episode nested checkpoint and
  fault-injection replay are not claimed.
- No overlapping partitions, no nonlinear or adaptive fields, no GPU/`chi`/analytic
  path, and no global readout.
- Fresh extraction is a statistical local `ZZ` estimate at fixed angles. It is not a
  matrix element of one shared pure state, and its error does not shrink when the
  budget is spent on training instead of correlation shots.

## Where next

- [API reference](API.md) — transport, aggregation, and checkpoint fields.
- [Block mean-field QAOA](BLOCK_MEAN_FIELD.md) — the episode engine this
  assembly distributes.
- [Incremental LS-RQAOA](INCREMENTAL_LS.md) — reuse of observations across
  epochs, which this page deliberately does not do.
