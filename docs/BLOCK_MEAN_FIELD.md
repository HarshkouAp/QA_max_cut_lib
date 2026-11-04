# Block mean-field QAOA on CPU

`BlockMeanFieldQAOACPU` (`type="laboratory.block_mean_field_qaoa_cpu"`) splits the
graph into blocks and runs a full statevector QAOA on one block at a time, passing
only first and second moments between blocks. Given a graph, a partition, and an
initial message, it returns an original-label assignment together with the local
moments that produced it. This page states the model the blocks share, the update
that keeps the shared moments consistent, and the checks that distinguish the model
mean cut from the cut of the returned partition.

The partition is fixed for the run: the method is a message-passing loop over a
given decomposition, not a partition search.

## The model held between blocks

For each block `S_a` the assembly stores only compact moments:

\[
m_i=\langle Z_i\rangle,\qquad q_{ij}=\langle Z_iZ_j\rangle\quad(i,j\in S_a).
\]

The mean cut of the induced product model follows from those moments. An edge
inside a block contributes through its pair moment, an edge between blocks only
through the product of means:

\[
\mathcal C(m,q)=
\frac12\sum_{\{i,j\}\in E,\ a(i)=a(j)}w_{ij}(1-q_{ij})
+\frac12\sum_{\{i,j\}\in E,\ a(i)\ne a(j)}w_{ij}(1-m_im_j).
\]

For the active block, every edge leaving the block becomes a local field:

\[
h_i=\frac12\sum_{j\notin S_a}w_{ij}m_j,\qquad
c_{\mathrm{aff}}=-\frac12\left(\sum_{\mathrm{internal}}w+
\sum_{\mathrm{cross}}w\right),
\]

with Ising coupling `J_{ik}=w_{ik}/2`. The local diagonal of the block problem is
computed from these coefficients,

\[
A_S(x)=-c_{\mathrm{aff}}-h\cdot x-J\cdot xx,
\]

not from a single induced subgraph: the external means contribute through `h`, and
the constant outside the block is tracked separately in `c_aff`. If the block were
solved from its induced subgraph alone, cross edges would either disappear or be
counted twice.

## One update

1. Build `h`, `J`, and `c_aff` from the current snapshot of messages (Gauss-Seidel)
   or from the frozen snapshot of the sweep (Jacobi).
2. The shared CPU QAOA/ADAM trains the full `2^{|S|}` statevector of the block.
3. Mix the angles:
   `theta_applied = (1-kappa)*theta_old + kappa*theta_candidate`.
4. At the **applied** angles, one sampling operation yields all `Z` values and all
   internal `ZZ` pairs of the block.
5. Mix `m` and `q` of the block with the same `lambda`. Using different factors for
   the two is not a physical pair distribution.
6. Jacobi publishes the whole sweep only after its barrier; Gauss-Seidel publishes
   each block as soon as it finishes.

The output is `sign_means(m)` with an explicit `zero_sign`, followed by an
independent full cut of that assignment on the original graph. The returned cut is
therefore a classical evaluation of a rounded mean vector, not a global quantum
readout.

## Running example

A single edge of weight `2` with blocks `[[0], [1]]` and initial means
`{0: 0.2, 1: -0.4}` is small enough to check by hand. Both endpoints sit in
different blocks, so the model uses the product term only:

```text
model mean cut = 0.5 * 2 * (1 - 0.2*(-0.4)) = 1.08
```

The partition is rounded from the same means and returns `{0: 1, 1: -1}`, whose cut
on the graph is `2.0`. The two numbers answer different questions: `1.08` is the
expectation of the product model, `2.0` is the weight of one rounded partition.

The tempting shortcut is to add the per-block affected expectations. Each block on
its own also evaluates to `1.08`, so the sum would be `2.16` — the edge would be
counted once per endpoint. This is why the model cut is computed once on the full
edge list rather than accumulated per block. The numbers above are asserted by
`tests/test_block_mean_field.py::test_model_cut_vs_rounding_and_double_counting`.

The same test file pins the damping convention: mixing `m=(1,1)` with `m=(-1,1)`
at `lambda=0.5` gives `m=(0,1)`, and the pair moment is mixed as well, giving
`q=0`. Keeping the candidate `q=-1` next to the damped means would instead produce
a negative entry in `pair_probabilities`, which the helper reports rather than
hides.

To run the shipped configuration end to end:

```bash
python -m QA_max_cut_lib.laboratory.block_mean_field_qaoa_cpu \
  --graph configs/graphs/block_mean_field_6.json \
  --config configs/solvers/block_mean_field_jacobi_cpu.json
```

With `seed=20261102` the run completes three Jacobi sweeps, executes nine local
updates, and spends 864 shots (720 training, 144 moment shots). The final model mean
cut is `-0.546`, while the rounded assignment scores `0`. This is an interface check,
not a performance claim: the exact optimum of that six-vertex graph is `7`, and the
short run returns `0`. A gap between the model value and the rounded cut is expected —
the model averages a product distribution, the return value is one discrete partition.

## Limits

- `max_sweeps` is finite and the result states the termination reason. Checkpoint
  and resume exist at the sweep boundary (`initialized`, `gauss_seidel_sweep_commit`,
  `jacobi_sweep_commit`, `paused`) with a compact bank codec; mid-sweep child
  checkpoints are not claimed.
- No overlapping blocks, no adaptive damping, no nonlinear or
  correlation-weighted fields, no GPU/`chi`/analytic path, and no final sampling of
  a global mixture.
- `local_surrogate` threshold probability is not the probability that
  `C_G >= A` on the original graph; the surrogate sees one block plus its fields.
- An empty graph returns `{}` without any quantum work.

## Where next

- [API reference](API.md) — partition, moment, and checkpoint fields with defaults.
- [Conditional QAOA](CONDITIONAL.md) — the case where outside spins are fixed
  instead of averaged.
- [Choosing an algorithm](CHOOSE_ALGORITHM.md) — when a message-passing loop is
  preferable to a single statevector.
