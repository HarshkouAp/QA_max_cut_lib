# Choosing an algorithm

Use this page when the graph is already loaded and the remaining question is
which solver to call. The choice is driven by the answer you need: a certified
optimum, a fast feasible cut, a quantum-state experiment, or a reproducible
comparison. Start with the exact reference on small graphs; use a heuristic only
when its cost and guarantees match the task.

## Decision path

1. **Do you need a certified optimum?** If `n` is small enough for `ExactCPU`,
   use it. It enumerates all `2^n` partitions and is the reference for every other
   method.
2. **Do you need a statevector, probabilities, or QAOA angles?** Use `QAOACPU`
   for a small graph, or `QAOAGPU` when CuPy/CUDA is available. This is the direct
   route to the quantum circuit model.
3. **Do you need a recursive reduction with measured correlations?** Use
   `RQAOACPU` (or `RQAOAGPU`). It is a heuristic: a small remaining graph is solved
   exactly, but that does not certify the original graph.
4. **Do you need local subproblems and a shared shot budget?** Use `LSRQAOACPU`.
   It is an explicit induced-subgraph heuristic, not an exact decomposition.
5. **Do you need a classical baseline?** Use `GoemansWilliamsonCPU` for an SDP
   relaxation with hyperplane rounding, or `SimulatedAnnealingCPU` for a
   stochastic local baseline. Install the `classical` extra.
6. **Do you need blocks with one free orientation?** Use `QAOAInQAOACPU` or its
   GPU counterpart; the block cover must be specified explicitly.

## Comparison

| Method | What it returns | Cost | Guarantee / caveat |
|---|---|---|---|
| `ExactCPU` | global optimum | `O(2^n)` time and memory | exact; only small `n` |
| `QAOACPU` | statevector + selected cut | statevector `2^n` | variational; depth `p` limits quality |
| `RQAOACPU` | recursive cut | repeated local QAOA | heuristic correlations; exact only on remainder |
| `LSRQAOACPU` | cut from local tasks | local statevectors + shots | induced subgraphs omit boundary fields |
| `QAOAGPU` | GPU statevector + cut | VRAM `2^n` (or paired) | needs CUDA; memory preflight is not a guarantee |
| `GoemansWilliamsonCPU` | rounded SDP cut | SDP solve | approximation, not a certificate here |
| `SimulatedAnnealingCPU` | sampled cut | reads × sweeps | stochastic; use seeds and repeats |
| `QAOAInQAOACPU` | hierarchical cut | local + meta solve | block cover and one-orientation model are explicit |

## A small comparison

The public example is deliberately small enough to compare exact enumeration and
depth-one QAOA on the same graph:

```python
from QA_max_cut_lib import ExactCPU, QAOACPU, WeightedGraph

graph = WeightedGraph(
    edges=[(10, 30, 1.5), (10, 80, 2.0), (30, 80, -0.5)],
    nodes=[10, 30, 80, 101],
)

exact = ExactCPU()
exact.solve(graph)

qaoa = QAOACPU({
    "seed": 42,
    "parameters": {"layers": 1},
    "objective": {"type": "mean_cut"},
    "estimation": {"mode": "exact"},
    "initialization": {"type": "fixed_angles", "parameters": {"values": [-0.8, 0.4]}},
    "optimizer": {"parameters": {"max_steps": 24}},
    "readout": {"mode": "best_of_shots", "shots": 128},
    "limits": {"shots_total": 128},
})
solution = qaoa.solve(graph)

print(exact.cut_value)       # 3.5
print(qaoa.expected_cut)     # 3.044365...
print(qaoa.cut_value)        # selected cut
```

The exact value is a reference; the QAOA value is an expectation and the
returned cut is one readout. Do not compare `expected_cut` with `exact.cut_value`
as if they were the same statistic.

## Cost and budget

- **Statevector methods** allocate `2^n` amplitudes. At `n=30`, a complex128
  vector alone is about 16 GiB; the CPU implementation is not a route to
  thousand-vertex graphs.
- **Shot-based training** spends `(4p+1)M_train` training shots per finite-
  difference step for `p` layers, plus readout shots. The exact accounting is in
  [Mathematical conventions](MATHEMATICS.md).
- **RQAOA/LS** repeat local solves; the total budget is shared and is checked by
  the coordinator, not by independent copies.
- **Classical GW/SA** use no quantum shots. `num_reads` and sweeps are classical
  work, not quantum measurements.

## What not to infer

A feasible cut is not an optimality certificate. A small exact remainder in
RQAOA does not prove the original optimum. A GPU memory preflight is not a
promise that the solver fits. A single seed is not an uncertainty estimate; use
independent seeds and report the spread.

For exact fields, defaults, and exceptions, see [API](API.md). For formulas and
sign conventions, see [Mathematical conventions](MATHEMATICS.md).
