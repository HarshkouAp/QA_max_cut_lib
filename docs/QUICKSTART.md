# Quickstart: your first cut and how to check it

Use this page if you have just opened QA_max_cut_lib and want one reproducible
result: install the CPU profile, solve a small weighted graph by exact
enumeration and QAOA, compare the cut, and understand the result fields. After
reading, you can run the example without a notebook and check its numbers by hand.

## Prerequisites

- Python 3.10 or newer;
- `QA_max_cut_lib` version `0.1.0rc2`;
- about 1 MB of memory for the full statevector of this four-vertex example.

The CPU profile does not install CuPy and does not require CUDA:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install ".[cpu]"
```

Run the installation command in the repository root when working from source.
For a published wheel, use
`python -m pip install qa_max_cut_lib-0.1.0rc2-py3-none-any.whl`.

## Run the example

```bash
python examples/public_quickstart.py
```

The script solves the graph with vertices `10, 30, 80, 101` and edges
`(10,30,1.5)`, `(10,80,2.0)`, `(30,80,-0.5)`. Vertex `101` is isolated.
The fixed `seed=42` makes training and readout reproducible.

Expected output:

```text
QA_max_cut_lib 0.1.0rc2
Exact optimum                    3.500000
Expected QAOA cut               3.044365
Best cut in 128 simulated shots 3.500000
Final-state optimal mass        0.805755
Readout spins                   {10: 1, 30: -1, 80: -1, 101: 1}
```

The four numbers answer different questions:

| Field | Meaning |
|---|---|
| `Exact optimum` | maximum `C(s)` over all `2^4` partitions |
| `Expected QAOA cut` | `E_ψ[C(s)]` in the final QAOA state |
| `Best cut in 128 shots` | maximum `C` among 128 measured outcomes |
| `Final-state optimal mass` | total probability of outcomes with `C=3.5` |
| `Readout spins` | one selected cut; keys are the original vertex labels |

`Expected QAOA cut` is below the optimum: depth `p=1` is not required to reach
it. `Best cut` equals the optimum only because optimal outcomes have a large
total probability and appeared in the sample. That is not a proof of global
optimality of QAOA.

## Check it by hand

For one edge of weight `w`, opposite spins give `C=w`; equal spins give `C=0`.
For positive `w=2`, opposite spins give `C=2` and `E=-2`. In the example, the
optimum `3.5` is reached when `10` is on one side and `30` and `80` are on the
other; isolated `101` can have either sign. Four configurations are optimal:
the global spin flip and the two values of `101` produce the same cut.

Check it in Python:

```python
from QA_max_cut_lib import ExactCPU, WeightedGraph

graph = WeightedGraph(
    edges=[(10, 30, 1.5), (10, 80, 2.0), (30, 80, -0.5)],
    nodes=[10, 30, 80, 101],
)
exact = ExactCPU()
exact.solve(graph)
print(exact.cut_value)       # 3.5
print(exact.optimal_count)   # 4
print(graph.cut_value({10: 1, 30: -1, 80: -1, 101: 1}))  # 3.5
```

## If your result differs

1. Confirm the version: `python -c "import QA_max_cut_lib;
   print(QA_max_cut_lib.__version__)"` must print `0.1.0rc2`.
2. Do not change `seed`, initialization angles, or `shots` while comparing with
   the table above.
3. Confirm `readout.mode="best_of_shots"` and `limits.shots_total >= 128`;
   otherwise the actual number of measurements differs.
4. For another graph, vertex order and weight signs can change the cut. Check
   `graph.cut_value(solution)`, not only the spin dictionary.

## Where next

- [Concepts and bit order](CONCEPTS.md) — why `probabilities` is indexed this way.
- [Scientific Max-Cut chapter](weighted_max_cut.rst) — derivation of the
  Hamiltonian and the one-edge formula.
- [Choosing an algorithm](CHOOSE_ALGORITHM.md) — when QAOA, RQAOA, or a classical
  method is appropriate.
- [API](API.md) — complete configuration and result fields.
