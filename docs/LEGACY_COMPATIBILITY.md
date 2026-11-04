# Legacy compatibility adapters (R19)

R19 closes the source archive by giving every materially different historical
branch a terminal disposition. This page explains the 18 branches that used to be
deferred and how to exercise them. The exact source-to-component mapping is in
`../SOURCE_CLOSURE.json`; the implementation is
`QA_max_cut_lib.laboratory.legacy`.

## What the adapters preserve

The historical files used private NumPy arrays, index maps, and separate
optimizer loops. Copying those loops again would create a second, incompatible
solver stack. The adapters instead preserve each branch as a validated recipe:

- `MK6V1`/`MK6V2`/`MK7V1`/`MK7V2` use the batch LS controller with parity or
  active-endpoint reconciliation and different block sizes;
- `LSRQAOAExp` and `LSRQAOAFast1`–`Fast4` use the LS controller with distinct
  builders and round schedules;
- `QAOAAdam`, `QAOAF2`, and `QAOAFM` use the QAOA controller with recorded
  optimizer/objective recipes;
- `CorrectedMeanMap`, `CorrelationExtractor`, `EdgeExtender`,
  `GraphDecomposerLight`, `GraphScanner`, and `partition_smoke_test` provide the
  small diagnostic and decomposition operations that do not belong in a solver.

The adapters delegate to the shared `LSRQAOACPU`, `BatchLSRQAOACPU`, and
`QAOACPU` implementations. They do not import or execute the archived modules at
runtime.

## Run one adapter

```python
from QA_max_cut_lib import WeightedGraph
from QA_max_cut_lib.laboratory.legacy import MK6V1

graph = WeightedGraph(
    edges=[(10, 20, 1.0), (10, 30, 2.0), (20, 30, 3.0)],
    nodes=[10, 20, 30, 101],
)
solver = MK6V1()
solution = solver.solve(graph)
print(solution)                 # original labels -> ±1
print(solver.result["status"])  # completed or stopped
```

The test matrix is `tests/test_legacy.py`; it exercises all nine solver adapters,
all three QAOA adapters, and the six diagnostic components. The catalog summary
now has no deferred variant IDs.

## What is not claimed

This is a compatibility closure, not a bitwise replay of the archived scripts.
The historical public names are retained in `SOURCE_ALIASES`; the implementation
uses the current validated controllers and their documented budgets, seeds, and
readout rules. A future exact replay would require pinning the old dependency
versions and recording a separate acceptance run.
