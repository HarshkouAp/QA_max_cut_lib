# Verification 0.1.0a13

Date: 2 November 2026.

## Base

- `QA_max_cut_lib 0.1.0a13`, Python 3.12.14, NumPy 2.5.4.
- CPU: `173 passed, 1 skipped, 1 warning`.
- Sphinx `-W`: success.
- Wheel/sdist verified by installation outside the `src` tree.
- GPU: the old matrix was not re-run; GPU fields are **not claimed** in this
  stage.

## A — fields and constants

Graph: nodes `(-7,10,30,80,101)`, edges
`(10,30)=2`, `(10,80)=3`, `(30,80)=-4`, `(30,101)=5`, `(80,101)=-2`,
`S=(10,30)`, `r_80=+1`, `r_101=-1`, `r_-7=-1`.

```text
J(10,30)=1
h=(1.5,-4.5)
c_aff=-3
C_TT=-2
A_S=[5,10,-2,-1]
C_full=[3,8,-4,-3]
```

The table was verified by an independent exhaustive enumeration of cuts on
the source graph.

## B — shifted threshold

For `b=7` the original scores are `[10,15,3,4]`; with a uniform active
distribution and threshold `A=12` the probability is `0.25`.
The check with `cut_offset=7` returned `0.25`; forgetting `C_TT` would have
given `0.5`.

## C — a single qubit

`10` is active, external `r_80=+1`, edge of weight 2. At
`gamma=-pi/4`, `beta=pi/4` the active spin equals `-1`, the full cut equals 2;
fixing the active spin for zero-field symmetry would have given an incorrect
cut of 0.

## D — full-graph limit

At `S=V` the fields are zero, `C_TT=0`; the probabilities matched a plain
`QAOACPU` at the same angles with an error of `0.0` on the verified graph.
A pinned embedding is implemented separately in the test mathematics.

## F — a real reduction

Before the reduction `S={10}`: `h=2.5`, `c_aff=-2.5`.
After removing 80 and retaining 30 at `sigma=-1`: `b=8`, the new edge
`(10,30)=-1`, reference `r_30=+1`; `h=-0.5`, `c_aff=0.5`, original scores
`[8,7]`. `retain_survivor` removes only the record of 80; the lift sets
`r_80=-r_30`.

## Standalone and LS evidence

- `examples/conditional_qaoa.py`: 384 shots, full cut 8, affected 10.
- `examples/environment_ls.py`: 1 reduction, 96 shots, `exact_finisher`,
  full cut verified on the source graph.
- `tests/test_conditional.py`: 7 tests, including the full graph limit,
  threshold, singleton and `retain_survivor`.

## Not verified

- checkpoint/resume of the new LS recipe: not implemented;
- GPU field backend, soft/mean-field exchange, correlation-weighted fields:
  `not_run`/deferred;
- large parent graphs, several local tasks and crash fault injection: not
  claimed as passed;
- the RAR container is absent from the working tree; provenance is taken from
  the extracted archive of stage 11.
