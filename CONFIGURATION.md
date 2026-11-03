# Configuration

Solver and graph configs are JSON-compatible dictionaries. Schema 1 is
the historical graph/solver family; schema 2 adds generators with
`limits` and resolved metadata. The solver registry is loaded through
`SolverFactory`; unknown types and fields are rejected before expensive
work.

Key conventions:

- `num_treads` is the CPU worker budget, never a CUDA launch dimension.
- graph labels are external integers; insertion order is state order.
- edge weights are signed; Max-Cut is `sum w(1-s_i s_j)/2`.
- source correlations are `c_ij=<Z_i Z_j>`; decision scores are separate.
- checkpoint and result are different artifacts.
