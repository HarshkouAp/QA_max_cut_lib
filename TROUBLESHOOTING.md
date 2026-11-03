# Troubleshooting

- `Solver ... is not implemented`: use an installed registry type and
  print `SolverFactory.available()`.
- CPU install fails importing CuPy: install only the CPU extra; CUDA is
  optional.
- `num_treads` rejected: the selected solver is not parallel.
- Checkpoint restore fails: do not edit the graph/config prefix; create
  an explicit fork instead.
- A plan has `completion_status=coverage_shortfall`: inspect
  `analyze_plan().uncovered_edges`; do not treat it as full coverage.
