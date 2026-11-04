# Git Stage 08

Implementation commit: `d31cf13` — `Stage 08: add classical baselines and exact block refinement (0.1.0a8)`.

## Contents

- `solvers/goemans_williamson_cpu.py` — CVXPY/SCS SDP, controlled spectral
  repair and hyperplane rounding;
- `solvers/simulated_annealing_cpu.py` — dimod spin BQM and the classical
  `dwave.samplers.SimulatedAnnealingSampler`;
- `solvers/exact_block_refiner.py` — conditional fields, sequential exact
  block enumeration and explicit state limits;
- `solvers/refined.py` — wrapper base solver → refiner with independent
  `base_run` and `refinement_run`;
- `tools/classical_plots.py` — `analyze_solution` and GW/SA/refinement plots;
- configurations, tests, notebook `09`, the report and the `classical` profile.

## Checks

```text
CPU: 134 passed, 1 skipped, 1 warning
GPU: 10 passed
Sphinx -W: success
wheel/sdist: 0.1.0a8
CLI from the installed wheel: GW, SA, refined
```

`dist/` is excluded from Git and is added separately to the delivered zip.
