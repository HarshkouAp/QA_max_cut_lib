# Build verification 0.1.0a3

**Version:** `QA_max_cut_lib 0.1.0a3`, 2 November 2026.  
**Purpose:** basic CPU RQAOA on the exact reduction core of stage 02.

## Verified state

- `RQAOACPU`, `QAOAStatevectorCorrelationSource`,
  `MaxAbsCorrelationSelector`, `ExactCPUFinisher` are implemented.
- Exact and shot modes work, as do a shared `ShotBudget`, `edges`/`all_pairs`,
  `restart`/`previous_step`, an exact remainder and
  `all_plus_reduced_fallback`.
- `reduction_history`, `parameter_trajectory`, `local_optimizer_history`,
  `correlation_tables` and `final_reduction_state` are enabled by separate
  flags.
- Plots of weights, energies, all `binomial(2p,2)` trajectories and a
  standalone 3D HTML were added.
- Full check set on the local base: **80 passed, 1 warning**.
- The `0.1.0a3` wheel was built and installed into a clean environment;
  notebook 04 executed; the `rqaoa_cpu` CLI performed a real run.

## Known limitations

- The source uses the full state vector and is bounded by its size.
- Analytic `p=1`, GPU, LS-RQAOA, QAOA-in-QAOA and Laboratory are not yet
  ported.
- `num_treads=1`; checkpoint/resume is not implemented.
- RQAOA does not guarantee a global optimum.

Final documentation and archive export for stage 03 were not completed before
work moved to stage 04; the local code and tests of this base were preserved
and used in `0.1.0a4`.
