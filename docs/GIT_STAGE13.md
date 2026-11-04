# Git stage 13

Commit `e1958a2` contains Stage 13: the shared `ConditionalCutProblem`, a full
CPU statevector for the active block, `ConditionalQAOACPU`,
`EnvironmentLSRQAOACPU`, Laboratory routes, source cards, configurations,
examples, tests, documentation and notebook 14.

Limitations: checkpoint/resume of the new LS recipe is not implemented; GPU
fields, soft/mean-field exchange and correlation-weighted fields are not
claimed. The old LS/QAOA contracts are preserved; the shared `LSRQAOACPU`
gained only the neutral hooks `_fallback_solution` and `_on_reduction`.
