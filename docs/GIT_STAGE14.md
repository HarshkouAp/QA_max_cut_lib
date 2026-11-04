# Git stage 14

Commit `368b6bb` contains Stage 14: `BlockMeanFieldQAOACPU`, a compact `m/q`
bank, two Gauss–Seidel/Jacobi schedules, angle/moment damping, local surrogate
criteria, full sign-rounded output, catalog cards, configurations, examples,
tests, documentation and notebook 15.

Checkpointing is implemented at full-sweep boundaries; a mid-sweep child codec,
fault injection and replay receipts remain limitations of this version.
