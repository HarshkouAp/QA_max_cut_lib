# Git stage 16

Version: `0.1.0a16`.

Added:

- `tools.mutual_information`: partial trace, entropy, QMI, Z-MI over P,
  counts and moments, pair scope, mask/statistics, hop distance;
- `tools.mi_plots`: offline MI heatmap, prefix profile and hop profile;
- `laboratory.qaoa_mi_cpu` / `QAOAWithMIObserverCPU`;
- `laboratory.rqaoa_mi_cpu` / `RQAOAWithMIObserverCPU`;
- `configs/solvers/qaoa_mi_cpu.json`, `rqaoa_mi_cpu.json`;
- `tests/test_mutual_information.py`;
- notebook `17_mutual_information_diagnostics.ipynb`, examples and docs.

Boundaries: MI is a passive simulator diagnostic; there is no tomography,
MI-guided selection, quantum-state lift or GPU MI. RQAOA observer checkpoint
is not implemented; the QAOA observer uses the baseline checkpoint and
recomputes MI from the stored final angles after resume. A partial MI cursor is
not stored.
