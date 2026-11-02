# Git stage 16

Версия: `0.1.0a16`.

Добавлены:

- `tools.mutual_information`: partial trace, entropy, QMI, Z-MI по P,
  counts и моментам, pair scope, mask/statistics, hop distance;
- `tools.mi_plots`: offline MI heatmap, prefix profile и hop profile;
- `laboratory.qaoa_mi_cpu` / `QAOAWithMIObserverCPU`;
- `laboratory.rqaoa_mi_cpu` / `RQAOAWithMIObserverCPU`;
- `configs/solvers/qaoa_mi_cpu.json`, `rqaoa_mi_cpu.json`;
- `tests/test_mutual_information.py`;
- notebook `17_mutual_information_diagnostics.ipynb`, examples и docs.

Границы: MI — passive simulator diagnostic; нет tomography, MI-guided
selection, quantum-state lift и GPU MI. RQAOA observer checkpoint не
реализован; QAOA observer использует baseline checkpoint и пересчитывает
MI из сохранённых final angles после resume. Частичный MI cursor не
сохраняется.
