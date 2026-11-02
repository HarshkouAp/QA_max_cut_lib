# Проверка 0.1.0a16

Дата: 2 ноября 2026 года.

## База

- `QA_max_cut_lib 0.1.0a16`, Python 3.12.14, NumPy 2.5.4.
- CPU: `197 passed, 1 skipped, 1 warning`.
- Sphinx `-W`: успешно.
- Wheel установлен в `.venv` и `.venv-wheelcheck`; CLI/examples вне src.

## A/B/C/D

- Product plus: `I_Q=0`, `I_Z=0`, `ZZ=0`.
- Bell Φ/Ψ: `I_Q=2 bits`, `I_Z=1 bit`; знак ZZ различается, MI нет.
- CZ plus: `I_Q=2`, `I_Z=0`.
- GHZ: `S_i=1`, pair `I_Q=I_Z=1`.
- Bell(A,E)⊗|0>_B: QMI пары A,B = 0; правильная S_B=0 (старый s2=s1 bug
  дал бы ложное значение).
- Complex RDM: labels `80,-7,101,10`, `ρ_{-7}=[[0.5,-0.5i],[0.5i,0.5]]`.

## E/F/G

- Counts `[[3,1],[1,3]]`, M=8: `I_Z=0.18872187554086717` bits.
- Mask/statistics: values `(2,0,0)` дают mean `2/3`, min `0`, max `2`,
  population std `2√2/3`; реальные нули не отфильтрованы.
- Prefix p=3, θ=(0.1,0.2,0.3,0.4,0.5,0.6): layer 2 использует
  `(0.1,0.2,0.4,0.5)`; theta halves не смешиваются.

## Observer passivity

QAOA target: baseline `qaoa_cpu` и `laboratory.qaoa_mi_cpu` на одном
graph/seed дают одинаковые solution, cut, selected angles и shots.
Observer добавляет MI frames, но не второй readout и не новые samples.
Для p=2/two iterations/M_train=16/readout=64 фактический расход — 352
shots, 3 prefix snapshots, 15 all-pairs на snapshot.

RQAOA observer: full-state CPU source, frames до reducer commit, 2
reduction steps, `max_abs_correlation` не изменён. Correlation/reuse
provenance сохраняется отдельно.

## Not run

- реальный cross-process restore анализа I;
- fault-injection/unknown-charge observer phase;
- полная 12-trial серия и CPU profile;
- GPU/tomography/MI-guided selector/reconstruction;
- all production pair-scope/analysis-limit combinations.

## Артефакты

- wheel: `qa_max_cut_lib-0.1.0a16-py3-none-any.whl`, SHA-256
  `a7fafe37bcac76c35ddb528518c3ac84973c0f4e8286034eac3535c37f3e8607`;
- sdist: `qa_max_cut_lib-0.1.0a16.tar.gz`, SHA-256
  `0e8e6b56f024b9ee5705950e620086d0b5a857f159547d1c49e11e8e85656bbf`.
