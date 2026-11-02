# Проверка 0.1.0a11

Дата: 2 ноября 2026 года.

## База

- Python 3.12.14, Windows 10; NumPy 2.5.4.
- `QA_max_cut_lib` `0.1.0a11`.
- CPU-тесты: `161 passed, 1 skipped, 1 warning`.
- GPU-тесты: `10 passed` (старый GPU-профиль; warm-start GPU не реализован).
- Sphinx `-W`: см. отчёт сборочного прогона.
- Wheel/sdist `0.1.0a11` установлены в `.venv` и `.venv-wheelcheck`.

## Inventory/catalog

```text
archive sha256: b4c684a5d4da469adc25bd99a405e402892d1a2524465e4c97ab6229c71a4aed
Python members: 89
parsed:         89
parse errors:   0
directories:    Diplom 28, ENV 3, Generators 3, Laboratory 22,
                Other 2, RQAOA_exp 2, Solvers 18, Tools 11
```

- `Diplom/Mk_5.py` и `Laboratory/LS_RQAOA_fast_5.py`: identical AST
  (`ast.dump(..., include_attributes=False)`), different raw SHA-256;
  equivalence marked provisional.
- Все 89 file records имеют путь, SHA-256, роль, variant ID и review/runtime
  status.
- `list/show` не импортирует CuPy/CVXPY и не запускает старый архив.

## Математические контроли

A: `q=(0.25,0.75)`, `P(cuts)=(3/16,1/16,9/16,3/16)`, initial expected
C=1.25, source cut=2.

B: epsilon=1/2, p=1 и p=2, несколько signed graph:
`rounded_modified` с beta и `warm_aligned` с `-beta` дают то же полное
распределение, что обычный QAOACPU с теми же gamma/beta (max error 0).

C: epsilon=0, aligned/modified сохраняют basis state; probability mass 1,
no NaN.

D: gamma=0 aligned сохраняет product distribution; modified меняет;
epsilon=1/4, gamma=0, beta=pi/2 даёт complement seed с тем же C.

E: `q=(0.1,0.6,0.95)`, epsilon=0.2 → `(0.2,0.6,0.8)`; q>1, epsilon<0,
epsilon>0.5, missing/duplicate node labels отклоняются.

## Запуски

```text
provided_spins: 544 shots = 480 training + 64 readout
pause after 1 iteration: known=160, held=64, available=320
resume: status=completed, shots=544, angles equal continuous
```

`WarmStartQAOACPU` с checkpoint:
- source/q/epsilon/mixer frozen;
- `restore()` не запускает solver;
- `resume()` продолжает canonical run;
- classical source не решается повторно при resume;
- `solve()` на восстановленном объекте — новый run.

## H

```text
3 graph instances × 2 trial seeds × 7 configs = 42 runs
base + {epsilon 0.25,0.50} × {x, warm_aligned, rounded_modified}
quantum budget = 42 × 544 = 22 848 shots
```

Фактический `ExperimentRunner`:
`planned=42 valid=42 failed=0`, success(valid)=0.8333, wall 0.203 s
(на ноутбуке, 4-node graphs). Exact reference — отдельный reference job.
Это не доказательство превосходства warm-start.

## Classical source

Реально выполнены `gw_cpu` и `simulated_annealing_cpu` в classical-профиле
(`.venv-stage02`):
- GW: 8 hyperplanes, SDP call, `source_cost` содержит `sdp_calls`,
  `hyperplanes`, `solve_seconds`, `total_seconds`;
- SA: 4 reads, 64 sweeps, `source_cost` содержит reads/sweeps/time;
- quantum shots остаются отдельными (`224` для p=1/1 step).

## Ограничения

- GPU warm-start не реализован; full CPU statevector only.
- `provided_*` source cost is external/unaccounted.
- Нет warm-start RQAOA, LS/hierarchy, fields/environment, MI strategies.
- Нет автоматического multiple-start/aggregation.
- H не является доказательством алгоритмического преимущества.

## Замечание об архиве

В рабочем дереве нет самого RAR; инвентаризация построена по извлечённому
`reference/RQAOA_main` (89 файлов). Значение SHA-256 архива взято из задания;
все 89 member hashes записаны в inventory. Проверка RAR-контейнера целиком
не выполнялась, поскольку контейнер отсутствует.
