# Проверка 0.1.0a17

Дата: 2 ноября 2026 года.

## База

- `QA_max_cut_lib 0.1.0a17`, Python 3.12.14, NumPy 2.5.4.
- CPU: `203 passed, 1 skipped, 1 warning`.
- Sphinx `-W`: успешно.
- Wheel установлен в `.venv` и `.venv-wheelcheck`; пример/CLI вне src.

## Реальный incremental run

Граф: complete N=6, labels `{-7,10,30,80,101,205}`, веса
`(2i+1)/8` по sorted pairs. Recipe:
`greedy_abs_weight_partition` + `survivor_neighbor_cover_v1` +
`keep_unobserved(age=2)` + `max_abs_weighted_correlation(q=1)`.

- status `completed`, reductions `2`, решение восстановлено на исходном graph;
- `4` query-bearing local tasks: 2 в эпохе 0 и 2 в эпохе 1;
- `224` shots: `160` training + `64` correlation;
- selected pairs эпох: `(10,80)` и `(30,80)`;
- final cut `18.0`, exact N=6 optimum `18.0`;
- P=1 и P=2 на сохранённом плане дают одинаковые solution, selection
  history и final bank; `num_treads_applied=2` при P=2.

Analytic p=1 exact Mean Cut: `0` shots, `4` source calls, `2` reductions,
cut `17.5`, без CuPy/CVXPY.

## Builders и selectors

- Star cover center 80, k=3: три membership, все direct neighbors
  покрыты, filler `-7` в последнем блоке; `k=6` даёт один full task.
- Densest objective на N=4: membership не меняется при замене всех
  весов на противоположные; internal edge count `3`.
- Fixture B: max-abs → `(10,30)`, q=1 → `(10,80)`, q=2 → `(30,80)`,
  min-wc → `(80,101)`; знаки `-,+,+,-`.
- Weighted+all_pairs и q вне 1/2 отклоняются валидацией.

## Бюджет

Для deterministic N=6 recipe:
- при `max_total_shots=47` и плане с обязательным минимумом 32 solver
  завершает первую эпоху, затем останавливается `budget_exhausted`;
- при `max_total_shots=224` проходит две эпохи ровно за 224 shots.
Точный минимум H на трёх задачах проверен арифметикой
`_round_robin_quotas`: 48 обязательных, 40 за полную iteration.

## Не выполнено

- checkpoint/resume банка, pending plan, child states и exact restore
  одного observation;
- fault-injection и unknown-charge replay;
- 48-trial серия;
- cache/performance result cache;
- MK_3 batch, odds-fusion, round-max, energy prediction, Mk_8 и GPU.

## Артефакты

- wheel: `qa_max_cut_lib-0.1.0a17-py3-none-any.whl`, SHA-256
  `2939482686e0096780d44ab8656c5cc588a165b2089cf4c2f8109bf9a1a7fc2f`;
- sdist: `qa_max_cut_lib-0.1.0a17.tar.gz`, SHA-256
  `45f54d106c4887b6e3e2d41bcbbb244153b7e34f53b0ece811516dfc827d5554`.
