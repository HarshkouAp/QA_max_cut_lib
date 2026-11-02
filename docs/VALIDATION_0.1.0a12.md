# Проверка 0.1.0a12

Дата: 2 ноября 2026 года.

## База

- `QA_max_cut_lib 0.1.0a12`, Python 3.12.14, NumPy 2.5.4.
- CPU: `166 passed, 1 skipped, 1 warning`.
- GPU: `10 passed` (старый GPU-профиль; warm-start GPU не заявлен).
- Sphinx `-W`: успешно после обновления документации.
- Wheel/sdist `0.1.0a12` установлены в `.venv` и `.venv-wheelcheck`.

## Каталог

```text
archive SHA (задание): b4c684a5d4da469adc25bd99a405e402892d1a2524465e4c97ab6229c71a4aed
Python members: 89
variant cards: 7
new card: warm_start_ensemble_rqaoa_cpu_v1
relation: component_recomposition
```

RAR-контейнер в рабочем дереве отсутствует; инвентаризация построена по
извлечённому `reference/RQAOA_main`. Все 89 member hashes записаны.

## A — два старта и отрицательное ребро

```text
узлы 10,30,80,101; рёбра 2,3,5
A=(+,+,−,+), B=(+,−,+,+)
raw batch: A, −A, B, A2
U=2; multiplicities A=3, B=1
C(A)=8, C(B)=7
selected c: (10,30)=0, (10,80)=0, (30,80)=-1
selected pair (30,80), sigma=-1
C_final=8, b=8, E_Ising=-7.5
reductions=1, finisher=exact_finisher
```

Проверка выполнена на реальном `WarmStartRQAOACPU` с provided batch, без
CVXPY.

## B/C/D/E

- ranking на текущем графе использует пересчитанный `C_t`, а не старый вес;
- global-flip dedup: A/−A/A2 дают один ключ, multiplicity не меняет вес;
- mixture A: `P=(1/4,1/4,1/4,1/4)`, `c=0`; coherent sum даёт
  `P=(1/2,0,0,1/2)`, `c=1` — ошибочная coherent aggregation ловится;
- nonlinear comparison: mean cut и upper-tail CVaR компонентов не
  подменяются критерием смеси;
- endpoint K=1 проверен как частный случай.

## GW source

Выполнен реальный `gw_rounds_cpu` на 4-вершинном графе:

```text
sdp_calls=1
rounds_requested=4
rounds_completed=4
source_cost.hyperplanes=4
sdp_status=optimal
```

Это один SDP, не K независимых вызовов. В серии использовано 8 округлений.

## G/H — бюджет и серия

```text
K=3, B=368, p=1, M_train=32, M_corr=16:
iterations=[1,1,0], reserve=48, quotas=[176,176,16]
K=2: iterations=[1,1], reserve=32
B=47: quantum child не запускается
B=48: iterations=[0,0,0], только correlations
```

Малая серия `experiments/benchmark_warm_start_rqaoa.json`:

```text
2 graphs × 2 seeds × 3 recipes = 12 trials
rqaoa_cpu, ws_rqaoa_k1, ws_rqaoa_k2
finish_at=2, p=1, max_steps=3, M_train=M_corr=32
planned=12 valid=12 failed=0
```

Фактический `wall=0.384 s` в текущей среде; `rqaoa_cpu` и K=1/K=2
расходуют бюджет по-разному, это не fixed-total-budget comparison.

## Интеграция

- `WarmStartRQAOACPU` использует существующие `GraphReducer`,
  `MaxAbsCorrelationSelector`, `ExactCPUFinisher`, `ShotBudget` и QAOA
  trainer; второго ADAM нет.
- `provided_by_graph` работает без CVXPY/CUDA.
- GW source требует classical-профиль.
- `aggregate_statevector=null`; физическая смесь не выдаётся за чистый
  statevector.
- `source_cost` classical preprocessing отделён от quantum shots.

## Ограничения

- Checkpoint/resume для `WarmStartRQAOACPU` **не реализован** в a12;
  `supports_checkpoint` не заявлен, config с checkpoint отклоняется.
- Нет adaptive weights, переноса стартов между редукциями, SA pools,
  fractional ensemble members, GPU warm-start.
- Нет `provided_by_graph` после произвольной редукции без явной записи
  нового snapshot; отсутствие записи является ошибкой.
- SDP primal value не сертификат optimum.
