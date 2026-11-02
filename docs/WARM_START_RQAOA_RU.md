# Warm-start RQAOA: ансамбль стартов

Версия `0.1.0a12`. Публичный маршрут:

```text
QA_max_cut_lib.laboratory.WarmStartRQAOACPU
type = laboratory.warm_start_rqaoa_cpu
```

## Что реализовано

`WarmStartRQAOACPU` использует существующий RQAOA-цикл, `GraphReducer`,
`MaxAbsCorrelationSelector`, `ExactCPUFinisher`, общий `ShotBudget` и
warm-start preparation/mixer этапа 11.

На каждом текущем графе:

1. `candidate_source` даёт конечный batch:
   - `gw_rounds_cpu`: один SDP и `L` округлений;
   - `provided_by_graph`: явные назначения для конкретного snapshot.
2. Кандидаты проверяются, канонизируются с global-flip dedup, сортируются по
   текущему `C`, выбираются до `max_starts`.
3. Каждый выбранный старт получает свой product state и независимый warm
   training task. Дискретного readout у child нет.
4. Физические `ZZ` корреляции детей смешиваются с равными весами.
5. Один общий selector и одна редукция применяются к текущему графу.
6. На следующем графе source batch создаётся заново; старые q/probabilities/
   factor не переиспользуются.

Класс поддерживает CPU full statevector и точный terminal finisher.

## Source

### `gw_rounds_cpu`

```json
{"type": "gw_rounds_cpu",
 "parameters": {"num_hyperplanes": 16, "solver": "SCS",
                "allow_inaccurate": false}}
```

Один SDP на граф, `L` округлений с `round_id/source_id`. Возвращаются все
фактические округления, включая повторы; число `L` не обещает `L` уникальных
разрезов. `candidate_roundings` — дополнительное поле результата GW, которое
запрашивается только ensemble source.

### `provided_by_graph`

```json
{"type": "provided_by_graph",
 "parameters": {"records": [
   {"graph": {"nodes": [10, 30, 80, 101],
              "edges": [[10,30,2],[10,80,3],[30,80,5]],
              "fingerprint": "..."},
    "assignments": [
      {"candidate_id": "A", "spins": {"10": 1, "30": 1, "80": -1, "101": 1}}
    ]}
 ]}
```

Запись выбирается только при совпадении порядка узлов и fingerprint текущего
графа. Отсутствие записи — ошибка; скрытый GW/старый seed не подставляется.

## Dedup и ranking

Для zero-field Max-Cut используется anchor = минимальная внешняя метка.
Если её спин `-1`, весь массив инвертируется. Ключ уникальности — tuple
спинов по каноническому порядку. `s` и `-s` — один старт; multiplicity
сохраняется, но не увеличивает вес. Сортировка: `-C_t`, canonical tuple,
source ID. Никакого нетранзитивного `isclose`-сортирования.

## Mixture

Для `K_t` детей:

```text
P_bar(b) = (1/K_t) sum_l |<b|psi_l>|^2
c_bar_uv = (1/K_t) sum_l <Z_u Z_v>_l
```

`rho_t = (1/K_t) sum |psi_l><psi_l|` не материализуется. Общий чистый
statevector смеси не создаётся: `aggregate_statevector=null`.

## Бюджет

Корневой `ShotBudget` один. Для `K_t`, `M_corr`, `p`, `M_train`:

```text
R_corr = K_t * M_corr
a = (4p+1)*M_train
```

Резерв корреляций проверяется до запуска quantum children. Training
итерации распределяются round-robin в canonical child order. Нельзя
уменьшать `K_t`, чтобы вписать training в остаток. Если хватает только
на корреляции, training iterations равны 0.

## Пример

```bash
python -m QA_max_cut_lib.laboratory.warm_start_rqaoa_cpu \
  --graph configs/graphs/warm_start_4.json \
  --config configs/solvers/warm_start_rqaoa_cpu_gw.json
```

Контроль A:

```text
узлы 10,30,80,101; рёбра 2,3,5
A=(+,+,-,+), B=(+,-,+,+)
U=2, multiplicities A=3, B=1
C(A)=8, C(B)=7
после mixture: c(10,30)=0, c(10,80)=0, c(30,80)=-1
выбрано (30,80), sigma=-1
b=8, C_final=8, E_Ising=-7.5
```

## Серия

`experiments/benchmark_warm_start_rqaoa.json`:

```text
2 graph instances × 2 trial seeds × 3 recipes = 12 trials
rqaoa_cpu, ws_rqaoa_k1, ws_rqaoa_k2
finish_at=2, p=1, max_steps=3, M_train=M_corr=32
```

Это не fixed-total-budget comparison: у K=2 больше полной работы. Exact
reference считается отдельным reference job.

## Ограничения

- Нет checkpoint/resume для `WarmStartRQAOACPU`: `supports_checkpoint` не
  заявлен; checkpoint config отклоняется. Это явное ограничение a12.
- CPU full statevector только; GPU/chi/analytic p=1 сюда не переносятся.
- Нет adaptive weights, переноса стартов между редукциями, SA pools и
  fractional ensemble members.
- `gw_rounds_cpu` требует CVXPY/SCS и является classical источником; его
  primal value не сертификат глобального optimum.
- Графовые редукции могут сделать знаковые веса отрицательными; гарантия
  GW для неотрицательного Max-Cut не переносится.
