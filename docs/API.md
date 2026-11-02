# API версии 0.1.0a11

Все конфигурации ниже — обычные JSON-совместимые Python `dict`; файл читается через `tools.load_json(path)`. `solve()` возвращает словарь решения. `algorithm.result` возвращает JSON-совместимый словарь, а не текстовую строку. Для текстового JSON применяется стандартный `json.dumps`.

## Публичные импорты и жизненный цикл

```python
from QA_max_cut_lib import (
    WeightedGraph, GraphFactory, SolverFactory, RunRecord, ExactCPU, QAOACPU,
    RQAOACPU, AnalyticP1Evaluator, AnalyticP1CorrelationSource,
    LSRQAOACPU, QAOAGPU, QAOAGPUCorrelationSource, RQAOAGPU, LSRQAOAGPU,
    QAOAInQAOACPU, QAOAInQAOAGPU, GoemansWilliamsonCPU, SimulatedAnnealingCPU,
    ExactBlockRefiner, RefinedSolver, BenchmarkSpec, ExperimentRunner,
    ParallelismProfiler, ProfileSpec, ReferencePolicy, RunControl, restore,
    inspect_checkpoint,
    GraphReducer, ReductionState, IsingProblem, SpinConstraint,
    QAOAStatevectorCorrelationSource, MaxAbsCorrelationSelector, ExactCPUFinisher,
)
from QA_max_cut_lib import tools
from QA_max_cut_lib.generators import (
    partition, RandomPartitionBuilder, GreedyAbsWeightPartitionBuilder,
)
from QA_max_cut_lib.solvers.ls_components import MeanCorrelationAggregator
from QA_max_cut_lib.solvers.objectives import validate_objective, objective_value
from QA_max_cut_lib._version import __version__
```

`SolverFactory.available()` возвращает `("exact_cpu", "qaoa_cpu", "rqaoa_cpu", "ls_rqaoa_cpu", "qaoa_gpu", "rqaoa_gpu", "ls_rqaoa_gpu", "qaoa_in_qaoa_cpu", "qaoa_in_qaoa_gpu", "gw_cpu", "simulated_annealing_cpu", "refined")`. `from_config(config)` выбирает класс по `config["type"]`; неизвестный тип — `ValueError`. Фабрика не выполняет расчёт.

Общий конструктор солвера — `ClassName(config=None)`. Он копирует конфигурацию. `solve(graph, *, report=None, logs=None, num_treads=None)` принимает новый граф и возвращает `dict[int, int]`. Аргументы уровня вывода и числа потоков переопределяют конфигурацию данного запуска. Один экземпляр рассчитан на один активный запуск. Для одновременных расчётов нужны разные экземпляры.

Повторный `solve` заново создаёт RNG из настроенного seed, очищает предыдущий результат в солвере и не использует прежнее состояние. Явного warm start в этой версии нет. `last_run` — независимый снимок; сохраните его в список для собственной истории. После изменения графа прежний снимок остаётся прежним. Параллельно изменять переданный граф во время создания снимка не следует.

До первого запуска `result` и `last_run` равны `None`. Необязательные научные атрибуты до расчёта равны `None`. Ошибка предварительной проверки не оставляет прежний результат как новый. Ошибка вычисления создаёт доступный `last_run` со статусом `failed` и повторно возбуждает исключение. Ошибка экспорта после успешного расчёта оставляет научный результат доступным.

## WeightedGraph

Сигнатура: `WeightedGraph(edges=(), nodes=())`.

| Аргумент/атрибут | Тип | Смысл |
|---|---|---|
| `edges` | iterable троек `(u,v,w)` | Явные конечные действительные веса |
| `nodes` | iterable целых | В том числе изолированные вершины; исходный порядок сохраняется |
| `.nodes` | tuple[int] | Порядок внешних меток; метки с пропусками и отрицательные допустимы |
| `.edges` | tuple троек | Каждое неориентированное ребро один раз |
| `.version` | int | Счётчик операций изменения экземпляра |
| `len(graph)` | int | Число вершин |

Булевы метки, петли, дубликаты меток в `nodes`, повторные неориентированные рёбра конструктора, NaN/Inf и булевы веса отклоняются. Отрицательный и нулевой вес допустимы. Ребро нулевого веса хранится как ребро и учитывается в структурной статистике; отсутствие ребра — отдельный случай.

| Метод | Возвращает | Действие |
|---|---|---|
| `add_node(node)` | None | Добавляет отсутствующую вершину |
| `set_weight(u,v,weight)` | None | Добавляет/обновляет ребро и отсутствующие концы |
| `remove_edge(u,v)` | None | Удаляет ребро; концы сохраняются; отсутствующее ребро — KeyError |
| `weight(u,v)` | float | Вес ребра либо 0 для отсутствующей пары |
| `cut_value(solution)` | float | C(s) по рёбрам; проверяет точное совпадение вершин и целые спины ±1 |
| `subgraph(nodes)` | WeightedGraph | Индуцированный подграф с заданным порядком исходных меток |
| `snapshot()` | dict | Взвешенный снимок, версия, рецепт, SHA-256 fingerprint |
| `from_snapshot(data)` | WeightedGraph | Восстановление снимка с проверкой fingerprint |
| `copy()` | WeightedGraph | Независимая копия |

Fingerprint учитывает содержимое задачи (метки, пары, веса), а не порядок вставки. Порядок отдельно фиксируется в `node_order`; его необходимо учитывать при интерпретации состояний и корреляций.

## Редукция Max-Cut и Изинга

Ядро не выбирает корреляции и пару. Оно принимает явное ограничение
`s_removed = sign * s_retained` и выполняет точное алгебраическое
преобразование. Исходный `WeightedGraph` не изменяется.

```python
from QA_max_cut_lib import GraphReducer, IsingProblem

reducer = GraphReducer()
state = reducer.start(graph)
event = reducer.reduce(state, removed=10, retained=30, sign=-1,
                       correlation=-0.6)
solution = reducer.restore(state, {30: 1, 80: -1, 101: 1})
snapshot = reducer.snapshot(state)
restored_state = reducer.from_snapshot(snapshot)
```

`GraphReducer` не хранит состояние. Все изменяемые данные находятся в
`ReductionState`. Один объект `ReductionState` описывает один рабочий процесс.
Для независимых процессов нужны разные состояния.

### IsingProblem

Сигнатура: `IsingProblem(constant=0.0, fields=None, couplings=None, nodes=())`.

| Аргумент/атрибут | Тип | Смысл |
|---|---|---|
| `constant` | float | Полная константа `c`; не сдвиг Max-Cut |
| `fields` | mapping int → float | Локальные поля `h_i`; отсутствующее поле равно 0 |
| `couplings` | mapping (int,int) → float | Пары `J_ij`, каждая неориентированная пара один раз |
| `nodes` | iterable int | Все вершины, включая изолированные; порядок сохраняется |
| `.nodes` | tuple[int] | Порядок вершин |
| `.fields` | dict[int,float] | Только ненулевые поля |
| `.couplings` | dict[(int,int),float] | Только ненулевые пары |
| `.edge_count` | int | Число ненулевых пар |
| `.sum_abs_couplings` | float | Сумма модулей ненулевых пар |
| `.energy(solution)` | float | Прямая сумма полинома для полного назначения |

`IsingProblem.from_maxcut(graph)` задаёт `J_ij=w_ij/2`, `h_i=0`,
`c=-Σw/2`. `to_weighted_graph()` возвращает граф `w=2J` только при нулевых
полях; иначе `ValueError`. Нулевые пары не хранятся. Малые ненулевые пары
сохраняются без скрытого порога.

### SpinConstraint

`SpinConstraint(removed, retained, sign)` — неизменяемая запись ограничения.
`sign` принимает только целые `+1` или `-1`. `removed` и `retained` — исходные
целые номера, не равные друг другу.

### ReductionState

| Атрибут/метод | Смысл |
|---|---|
| `.original_nodes` | Все исходные номера в порядке вставки |
| `.original_graph` | Независимая копия исходного графа для Max-Cut |
| `.original_ising` | Независимая копия исходной задачи для общего Изинга |
| `.ising` | Текущая рабочая задача; рассматривать как read-only |
| `.current_nodes` | Текущие номера |
| `.constraints` | Ограничения в порядке применения |
| `.trace` | События редукций в порядке применения |
| `.cut_offset` | Накопленный `b_t`; `None` для общего Изинга |
| `.version` | Число применённых редукций |
| `.current_graph()` | Текущий граф `w=2J` для Max-Cut-состояния |
| `.current_cut_value(residual)` | `b_t + C_current(residual)` |
| `.original_cut_value(solution)` | `C` на исходном графе |
| `.original_energy(solution)` | `E=-C` или исходный полином Изинга |
| `.current_energy(residual)` | Текущий полином Изинга |
| `.apply_pair(...)` | Применить одну редукцию и вернуть событие |
| `.restore(residual)` | Восстановить все исходные спины в обратном порядке |
| `.snapshot()` | JSON-совместимый снимок математического состояния |
| `ReductionState.from_snapshot(data)` | Восстановить состояние |

`GraphReducer.start(graph)` и `start_from_ising(problem)` создают состояние.
`GraphReducer.reduce(state, ...)` делегирует `apply_pair`.
`GraphReducer.restore(state, residual)` делегирует восстановление.
`GraphReducer.snapshot/from_snapshot` сохраняют и читают состояние.

### Событие редукции

Каждый вызов `reduce` добавляет одно событие в `state.trace`.

| Поле | Значение |
|---|---|
| `step`, `task_version_before/after` | Номер шага и версии рабочей задачи |
| `removed_node`, `retained_node`, `sign` | Исходные номера и ограничение |
| `selected_coupling` | `J_uv` до преобразования |
| `selected_weight` | `w_uv=2J_uv` для Max-Cut-происхождения; иначе `None` |
| `physical_correlation`, `correlation_reason` | Переданный `c_uv` или причина отсутствия |
| `selection_score`, `selection_score_reason` | Отдельная оценка выбора, не корреляция |
| `fixed_energy` | `-w_uv(1-sign)/2` для Max-Cut |
| `expected_fixed_energy` | `-w_uv(1-c_uv)/2`, если есть корреляция |
| `constraint_probability` | `(1+sign*c_uv)/2`, если есть корреляция |
| `cut_offset`, `cut_offset_delta` | `b_t` и `Δb` |
| `ising_constant`, `ising_constant_before`, `ising_constant_delta` | Полная константа Изинга |
| `vertices_after`, `edges_after` | Структура после шага; рёбра — ненулевые пары |
| `sum_abs_weight`, `mean_abs_weight` | Область ненулевых рёбер после шага |
| `sum_abs_coupling`, `mean_abs_coupling` | То же для `J` |
| `mean_abs_weight_reason` | Причина `None` для пустой области |
| `maxcut_metrics_applicable`, `maxcut_metrics_reason` | Применимость графовых метрик |

Для общего Изинга с ненулевыми полями `selected_weight`, `fixed_energy`,
`cut_offset` и `mean_abs_weight` равны `None` с причиной
`maxcut_metrics_reason`. Поля сохраняются в рабочей задаче и учитываются в
`ising_constant` и `.current_energy`.

### Снимок состояния

Снимок содержит `schema_version=1`, `state_type="reduction_state"`,
`energy_convention`, `version`, `maxcut_origin`, `original_nodes`,
`original_graph`, `original_ising`, `ising`, `cut_offset`, `constraints` и
`trace`. Он JSON-совместим, но не является checkpoint/resume оптимизатора.
После `json.dumps`/`json.loads` снимок продолжает принимать редукции и
восстанавливать решение.

### Ошибки редукции

`TypeError` — номер не целый, вес не действительный, передан bool.
`ValueError` — самопара, отсутствующая или повторно удалённая вершина,
`sign` не ±1, корреляция вне `[-1,1]`, неполный остаток.
`ArithmeticError` — новая константа, поле или пара не конечны. Ошибочный
запрос не изменяет состояние.

## RQAOA на CPU

`RQAOACPU` — первый полный рекурсивный солвер. Он обучает QAOA на текущем
рабочем графе, получает физические корреляции `⟨Z_uZ_v⟩`, выбирает одну
разрешённую пару, применяет точное ограничение `s_u=σs_v` ядром этапа 02,
повторяет цикл и решает остаток `ExactCPU`. Исходные спины восстанавливаются в
обратном порядке. Это эвристика, а не точный солвер исходной задачи.

```python
from QA_max_cut_lib import SolverFactory, tools

config = tools.load_json("configs/solvers/rqaoa_cpu.json")
algorithm = SolverFactory.from_config(config)
solution = algorithm.solve(graph, report=1, logs=0)
result = algorithm.result
trajectory = algorithm.parameter_trajectory
history = algorithm.reduction_history
```

### Конфигурация

| Поле | Значение |
|---|---|
| `parameters.finish_at` | Целое ≥1; остаток такого размера решает точный финишёр |
| `parameters.max_reductions` | `null` или целое ≥0; 0 запрещает редукции |
| `correlation_source.type` | В этой версии только `qaoa_statevector_cpu` |
| `correlation_source.parameters.layers` | Глубина `p≥1`; порядок углов γ₀…γₚ₋₁,β₀…βₚ₋₁ |
| `correlation_source.parameters.max_vertices` | Защитный предел полного вектора, не более 24 |
| `correlation_source.objective` | Один из пяти критериев QAOA |
| `correlation_source.objective_reference` | `working_graph` или `original_graph` |
| `correlation_source.estimation` | `exact` или `shots` для критерия обучения |
| `correlation_source.optimizer` | ADAM и центральные разности |
| `correlation_source.initialization` | `fixed_angles` или `random_angles` |
| `correlation_source.step_initialization` | `restart` или `previous_step` |
| `correlation_source.correlations` | `exact` или `shots`; в shots обязателен `shots` |
| `selection.parameters.pairs` | `edges` или `all_pairs` |
| `selection.parameters.zero_sign` | `+1` или `-1` для точной нулевой корреляции |
| `finisher.type` | `exact_cpu` |
| `finisher.parameters.max_vertices` | Не меньше `finish_at` |
| `limits.shots_total` | `null` или строгий общий лимит шотов |
| `collect.reduction_history` | Компактные события редукций |
| `collect.parameter_trajectory` | Углы перед каждым слиянием |
| `collect.local_optimizer_history` | Истории локального ADAM |
| `collect.correlation_tables` | Полные таблицы корреляций |
| `collect.final_reduction_state` | JSON-снимок математического состояния |

Источник не имеет собственного `save`, публичного readout или отдельного
`shots_total`. Неизвестные поля и несовместимые сочетания отклоняются.
`num_treads` в этой версии принимает только 1. Значения `finish_at` и
`max_vertices` проверяются до выделения полного вектора.

### Результатные поля

Помимо общих полей `RunRecord` доступны:

| Поле | Смысл |
|---|---|
| `reductions_completed` | Число применённых ограничений |
| `remaining_node_order` | Внешние номера финального остатка |
| `cut_offset` | Накопленный сдвиг `b_t` |
| `ising_constant` | Полная константа `c_t` рабочей задачи |
| `reduced_solution`, `reduced_cut_value` | Решение и разрез остатка |
| `solution_source` | `exact_finisher`, zero-interaction или `all_plus_reduced_fallback` |
| `termination_reason` | `finish_at`, `no_interactions`, `max_reductions`, `shot_budget` |
| `reduction_history` | События с весом, корреляцией, `σ`, score, энергиями |
| `parameter_trajectory` | Углы до соответствующего слияния |
| `local_optimizer_history` | Локальные истории оптимизации |
| `correlation_tables` | Таблицы с покрытием и версией |
| `final_reduction_state` | Снимок математического состояния |
| `resources` | Шоты, оценки, шаги, этапы, `num_treads` |

Для отсутствующего ребра `selected_weight=0`. `fixed_energy=-w(1-σ)/2`;
`expected_fixed_energy=-w(1-c)/2`. Суммы этих вкладов не заменяют полную
энергию решения или константу редукции. `score=abs(c)` хранится отдельно от
знаковой корреляции.

При `status="stopped"` возвращается полное допустимое решение
`all_plus_reduced_fallback`, а не частичный словарь. Глобальная оптимальность
не заявляется.

### Инструменты RQAOA

| Функция | Назначение |
|---|---|
| `tools.plot_reduction_weights(source, config)` | Вес выбранной пары и средний модуль ненулевого веса |
| `tools.plot_reduction_energy(source, config)` | `fixed`, `expected`, накопленные суммы, `cut_offset`, `ising_constant` |
| `tools.plot_parameter_trajectories(source, config)` | Все `binomial(2p,2)` плоских траекторий; при `layout.panels_per_page` — детерминированные страницы `filename_page_001` |
| `tools.plot_parameter_trajectories_3d(source, config)` | Автономный HTML; Z — номер редукции |
| `tools.plot_analytic_gamma_profile(source, config)` | `min_beta E(gamma,beta)` и `beta_star(gamma)` для выбранного шага |

`source` — солвер, `RunRecord` или `result`. Графики читают сохранённую запись
и не запускают солвер, оптимизатор или новые измерения. При `save.format="html"`
3D-график встраивает Plotly.js и данные; файл открывается без сервера и сети.
Пустая история даёт явную ошибку с указанием включить соответствующий `collect`.

## Аналитический источник p=1

`AnalyticP1Evaluator` вычисляет моменты одного слоя без полного вектора
состояния. `AnalyticP1CorrelationSource` с типом `qaoa_analytic_p1_cpu`
организует поиск углов и возвращает общий `CorrelationResult` этапа 03.

```python
from QA_max_cut_lib import AnalyticP1Evaluator, SolverFactory, tools

evaluator = AnalyticP1Evaluator(graph)
correlations = evaluator.correlations(0.3, 0.2, pair_domain="all_pairs")
profile = evaluator.energy_min_beta(0.3)
print(profile["beta_star"], profile["energy_min"])

config = tools.load_json("configs/solvers/rqaoa_analytic_p1_cpu.json")
algorithm = SolverFactory.from_config(config)
solution = algorithm.solve(graph)
```

### Формулы и область

Для `E=-C`, `J=w/2`, `|+>` и одного слоя

\[
c_{uv}=a_{uv}\sin(4\beta)+d_{uv}\sin^2(2\beta),
\]

где `a_{uv}=0.5 sin(gamma w_uv)(P_{u\setminus v}+P_{v\setminus u})`,
`d_{uv}=0.5(P^-_{uv}-P^+_{uv})`. Произведения берутся по всем `k`, кроме `u,v`.
Несмежная пара может иметь ненулевой коррелятор из-за общих соседей.

Средняя энергия:

\[
E(\gamma,\beta)=E_0+A\sin(4\beta)+B\sin^2(2\beta),
\]

\[
\min_\beta E=F_0-\operatorname{hypot}(F_s,F_c),\qquad
\beta_*=\frac14\operatorname{atan2}(-F_s,-F_c)\bmod\frac{\pi}{2}.
\]

`E0=-0.5 Σw`, `F0=E0+B/2`, `Fs=A`, `Fc=-B/2`. При `R=0` выбирается
`beta_star=0`. Минимум по gamma — лучшая точка явной конечной сетки, не
доказанный непрерывный глобальный оптимум.

### Параметры вычислителя

| Аргумент/поле | Значение |
|---|---|
| `AnalyticP1Evaluator(graph, pair_batch_size=2048)` | Плотная матрица `N×N`; пакеты пар `O(pair_batch_size*N)` |
| `parameters.layers` | Только 1 |
| `parameters.dtype` | Только `float64` |
| `parameters.pair_batch_size` | Положительное целое |
| `optimizer.type` | `gamma_grid_exact_beta` |
| `optimizer.parameters.gamma_bounds` | Две возрастающие конечные границы, включительно |
| `optimizer.parameters.points` | Целое ≥2 |
| `optimizer.parameters.gamma_batch_size` | Положительный размер планировочной группы |
| `estimation.mode` | Только `exact` |
| `correlations.mode` | Только `exact` |
| `objective.type` | Только `mean_cut` |

Поля `initialization`, `step_initialization`, ADAM, `shots`, полное
распределение, `readout`, CVaR/Gibbs/Best-of-K/Threshold Probability, локальные
поля и `p>1` отклоняются. Источник не строит `2**N`; он не поддерживает
дискретный readout. Аналитический запуск не расходует шоты, поэтому
`limits.shots_total=0` допустим.

### Результат источника и RQAOA

`CorrelationResult` содержит `node_order`, версию задачи, покрытие, физические
корреляции, `[gamma,beta]`, `layers=1` и `training`:

| Поле `training` | Смысл |
|---|---|
| `gamma_profile.type` | `analytic_gamma_grid` |
| `gamma_profile.gamma_values` | Полная сетка, включая границы |
| `gamma_profile.profile_energy` | `min_beta E` на сетке |
| `gamma_profile.profile_energy_original` | То же со сдвигом `-b_t` при `original_graph` |
| `gamma_profile.beta_star` | Точный beta для каждой gamma |
| `gamma_profile.A`, `B`, `E0` | Коэффициенты энергии |
| `gamma_profile.best_index`, `best_gamma`, `best_beta` | Детерминированная лучшая точка |
| `classical_counters` | `gamma_points`, `coefficient_passes`, `pair_correlators`, `classical_seconds` |

Корневой `RQAOA` сохраняет `shots_used=0`, `training_shots=0`,
`correlation_shots=0`, `readout_shots=0`, а также `gamma_points`,
`coefficient_passes`, `pair_correlators` и `classical_seconds`. Профиль входит
в `local_optimizer_history` с `profile_type="analytic_gamma_grid"`.

### Инструмент профиля

`tools.plot_analytic_gamma_profile(source, config)` строит `min_beta E(gamma,beta)`
и `beta_star(gamma)` для выбранного шага редукции. Он читает
`local_optimizer_history` и не запускает новый поиск. При отсутствии профиля
выдаётся ошибка с указанием включить `collect.local_optimizer_history`.

## Generators

`GraphFactory.from_recipe(recipe)` принимает:

| Поле | Значение/по умолчанию |
|---|---|
| `schema_version` | 1 |
| `type` | `complete`, `gnm`, `erdos_renyi` |
| `parameters.n_vertices` | Обязательное целое 0…1000 |
| `parameters.nodes` | Список из n уникальных целых; по умолчанию 0…n−1 |
| `parameters.n_edges` | Только для `gnm`, целое 0…n(n−1)/2 |
| `parameters.probability` | Только для `erdos_renyi`, число в [0,1] |
| `seed` | Seed топологии; по умолчанию null |
| `weights.type` | `constant` по умолчанию, либо `signed_discrete`, `uniform` |
| `weights.parameters.value` | Для `constant`, по умолчанию 1.0 |
| `weights.parameters.values` | Для `signed_discrete`, непустой список конечных весов; равновероятная выборка с повторениями |
| `weights.parameters.low`, `.high` | Для `uniform`, конечные low<high; интервал [low, high) |
| `weights.seed` | Отдельный RNG весов; по умолчанию seed топологии |

Одинаковые фиксированные seeds дают одинаковые графы в той же программной среде. Для долговременной воспроизводимости сохраняется фактический снимок: один рецепт без версий зависимостей не обещает побитового совпадения во всех будущих NumPy.

`partition(graph, max_vertices, *, seed=0)` возвращает список непересекающихся индуцированных подграфов, покрывающих все вершины. Максимальный размер — положительное целое. Рёбра между частями не входят в подграфы; это не разложение исходной целевой функции без остатка. Пустому графу соответствует пустой список.

## Общая конфигурация солвера

Неизвестные корневые поля отклоняются. Вложенные поля алгоритмов перечислены ниже.

| Поле | Тип/по умолчанию | Назначение |
|---|---|---|
| `schema_version` | int, 1 | Версия схемы |
| `type` | str | Тип класса; при прямом конструкторе можно опустить |
| `seed` | int или null, null | Локальный NumPy RNG, заново на каждом запуске |
| `num_treads` | int, 1 | В этой версии только 1; другие значения — ValueError |
| `parameters` | dict | Физические/вычислительные параметры данного солвера |
| `output.report` | 0/1/2, 0 | Нет/краткий/подробный терминальный отчёт |
| `output.logs` | 0/1/2, 0 | Нет/начало и конец/дополнительно шаги |
| `output.color` | auto/always/never, auto | ANSI-цвет; auto включается в TTY |
| `collect` | dict | Необязательные массивы/история; по умолчанию отключены |
| `save` | dict, отсутствует | Общий экспорт после вычисления |
| `checkpoint.enabled` | bool, false | true вызывает ошибку: resume ещё не реализован |

`report` и `logs` независимы. Отчёт в JSON не зависит от терминального уровня. Импорт не выводит текст и не запускает примеры. Цвета отображаются терминалами с поддержкой ANSI; JSON не содержит цветовых кодов.

## ExactCPU

| Поле `parameters` | По умолчанию | Допустимость |
|---|---|---|
| `max_vertices` | 24 | Положительное целое ≤30; это защитный предел, а не гарантия быстрого расчёта |
| `batch_size` | 65536 | Положительное целое, число представителей в рабочем блоке |
| `tie_tolerance` | 0.0 | Конечное неотрицательное абсолютное отклонение C для подсчёта вырождения |

Поля обучения, readout и непустой `collect` для этого класса недопустимы. `limits.shots_total` допустим для общего интерфейса; классический перебор не использует шоты. Число проверенных представителей с учётом двух проходов — `states_evaluated`. Число этапов — число блоков обоих проходов.

Дополнительные атрибуты: `worst_cut_value` (минимум C), `optimal_count` (число полных спиновых решений с учётом допуска), `states_evaluated`. Вероятности, состояние и история оптимизатора не применяются и равны `None`.

## QAOACPU

### Схема и критерий

| Поле | По умолчанию | Ограничения |
|---|---|---|
| `parameters.layers` | 1 | Положительное целое p |
| `parameters.max_vertices` | 20 | Положительное целое; дополнительный жёсткий предел симулятора 24 |
| `parameters.dtype` | complex128 | complex128 или complex64; форма состояния (2^N,) |
| `objective.type` | mean_cut | mean_cut/cvar/gibbs/best_of_k/threshold_probability |
| `objective.parameters` | {} для mean_cut | cvar: alpha∈(0,1]; gibbs: eta>0; best_of_k: целое k≥1; threshold_probability: конечный threshold |
| `estimation.mode` | exact | exact или shots |
| `estimation.shots_per_evaluation` | 256 | Положительное целое; используется только в shots |
| `initialization.type` | fixed_angles | fixed_angles или random_angles |
| `initialization.parameters.values` | p значений 0.5, затем p значений 0.3 | Для fixed_angles: ровно 2p конечных углов в радианах |

У `random_angles` параметры отсутствуют: γ независимо выбираются из [−π,π), β из [−π/2,π/2). Это эвристическое окно инициализации, а не утверждение о периоде взвешенной задачи. Обучение не ограничивает последующие углы этим окном.

### Оптимизатор

| Поле | По умолчанию | Назначение |
|---|---|---|
| `optimizer.type` | adam | Только adam в этом этапе |
| `optimizer.parameters.max_steps` | 20 | Целое ≥0; 0 выполняет только readout начального состояния |
| `.learning_rate` | 0.05 | Конечное положительное число |
| `.beta1`, `.beta2` | 0.9, 0.999 | Коэффициенты моментов в [0,1) |
| `.epsilon` | 1e-8 | Положительная стабилизирующая константа |
| `.gradient_tolerance` | 0.0 | Неотрицательный порог нормы градиента; 0 отключает остановку |
| `optimizer.gradient.type` | central_difference | Только центральные разности |
| `optimizer.gradient.parameters.step` | 0.01 | Конечный положительный шаг h |

### Readout, бюджет и накопление

| Поле | По умолчанию | Назначение |
|---|---|---|
| `readout.mode` | most_probable | Наиболее вероятное состояние либо best_of_shots |
| `readout.shots` | Обязателен для best_of_shots | Положительное целое M; для most_probable не задаётся |
| `limits.shots_total` | null | Общий целый бюджет ≥0 либо без ограничения |
| `collect.optimizer_history` | false | Наблюдения в центрах шагов |
| `collect.objective_history` | false | Все оценки центра и градиентных проб |
| `collect.probabilities` | false | Полный вектор вероятностей, float64 |
| `collect.statevector` | false | Полный комплексный вектор выбранного dtype |
| `collect.correlations` | false | Физическая матрица ⟨ZᵢZⱼ⟩ формы (N,N), float64 |

Шоты — выборки классического симулятора, без модели аппаратного шума. Конечный readout резервируется до обучения. Статус `stopped` с причиной `shot_budget` означает корректную раннюю остановку с решением. Значение `objective_value=None` допустимо, если обучение не выполнило ни одной оценки.

### Результаты и атрибуты

Каждое поле читается напрямую, например `algorithm.resources`, и через `algorithm.result["resources"]`. Прямой доступ возвращает независимые копии массивов/словарей; большие данные разумно получить один раз. Отсутствующие необязательные данные не вычисляются по запросу инструмента.

| Поле | Тип и смысл |
|---|---|
| `solution` | dict[int,int], по одной ±1 на каждую исходную вершину |
| `cut_value`, `solution_energy` | float: C и −C выбранного дискретного решения |
| `expected_cut`, `expected_energy` | float: точные ожидания конечного состояния; для ExactCPU None |
| `expected_cut_source` | У QAOA: exact_final_statevector |
| `objective_value`, `optimizer_loss` | Лучшее оценённое F и −F; None, если оценок не было |
| `objective_source` | best_training_evaluation либо not_evaluated |
| `optimal_angles` | ndarray (2p,), лучший уже оценённый вектор; при отсутствии обучения начальный |
| `optimizer_history` | list[dict] или None; поля optimizer_step, objective_evaluations, objective_value, optimizer_loss, gradient_norm, angles, shots_used |
| `objective_history` | list[dict] или None; evaluation_id, optimizer_step, kind, objective_value, optimizer_loss, angles, shots_used |
| `probabilities`, `statevector`, `correlations` | Массивы описанных выше форм либо None |
| `data_status` | Собрано/не собрано/не применимо для дополнительных данных |
| `node_order` | list[int], порядок соответствия индексов внешним вершинам |
| `graph` | Фактически решённый взвешенный снимок и рецепт |
| `config` | Полностью разрешённая конфигурация с использованными значениями по умолчанию |
| `status`, `termination_reason` | completed/stopped/failed и конкретная причина |
| `resources` | Счётчики, перечисленные далее |
| `timing.compute_seconds` | Время создания снимка и вычисления до построения итоговой записи; без печати и файлового экспорта |
| `report` | Структурированные итоговые качество, статус, время и ресурсы |
| `started_at`, `finished_at` | ISO 8601, UTC |
| `run_id` | UUID данного запуска |
| `algorithm_type`, `package_version`, `schema_version`, `seed`, `num_treads` | Происхождение и настройки запуска |

`resources` QAOA: `shots_used`, `training_shots`, `readout_shots`, `objective_evaluations`, `optimizer_steps`, `num_stages`, `num_treads`. `num_stages` — полные оптимизационные шаги плюс конечный readout. Шоты в `optimizer_history` относятся к обучению; итоговые `shots_used` дополнительно включают readout.

## RunRecord и JSON

`RunRecord(fields)` копирует исходные данные. `.get(name, default=None)` и `record.attribute` возвращают независимые нативные значения. `.field_names` содержит доступные имена. `.result` создаёт полный JSON-совместимый снимок. Это не журнал всех запусков солвера.

Массив в JSON: `{"__type__":"ndarray","dtype":"<f8","shape":[...],"data":[...]}`. Для комплексного массива вместо `data` используются `real` и `imag`; dtype и shape сохраняются. Комплексный скаляр имеет `__type__="complex"`. Ключи спинового решения становятся строками по правилам JSON и восстанавливаются как целые при загрузке. NaN/Inf и неизвестные типы не сериализуются молча.

`tools.load_result(path, *, load_original_data=False)` читает полный результат либо выборочный экспорт и возвращает `RunRecord`. При `True` также читает `.npy` из той же папки без pickle. Если необходимые графику данные не были экспортированы/накоплены, инструмент сообщает об их отсутствии.

## Общий экспорт

`tools.save_result(source, config)` возвращает `SavedFiles` с `.primary` (Path JSON) и `.files` (tuple всех путей). `source` — солвер, RunRecord либо полный JSON-словарь.

| Поле save | По умолчанию | Правило |
|---|---|---|
| `path` | Обязательно | Папка назначения |
| `filename` | Обязательно | Базовое имя без пути и зарезервированных символов |
| `atts` | Все поля записи | Уникальные имена атрибутов для JSON; [] оставляет метаданные и отчёт |
| `save_original_data` | [] | Имена собранных числовых ndarray для файлов filename_attribute.npy |
| `makedir` | false | Создать отсутствующий родительский путь |
| `savedir` | false | Создать новую папку path/filename и положить туда весь комплект |

Массивы, перечисленные одновременно в двух списках, сохраняются в обоих представлениях. Наборы независимы. Нативный экспорт иных специальных типов будет добавляться с соответствующими алгоритмами; текущий неподдерживаемый тип вызывает ошибку.

Основной JSON содержит `metadata`, `attributes`, `original_data`, `saved_at` и `export_complete`. Метаданные и отчёт обязательны независимо от `atts` и уровней вывода. Ссылки на NPY относительные, с dtype/shape. Комплект можно переносить целиком. Файл JSON публикуется последним; нет гарантии атомарного появления всех файлов одновременно. При обычной ошибке выполняется очистка файлов текущей попытки. При аварийном завершении процесса могут остаться дополнительные файлы без основного JSON; они не считаются завершённым экспортом.

Существующая папка `path/filename` при `savedir=True`, даже пустая, вызывает `FileExistsError`. При `False` конфликт любого конечного файла также отклоняется. Экспорт не перезаписывает существующие данные даже при одновременных попытках с одним именем. Проверяется весь подготовленный комплект. Предварительная сериализация может требовать дополнительной памяти для больших массивов; потоковый экспорт ещё не реализован.

## Численные инструменты

| Функция | Аргументы | Результат |
|---|---|---|
| `correlation_matrix(probabilities)` | Нормированный конечный неотрицательный вектор длины 2^N | Физические ⟨ZᵢZⱼ⟩, ndarray (N,N); порядок тот же, что у битов |
| `quality_score(cut_value, optimum, worst)` | Конечные значения одной задачи; optimum≥worst | (cut_value−worst)/(optimum−worst), либо None для нулевого диапазона |
| `load_json(path)` | UTF-8 JSON-файл | Стандартный разобранный JSON |

`quality_score` не проверяет тождество графов: это обязанность вызывающего кода. В текущем этапе ориентир вычисляет ExactCPU; GW и база лучших известных решений пока отсутствуют.

## Карты и графики

`scan_landscape(graph, config)` не изменяет граф и возвращает RunRecord. Поля конфигурации: `gamma`, `beta` — строго возрастающие конечные списки длиной 2…256 каждый; `objective` — как выше; `quantity` — `optimizer_loss` по умолчанию либо `objective_value`; необязательный общий `save`. Допускается 1…16 вершин и p=1. Атрибуты записи: `gamma` (G,), `beta` (B,), `values` (B,G), graph, config, timestamps, timing, resources. Число `resources.grid_points` равно B·G; шотов нет.

Все графические функции возвращают Matplotlib `Figure`. `figure.qa_record` — результат инструмента с конфигурацией, происхождением, `plot_data` и временной отметкой; `figure.qa_saved_files` появляется при сохранении. Для пользовательской истории инструмента можно сохранить `.qa_record`. Построение не вызывает повторное обучение и не включает сбор пропущенных данных.

| Функция | Вход и дополнительные поля config |
|---|---|
| `plot_graph(source, config=None)` | WeightedGraph или результат; рисует исходный снимок |
| `plot_graph_solution(source, solution=None, config=None)` | Для графа явно передать dict решения; для солвера/записи можно опустить solution |
| `plot_graph_info(source, config=None)` | Добавляет N, число рёбер, плотность, компоненты, изоляты, min/max/средний модуль веса |
| `plot_probabilities(source, config=None)` | Требует probabilities; `selection.top_k=16`, положительное целое |
| `plot_optimization(source, config=None)` | Требует непустую optimizer_history; `axes.x`: optimizer_step/objective_evaluations/shots_used; `axes.y`: optimizer_loss (по умолчанию)/objective_value/gradient_norm |
| `plot_landscape(source, config=None)` | Запись scan_landscape, γ на X, β на Y, цвет — сохранённая величина |

Общие поля визуализации:

| Поле config | По умолчанию/смысл |
|---|---|
| `title` | Заголовок конкретного инструмента |
| `style.text_engine` | mathtext или latex; по умолчанию mathtext |
| `style.figsize` | [10,6] в дюймах; при metadata снизу добавляется место подписи |
| `style.label_size`, `.tick_size`, `.legend_size` | 19,16,15 |
| `style.latex_preamble` | Пользовательская строка TeX при latex |
| `output.show` | false; при true дополнительно plt.show() |
| `graph.layout` | Для графов spring по умолчанию либо circular |
| `graph.seed` | 0 для spring layout |
| `graph.weight_labels` | true |

Вероятностная диаграмма сохраняет исходные массы: top-K не перенормируется; заголовок показывает отображённую массу и остаток. Синие/оранжевые вершины обозначают +1/−1; пересечённые рёбра выделены, отрицательные показаны пунктиром. Для крупного плотного графа разумно отключить подписи весов; фильтр рёбер и специализированная диаграмма больших распределений будут добавлены позже.

Для сохранения графика `save` использует общие `path`, `filename`, `makedir`, `savedir` и дополнительно `format` (png по умолчанию, также jpeg/pdf/svg), `DPI` (250) и `metadata` (false). При metadata=True под графиком отображаются дата/время результата, тип и параметры алгоритма, параметры графа и fingerprint. Полный взвешенный граф хранится в записи; видимая подпись не печатает таблицу всех рёбер. Для запуска без рецепта подпись содержит размер и fingerprint, а веса находятся в сохранённом снимке.

Указание `atts` или `save_original_data` в save графика включает дополнительный JSON результата инструмента. В первой версии числовые серии находятся внутри `plot_data`, поэтому нативный экспорт из этой записи не разбирает вложенные поля по точечным путям. Для нативного экспорта карты передайте сам RunRecord сканера в `save_result` и выберите `values`.

Интерактивный HTML, трёхмерные поверхности и траектории редукции реализованы для сохранённых траекторий `RQAOA`/`LS-RQAOA` через `plot_parameter_trajectories_3d`; Plotly.js встраивается в автономный HTML.

## LS-RQAOA (`ls_rqaoa_cpu`)

`LSRQAOACPU.solve(graph, *, report=None, logs=None, num_treads=None)` выполняет один и тот же план локальных задач последовательно при `num_treads=1` и в пуле процессов `spawn` при большем значении. Явный аргумент `num_treads` переопределяет JSON. Запрошенное и фактически применённое значения находятся в `resources.num_treads_requested` и `resources.num_treads_applied`.

### Секции конфигурации

| Секция | Обязательные поля | Смысл |
|---|---|---|
| `parameters` | `finish_at`, `max_reductions` | Размер точного остатка и необязательный предел числа глобальных редукций |
| `subgraph_builder` | `type`, `parameters.max_size`, `parameters.rounds` | `random_partition` или `greedy_abs_weight_partition`; `boundary` только `induced_no_fields` |
| `correlation_source` | `type`, `objective_reference` | `qaoa_statevector_cpu` p≥1 или `qaoa_analytic_p1_cpu`; локальная цель только `working_graph` |
| `aggregation` | `type` | В этой версии только `mean`; среднее по числу наблюдений каждой пары |
| `selection` | `type`, `parameters.pairs` | `max_abs_correlation`, `edges` или `all_pairs`, `zero_sign` |
| `coverage` | `require_full`, `empty_policy` | `anchor_pair` — явная задача ремонта пустого покрытия |
| `update` | `type` | Только `refresh_all`: на следующем шаге строится новый план |
| `execution` | `backend`, `start_method`, `native_threads` | `process`, `spawn`, `1` |
| `budget_policy` | `type` | Только `step_round_robin` |
| `finisher` | `type`, `parameters.max_vertices` | Существующий `ExactCPUFinisher` |
| `limits` | `shots_total`, `max_local_tasks_total` | Общий шотовый лимит и предел числа локальных вызовов на шаг |
| `collect` | флаги | `reduction_history`, `parameter_trajectory`, `local_task_history`, `local_optimizer_history`, `correlation_tables`, `final_reduction_state` |

`full_graph_when_fits=true` у построителя создаёт одну локальную задачу на всём остатке, если он помещается в `max_size`; в этом режиме обычные повторные проходы не создаются. Дубли одного набора в разных проходах являются отдельными задачами с разными ID/seeds.

### Покрытие и бюджеты

Разрешённая область `A_t` определяется `selection.parameters.pairs`. Наблюдаемая область — объединение разрешённых пар, обе вершины которых попали в запланированные задачи. При `require_full=true` неполное покрытие обнаруживается до обучения и вызывает `ValueError`. При `false` выбор идёт только по наблюдённым парам; отсутствующая оценка не считается нулём.

`budget_policy=step_round_robin` резервирует сумму обязательных корреляционных шотов всех задач шага, затем выдаёт полные итерации по одной каждой задаче в каноническом порядке. Для шотовой оценки statevector стоимость итерации — `(4*p+1)*M_train`; точные и аналитические итерации стоят 0. Квоты фиксируются до запуска worker-процессов. При недостатке минимума шаг не начинается, граф не меняется, возвращается полное решение и `status="stopped"`.

### Результаты LS

Кроме общих полей: `cut_offset`, `ising_constant`, `reduced_solution`, `coverage_history`, `reduction_history`, `parameter_trajectory` (строки локальных задач), `local_task_history`, `local_optimizer_history`, `correlation_tables`, `final_reduction_state`. В событии редукции: `physical_correlation`, `correlation_origin="local_subgraph_mixture"`, `expected_fixed_energy_local`, `local_task_ids`, `aggregation_observations`, `coverage`, `global_expected_energy=None` и причина. Локальное среднее не объявляется глобальным ожиданием полного графа.

Построители:

```python
from QA_max_cut_lib.generators import (
    RandomPartitionBuilder, GreedyAbsWeightPartitionBuilder,
)
builder = GreedyAbsWeightPartitionBuilder({
    "type": "greedy_abs_weight_partition",
    "parameters": {"max_size": 10, "rounds": 2, "full_graph_when_fits": True},
}, seed=42)
tasks = builder.build(graph, version=0)
```

`build()` возвращает `round`, `block`, `nodes`, `task_id` и `boundary`; он не меняет граф и не вызывает QAOA.

## Классические методы (`gw_cpu`, `simulated_annealing_cpu`, `refined`)

`GoemansWilliamsonCPU` (`gw_cpu`) решает SDP через CVXPY/SCS и округляет независимыми гиперплоскостями. `parameters` содержит `num_hyperplanes`, `solver="SCS"`, `eps_abs`, `eps_rel`, `max_iters`, `allow_inaccurate`, `psd_tolerance`, `diag_tolerance` и `repair_tolerance`. `sdp_upper_bound` равен `null`: строгий dual certificate в этом этапе не строится; численное значение релаксации не выдаётся за сертифицированную границу.

`SimulatedAnnealingCPU` (`simulated_annealing_cpu`) строит spin BQM с `h=0`, `J=w/2`, `offset=-sum(w)/2` и вызывает классический `dwave.samplers.SimulatedAnnealingSampler` по одному логическому read на вызов. `parameters`: `num_reads`, `num_sweeps`, `beta_schedule_type` (`linear`/`geometric`), `beta_range`. `limits`: `anneal_reads_total`, `anneal_sweeps_total`. Это не quantum shots.

`ExactBlockRefiner` (`exact_block`) принимает `refine(graph, spins, context=...)` и возвращает улучшенную независимую копию. `selector` — `explicit_blocks` или `pair_greedy_abs_weight` с `max_seed_pairs`/`ranked_pairs`. Блоки применяются последовательно; поля пересчитываются после каждого принятого блока. `limits.refinement_states_total` резервирует полные `2**|S|` перечисления до старта блока.

`RefinedSolver` (`refined`) принимает `base_solver` и `refiner`; `base_run` и `refinement_run` независимы. Итоговые `cut_value` и `solution` относятся к улучшенному назначению; квантовые углы и `expected_energy` остаются в `base_run`. `limits` допускает `anneal_reads_total`, `anneal_sweeps_total` и `refinement_states_total`.

## QAOA-in-QAOA (`qaoa_in_qaoa_cpu`, `qaoa_in_qaoa_gpu`)

`QAOAInQAOACPU` и `QAOAInQAOAGPU` принимают `parameters.finish_at`, `parameters.max_levels`, `partitioner`, `local_solver`, `execution`, `budget_policy`, `finisher`, `limits` и `collect`. Блоки текущего уровня не пересекаются и покрывают все его вершины. Внутри блока фиксируется рисунок спинов; anchor каждого блока становится вершиной следующего метаграфа. `GraphReducer` применяет точные ограничения `s_u=r_u s_anchor` ко всему графу.

`local_solver` — один из явных адаптеров:

- `qaoa_cpu` (`QAOACPU`), с `most_probable` или `best_of_shots`;
- `qaoa_gpu` (`QAOAGPU`), только в `qaoa_in_qaoa_gpu`;
- `exact_cpu` — явный тестовый компонент, не квантовый запуск.

Локальный solver не содержит собственного корневого `limits`, `save`, `output` или `seed`. Координатор передаёт ему стабильный seed и фиксированную квоту. `execution.backend="process"` использует общий CPU-пул; `backend="gpu_owner"` обслуживает блоки одним устройством. `best_of_shots` резервирует M каждого readout; `level_round_robin` сначала покрывает все минимумы уровня, затем распределяет полные итерации.

`hierarchy_history` содержит уровень, блоки, anchor, ID задач, raw/normalized рисунки, `Delta b`, `b`, квоты и времена. `hierarchy_tree` использует составные ID `L{level}:A{anchor}`; листья исходных вершин имеют `leaf:{node}`. Углы хранятся по локальным задачам в `parameter_trajectory`; они не усредняются в глобальный вектор.

`limits.max_local_tasks_total` и `parameters.max_levels` дают полный `hierarchy_all_plus_fallback` со статусом `stopped`; ошибки partition, projection, worker или GPU не маскируются как лимит. `finish_at` решается `ExactCPUFinisher`; при `finish_at=0` малый остаток не обещается.

## NVIDIA GPU QAOA (`qaoa_gpu`)

`QAOAGPU` принимает `parameters.layers`, `parameters.max_vertices`, `parameters.dtype` и `parameters.representation` (`full` или `spin_flip_pairs`). `gpu` содержит `device_id`, `diagonal_dtype`, `accumulation_dtype`, `memory.fraction_of_available`, `memory.reserve_bytes`, `kernels.mixer="pairwise"` и `kernels.cuda_graphs=false`.

GPU-путь не импортирует CuPy при импорте пакета. Без CuPy/устройства вызов завершается `RuntimeError`; CPU-солверы не переключаются скрыто на NumPy. Preflight памяти выполняется до создания диагонали и состояния. `state_representation` содержит `type`, `n_vertices`, `node_order`, `dtype`, `amplitude_kind`, `probability_kind`, `representative_rule`, `normalization` и сведения об устройстве.

Сборы в pair-режиме:

- `collect.compressed_statevector=true` — `chi`;
- `collect.pair_probabilities=true` — `q`;
- `collect.statevector=true` или `collect.probabilities=true` отклоняются, пока пользователь не выбрал full-представление. Это не позволяет молча вернуть `chi` как физический `psi`.

`gradient.type="central_difference"` поддерживает все критерии и exact/shots. `gradient.type="adjoint"` поддерживает только exact Mean Cut; для остальных критериев или shots конфигурация отклоняется. Adjoint считает обратный проход по состоянию и не выдаёт себя за `4p+1` измерений.

`QAOAGPUCorrelationSource` (`qaoa_statevector_gpu`) использует собственный `gpu`-раздел и возвращает обычный `CorrelationResult` с исходными номерами и версией задачи. `RQAOAGPU` использует тот же редуктор и финишёр; `LSRQAOAGPU` требует `execution.backend="gpu_owner"` и обслуживает локальные задачи последовательно одним устройством.

## Запуск модулей

В каждом реализованном модуле солвера есть `main(argv=None)` и `if __name__ == "__main__":`. Аргументы CLI: `--graph` (JSON-рецепт), `--config` (JSON солвера), `--report` и `--logs` (0/1/2). Без файлов запускается небольшой воспроизводимый пример. Эти демонстрационные расчёты не выполняются при импорте.


## Эксперименты и профилирование

```python
from QA_max_cut_lib import BenchmarkSpec, ExperimentRunner, ParallelismProfiler
from QA_max_cut_lib import tools

spec = BenchmarkSpec(tools.load_json("experiments/benchmark_small.json"))
plan = spec.plan()
print(plan.summary)
series = ExperimentRunner(spec).run()
profile = ParallelismProfiler(spec, tools.load_json("tools/parallelism_sweep.json")).run()
```

`BenchmarkSpec` принимает JSON-совместимый словарь и проверяет закрытую схему.
`plan()` не вызывает солверы; `ExperimentPlan.result` содержит `counts`,
`graph_instances`, `cases`, `trials`, `reference_jobs` и `rejected`.
`ExperimentRunner.run()` возвращает `ExperimentSeries` с `result`, `trials`,
`references`, `analysis`, `budget`, `environment` и `save_status`.
`ParallelismProfiler.run()` возвращает `ParallelismProfile` с `result`,
`measurements`, `aggregate`, `per_solver`, `recommended_num_treads`,
`analysis["fit_amdahl"]` и `analysis["uncertainty"]`.

`ReferencePolicy` принимает `type`, `exact`, `gw`, `best_known`. Политика
возвращает `ReferenceJob`; `run_job` создаёт независимый reference record.
Точный минимум выполняется отдельным запуском на графе с весами `-w`.
GW-запись имеет `proof="heuristic"`; `sdp_upper_bound` не выдумывается.

CLI:

```bash
python -m QA_max_cut_lib.tools.experiments plan --config experiments/benchmark_small.json
python -m QA_max_cut_lib.tools.experiments run --config experiments/benchmark_small.json
python -m QA_max_cut_lib.tools.parallelism --benchmark experiments/benchmark_small.json --config tools/parallelism_sweep.json
```


## Контрольные точки

`QAOACPU.solve(..., control=RunControl(pause_after={"optimizer_iterations": 2}))`
возвращает полное anytime-назначение и `last_checkpoint`. `tools.inspect_checkpoint`
читает manifest без solver. `restore(path)` создаёт объект; `resume()` продолжает
тот же canonical run. Новый `solve()` на восстановленном объекте — новый run.
CLI: `python -m QA_max_cut_lib.tools.checkpoints inspect --checkpoint PATH`.
Полная матрица границ — в `CHECKPOINTS_RU.md`.


## Laboratory и warm-start

```python
from QA_max_cut_lib import GraphFactory, tools
from QA_max_cut_lib.laboratory import WarmStartQAOACPU

graph = GraphFactory.from_recipe(tools.load_json("configs/graphs/warm_start_4.json"))
config = tools.load_json("configs/solvers/warm_start_qaoa_cpu_provided.json")
solver = WarmStartQAOACPU(config)
solution = solver.solve(graph)
```

`type="laboratory.warm_start_qaoa_cpu"`. Обязательный раздел `warm_start`:

```json
{
  "epsilon": 0.25,
  "source": {"type": "provided_spins", "parameters": {"entries": [{"node": 10, "spin": -1}]}},
  "mixer": {"type": "rounded_modified"}
}
```

Источники: `provided_spins`, `provided_probabilities`, `classical_solver`
(`gw_cpu`, `simulated_annealing_cpu`, `exact_cpu`). Mixer: `x`,
`warm_aligned`, `rounded_modified`. `epsilon` фиксирован на run.
Source entries должны покрывать все метки текущего графа без повторов.
Classical child не запускается в конструкторе; его `source_cost` отдельно.

`result` дополнительно содержит `variant_id`, `warm_start_data`
(`q_raw`, `q_clipped`, `warm_start_angles`, `node_order`), `mixer_id`,
`source_type`, `source_cut`, `source_run`, `source_cost`,
`initial_expected_cut`, `initial_expected_energy`. `source_cut` — дискретный
seed cut, а не ожидание product state. `warm_start_angles` не входят в
`optimal_angles` и не являются trainable параметрами.

Каталог:

```bash
python -m QA_max_cut_lib.laboratory list
python -m QA_max_cut_lib.laboratory show --id warm_start_product_qaoa_cpu_v1
```

`list/show` не требуют CuPy/CVXPY. Полный обзор — `LABORATORY_RU.md`.
