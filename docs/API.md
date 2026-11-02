# API версии 0.1.0a2

Все конфигурации ниже — обычные JSON-совместимые Python `dict`; файл читается через `tools.load_json(path)`. `solve()` возвращает словарь решения. `algorithm.result` возвращает JSON-совместимый словарь, а не текстовую строку. Для текстового JSON применяется стандартный `json.dumps`.

## Публичные импорты и жизненный цикл

```python
from QA_max_cut_lib import (
    WeightedGraph, GraphFactory, SolverFactory, RunRecord, ExactCPU, QAOACPU,
    GraphReducer, ReductionState, IsingProblem, SpinConstraint,
)
from QA_max_cut_lib import tools
from QA_max_cut_lib.generators import partition
from QA_max_cut_lib.solvers.objectives import validate_objective, objective_value
from QA_max_cut_lib._version import __version__
```

`SolverFactory.available()` возвращает `("exact_cpu", "qaoa_cpu")`. `from_config(config)` выбирает класс по `config["type"]`; неизвестный тип — `ValueError`. Фабрика не выполняет расчёт.

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

Интерактивный HTML, трёхмерные поверхности и траектории редукции в текущем этапе не реализованы; запрос формата html вызывает понятную ошибку.

## Запуск модулей

В каждом реализованном модуле солвера есть `main(argv=None)` и `if __name__ == "__main__":`. Аргументы CLI: `--graph` (JSON-рецепт), `--config` (JSON солвера), `--report` и `--logs` (0/1/2). Без файлов запускается небольшой воспроизводимый пример. Эти демонстрационные расчёты не выполняются при импорте.
