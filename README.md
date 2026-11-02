# QA_max_cut_lib

Научная библиотека для взвешенного Max-Cut. Версия **0.1.0a18**: изменяемый граф, общий жизненный цикл солвера, полный результат, инструменты анализа, визуализация, безопасное сохранение, точное ядро парных редукций Max-Cut → Изинг, CPU RQAOA, аналитический CPU-вычислитель моментов p=1, LS-RQAOA с процессами, NVIDIA CUDA QAOA/семплер, иерархический QAOA-in-QAOA, классические методы GW/SA с точным улучшением блоков, общий `ExperimentRunner`, `ReferencePolicy` и `ParallelismProfiler`, а также checkpoint/resume `QAOACPU`, версионированный Laboratory-каталог и первый warm-start CPU-перенос (`WarmStartQAOACPU`) warm-start RQAOA с ансамблем стартов (`WarmStartRQAOACPU`), а также условный QAOA на фиксированных внешних спинах (`ConditionalQAOACPU`), environment LS-RQAOA (`EnvironmentLSRQAOACPU`), блочный mean-field QAOA (`BlockMeanFieldQAOACPU`) и distributed LS-RQAOA (`DistributedLSRQAOACPU`) пассивные MI-наблюдатели (`QAOAWithMIObserverCPU`, `RQAOAWithMIObserverCPU`) и incremental LS-RQAOA с банком корреляций (`IncrementalLSRQAOACPU`), расширенные генераторы, Wishart/Hebbian-конструкции и typed plans покрытия (`SubproblemPlan`).

Это ещё не перенос всего исследовательского архива. Сейчас доступны CPU-солверы, `QAOAGPU`, `RQAOAGPU`, `LSRQAOAGPU`, `QAOAInQAOACPU`, `QAOAInQAOAGPU`, `GoemansWilliamsonCPU`, `SimulatedAnnealingCPU`, `ExactBlockRefiner`, `RefinedSolver`, два построителя подграфов и `GraphReducer`. Классические эталоны GW/SA работают на CPU; CVXPY/SCS и dimod/dwave-samplers являются отдельным профилем `classical`. Laboratory и control-point resume остаются следующими этапами; общие серии экспериментов и `ParallelismProfiler` реализованы в `0.1.0a9`.

## Checkpoint/resume

```python
from QA_max_cut_lib import RunControl, restore

solver = tools.SolverFactory.from_config(config)
solver.solve(graph, control=RunControl(pause_after={"optimizer_iterations": 2}))
path = solver.last_checkpoint.path
continued = restore(path)
final = continued.resume()
```

Подробности и capability matrix — в [руководстве](docs/CHECKPOINTS_RU.md).

## Установка

Python 3.10 или новее. Проверенная среда этой сборки — Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[cpu,classical,notebooks]"
```

В Windows активация окружения: `.venv\Scripts\activate`.

В этом рабочем проекте окружение уже подготовлено: `.venv-stage02` на Python
3.12.14 с CPU-профилем, ноутбуками и Sphinx; `.venv` с CuPy сохранено отдельно
для будущих GPU-этапов. Для этапа 03 то же CPU-окружение прошло полный набор
проверок; чистая установка wheel проверена в отдельном `.venv-wheelcheck`.
Подробности, версии пакетов и команды проверки — в
[описании проекта](docs/PROJECT_SETUP_RU.md). Локальная проверка Windows — в
[VALIDATION_LOCAL.md](docs/VALIDATION_LOCAL.md), отчёт этапа 04 — в
[VALIDATION_0.1.0a4.md](docs/VALIDATION_0.1.0a4.md), отчёт этапа 05 — в
[VALIDATION_0.1.0a5.md](docs/VALIDATION_0.1.0a5.md), отчёт этапа 06 — в
[VALIDATION_0.1.0a6.md](docs/VALIDATION_0.1.0a6.md), отчёт этапа 07 — в
[VALIDATION_0.1.0a7.md](docs/VALIDATION_0.1.0a7.md), отчёт этапа 08 — в
[VALIDATION_0.1.0a8.md](docs/VALIDATION_0.1.0a8.md), отчёт этапа 09 — в
[VALIDATION_0.1.0a9.md](docs/VALIDATION_0.1.0a9.md), отчёт этапа 10 — в
[VALIDATION_0.1.0a10.md](docs/VALIDATION_0.1.0a10.md).

CPU-профиль не устанавливает и не импортирует CUDA или CuPy. NVIDIA-зависимости вынесены в отдельные extras `gpu-cu12`/`gpu-cu13` и файлы `requirements/gpu-nvidia-cu12.in`, `requirements/gpu-nvidia-cu13.in`; GPU-путь проверен с `cupy-cuda13x 14.2.0`. См. [описание зависимостей](docs/INSTALLATION.md) и [GPU-руководство](docs/GPU.md).

## Первый расчёт

```python
from QA_max_cut_lib import WeightedGraph, ExactCPU, QAOACPU
from QA_max_cut_lib import tools

graph = WeightedGraph(
    edges=[(10, 30, 1.5), (30, 80, -0.5), (10, 80, 2.0)],
    nodes=[10, 30, 80, 101],  # Include an isolated node.
)
reference = ExactCPU()
reference.solve(graph)

algorithm = QAOACPU({
    "seed": 42,
    "optimizer": {"parameters": {"max_steps": 30}},
    "readout": {"mode": "best_of_shots", "shots": 256},
    "collect": {"optimizer_history": True, "probabilities": True},
})
solution = algorithm.solve(graph, report=1, logs=0)
print(solution)                   # dict: original integer node -> +1/-1
print(algorithm.solution_energy) # E = -C; lower is better.
record = algorithm.last_run      # Independent record of this run.
json_data = algorithm.result     # Complete JSON-compatible dict.

figure = tools.plot_graph_solution(algorithm)
probabilities = tools.plot_probabilities(algorithm)
optimization = tools.plot_optimization(algorithm)
```

Один экземпляр солвера принимает разные графы. Каждый `solve()` начинает новый запуск с заново установленным `seed`. В солвере остаётся последний запуск; ранее сохранённый `last_run` не меняется. Инструментам можно передавать солвер, `RunRecord` или полный `algorithm.result`. Для графов также допустим непосредственно `WeightedGraph`.

## Серии экспериментов

```python
from QA_max_cut_lib import tools

spec = tools.load_json("experiments/benchmark_small.json")
plan = tools.plan_experiments(spec)
print(plan.summary)
series = tools.ExperimentRunner(spec).run()
print(series.analysis["by_solver"])

profile = tools.profile_parallelism(
    spec, config=tools.load_json("tools/parallelism_sweep.json")
)
print(profile.recommended_num_treads)
print(profile.analysis["uncertainty"])
```

Подробности — в [руководстве по экспериментам](docs/EXPERIMENTS_RU.md).

## Точная редукция

Ядро принимает явное ограничение `s_u = σ s_v` и сохраняет равенство энергии на
всём подпространстве ограничения. Оно не выбирает корреляцию и пару.

```python
from QA_max_cut_lib import GraphReducer, WeightedGraph

graph = WeightedGraph(
    edges=[(10, 30, 2.0), (10, 80, 3.0), (30, 80, 5.0)],
    nodes=[10, 30, 80, 101],
)
reducer = GraphReducer()
state = reducer.start(graph)
event = reducer.reduce(state, removed=10, retained=30, sign=-1,
                       correlation=-0.6)

print(event["fixed_energy"])            # -2.0
print(state.cut_offset)                 # 5.0
print(state.ising.constant)             # -6.0
working = state.current_graph()
solution = reducer.restore(state, {30: 1, 80: -1, 101: 1})
print(state.original_energy(solution))  # -7.0
```

`state.snapshot()` сохраняет рабочую задачу, исходные номера, ограничения,
сдвиг и трассу в JSON-совместимый словарь. Это математический снимок, а не
checkpoint оптимизатора. Подробные поля и ошибки — в [API](docs/API.md), вывод
формул — в [математике](docs/MATHEMATICS.md).

## RQAOA на CPU

`RQAOACPU` обучает QAOA на текущем графе, получает физические ZZ-корреляции,
выбирает одну пару, применяет точную редукцию и повторяет цикл до `finish_at`.
Остаток решает `ExactCPU`; исходные спины восстанавливаются ядром этапа 02.
Алгоритм эвристический: точное решение остатка не доказывает глобальный оптимум
исходного графа.

```python
from QA_max_cut_lib import SolverFactory, tools

config = tools.load_json("configs/solvers/rqaoa_cpu.json")
algorithm = SolverFactory.from_config(config)
solution = algorithm.solve(graph)
result = algorithm.result
trajectory = algorithm.parameter_trajectory
history = algorithm.reduction_history

figure = tools.plot_reduction_energy(algorithm, {"style": {"text_engine": "mathtext"}})
trajectory_figure = tools.plot_parameter_trajectories(algorithm)
interactive = tools.plot_parameter_trajectories_3d(algorithm)
```

Полный вектор состояния ограничивает размер исходного графа. Для этой версии
источник рассчитан на небольшие задачи, а не на ориентир в 1000 вершин.
Шотовый пример находится в `configs/solvers/rqaoa_cpu_shots.json`; общий лимит
`limits.shots_total` соблюдается строго, а при остановке возвращается полное
допустимое решение с явным `status="stopped"`.

## Аналитический p=1

`AnalyticP1Evaluator` вычисляет среднюю энергию и физические ZZ-корреляции для
стандартного `|+>`, X-миксера и одного слоя без вектора `2**N`. Источник
`qaoa_analytic_p1_cpu` перебирает конечную сетку gamma и точно исключает beta:

```python
from QA_max_cut_lib import AnalyticP1Evaluator, SolverFactory, tools

evaluator = AnalyticP1Evaluator(graph)
print(evaluator.correlations(0.3, 0.2, pair_domain="all_pairs"))
print(evaluator.energy_min_beta(0.3))

config = tools.load_json("configs/solvers/rqaoa_analytic_p1_cpu.json")
config.pop("save", None)
algorithm = SolverFactory.from_config(config)
solution = algorithm.solve(graph)       # the same RQAOACPU coordinator.
```

Поддерживаются Mean Cut и точные корреляции для существующих или всех пар,
включая несмежные. Шоты, полное распределение, CVaR, Gibbs, Best-of-K,
Threshold Probability, поля и `p>1` этим источником не поддерживаются; такие
запросы отклоняются до вычисления. Аналитический запуск использует
`shots_used=0`; `limits.shots_total=0` допустим. Для графа из 30 вершин
ноутбук `05_analytic_p1_rqaoa.ipynb` выполняет короткий полный RQAOA без
объекта `2**30`. Ориентир 1000 вершин относится к подготовке/пакетному
вычислению, а не к обещанию полной цепочки RQAOA.

Проверка формул и ограничений находится в `tests/test_analytic_p1.py`;
локальный профиль времени и памяти — в
[PERFORMANCE_ANALYTIC_P1_0.1.0a4.txt](docs/PERFORMANCE_ANALYTIC_P1_0.1.0a4.txt).

## NVIDIA GPU QAOA

`QAOAGPU` (`qaoa_gpu`) реализует полный и сокращённый режимы:

- `parameters.representation="full"` — физический вектор длины `2**N`;
- `parameters.representation="spin_flip_pairs"` — нормированные пары
  `chi_r = sqrt(2) * psi_r`, представитель с последним битом `node_order`
  равным нулю; `q_r=P(r)+P(bar r)`, а отдельная физическая вероятность
  равна `q_r/2`.

GPU-путь использует CuPy, точную диагональ `C`, прямой mixer и все пять
критериев: Mean Cut, CVaR, Gibbs, Best-of-K и Threshold Probability. Семплер
выполняется на устройстве и возвращает обе ориентации пары. Для exact Mean Cut
доступен `gradient.type="adjoint"`; остальные критерии используют центральные
разности. `gpu.memory` задаёт preflight бюджета VRAM, а `max_vertices` —
пользовательский предел, не обещание размещения.

```bash
python -m QA_max_cut_lib.solvers.qaoa_gpu --help
python -m QA_max_cut_lib.solvers.rqaoa_gpu --graph configs/graphs/complete_signed_7.json --config configs/solvers/rqaoa_gpu.json --report 1
python -m QA_max_cut_lib.solvers.ls_rqaoa_gpu --graph configs/graphs/complete_signed_7.json --config configs/solvers/ls_rqaoa_gpu.json --report 1
```

CPU-импорт, CPU-солверы, сохранённые GPU-результаты и их отрисовка не требуют
CuPy. `QAOAGPUCorrelationSource` (`qaoa_statevector_gpu`) подключается к
`RQAOAGPU`; `LSRQAOAGPU` использует одну последовательность обслуживания
`gpu_owner`, а не CUDA-контекст в каждом worker. Подробности — в
[GPU-руководстве](docs/GPU.md).

## Классические базовые методы и улучшение

`GoemansWilliamsonCPU` (`gw_cpu`) решает SDP-релаксацию через CVXPY/SCS и
округляет гиперплоскостями. `SimulatedAnnealingCPU`
(`simulated_annealing_cpu`) использует классический
`dwave.samplers.SimulatedAnnealingSampler` с явными reads, sweeps и seed-планом.
`ExactBlockRefiner` (`exact_block`) точно переоптимизирует малые блоки при
фиксированном внешнем окружении, а `RefinedSolver` (`refined`) запускает
базовый солвер и сохраняет независимый `base_run`.

```bash
python -m QA_max_cut_lib.solvers.goemans_williamson_cpu \
  --graph configs/graphs/erdos_signed_32.json --config configs/solvers/gw_cpu.json
python -m QA_max_cut_lib.solvers.simulated_annealing_cpu \
  --graph configs/graphs/erdos_signed_128.json --config configs/solvers/simulated_annealing_cpu.json
python -m QA_max_cut_lib.solvers.refined \
  --graph configs/graphs/erdos_signed_64.json --config configs/solvers/refined_sa.json
```

Классические методы используют `E=-C`, полный словарь спинов и не расходуют
квантовые шоты. `num_reads`/sweeps — не quantum shots. Для установки
дополнительных зависимостей: `pip install -e ".[cpu,classical]"`.

## QAOA-in-QAOA

`QAOAInQAOACPU` (`qaoa_in_qaoa_cpu`) и `QAOAInQAOAGPU` (`qaoa_in_qaoa_gpu`)
реализуют базовую иерархию: локальный рисунок спинов блока, одна свободная
ориентация блока, точная метазадача и восстановление исходных вершин. Это не
LS-RQAOA и не выбор одной пары по корреляции.

```bash
python -m QA_max_cut_lib.solvers.qaoa_in_qaoa_cpu \
  --graph configs/graphs/erdos_signed_64.json \
  --config configs/solvers/qaoa_in_qaoa_cpu.json --report 1
python -m QA_max_cut_lib.solvers.qaoa_in_qaoa_gpu \
  --graph configs/graphs/complete_signed_7.json \
  --config configs/solvers/qaoa_in_qaoa_gpu.json --report 1
```

Используются готовые `qaoa_cpu`/`qaoa_gpu`/`exact_cpu` локальные адаптеры,
`GraphReducer` для `s_u=r_u s_anchor`, общий `ShotBudget` и `level_round_robin`
квоты. `best_of_shots` резервирует реальный readout каждого блока. GPU-вариант
имеет одного владельца устройства. См. [API](docs/API.md) и
[математику](docs/MATHEMATICS.md).

## LS-RQAOA на CPU

`LSRQAOACPU` (тип `ls_rqaoa_cpu`) строит несколько индуцированных подграфов
текущего графа, решает их независимо, усредняет физические корреляции по
наблюдениям каждой пары и применяет **одно** ограничение на полном графе.
Граничные рёбра в локальную задачу не входят. Это явная эвристика
`induced_no_fields`, а не точное разложение полного гамильтониана.

Доступны построители `random_partition` и `greedy_abs_weight_partition`.
При `num_treads>1` независимые локальные задачи выполняются в пуле процессов
со `spawn`; при `num_treads=1` тот же план выполняется последовательно.
Число фактически применённых процессов не превышает запрошенное значение,
доступные CPU и число задач и сохраняется отдельно от запроса.

```python
from QA_max_cut_lib import LSRQAOACPU, tools
from QA_max_cut_lib.generators.graphs import GraphFactory

graph = GraphFactory.from_recipe(tools.load_json(
    "configs/graphs/erdos_signed_30.json"
))
config = tools.load_json("configs/solvers/ls_rqaoa_analytic.json")
config.pop("save", None)
config["num_treads"] = 2
algorithm = LSRQAOACPU(config)
solution = algorithm.solve(graph)          # explicit argument overrides JSON too.
print(algorithm.resources)
print(algorithm.coverage_history)
```

Аналитический p=1 работает при `limits.shots_total=0`. Шотовый
statevector-пример находится в `configs/solvers/ls_rqaoa_shots.json`: бюджеты
резервируются координатором до запуска задач, поэтому копия `ShotBudget` в
каждом процессе не может обойти общий лимит. Остановка по лимиту возвращает
полное допустимое решение с `status="stopped"`.

`parameter_trajectory` содержит отдельные строки локальных задач; это не
средняя глобальная траектория. `plot_parameter_trajectories` и
`plot_parameter_trajectories_3d` показывают их по `reduction_step` и `task_id`;
`plot_ls_coverage` показывает разрешённые, наблюдаемые и пропущенные пары.

## Сохранение

Без `save` автоматической записи нет. Имена и параметры задаются через JSON-совместимый словарь.

```python
files = tools.save_result(algorithm, {
    "path": "results",
    "filename": "qaoa_run_001",
    "atts": ["solution", "cut_value", "solution_energy", "optimal_angles"],
    "save_original_data": ["probabilities"],
    "makedir": True,
    "savedir": True,
})
restored = tools.load_result(files.primary, load_original_data=True)

figure = tools.plot_probabilities(restored, {
    "selection": {"top_k": 16},
    "style": {"text_engine": "mathtext", "figsize": [10, 6]},
    "save": {
        "path": "results", "filename": "probabilities_001",
        "format": "png", "DPI": 250, "metadata": True,
        "makedir": True, "savedir": True,
    },
})
```

Получатся `results/qaoa_run_001/qaoa_run_001.json`, отдельный `.npy` и изображение в папке `probabilities_001`. Повторная запись с тем же именем вызывает `FileExistsError`; прежние данные сохраняются. Формат массива — **NPY**, расширение `.npy`.

Оформление использует локальную тему Seaborn и Computer Modern. По умолчанию работает MathText; `style.text_engine="latex"` включает настоящий LaTeX и требует исправной системной TeX-установки. Переключение движка не происходит скрыто.

## Обучение и справочник

- [Быстрый старт](notebooks/01_quickstart.ipynb): запуск, конфигурация, графики, сохранение и загрузка.
- [Подробное руководство текущей версии](notebooks/02_current_api_guide.ipynb): все реализованные компоненты, пять критериев, шоты, повторные запуски, карты и экспорт.
- [Ядро редукций](notebooks/03_reduction_core.ipynb): контрольный пример, цепочка, восстановление и снимок состояния.
- [RQAOA на CPU](notebooks/04_rqaoa_cpu.ipynb): обучение, корреляции, редукции, восстановление, графики, сохранение и остановка по бюджету.
- [Аналитический p=1 и RQAOA](notebooks/05_analytic_p1_rqaoa.ipynb): формула, точная beta-минимизация, профиль gamma, подключение к RQAOA и задача из 30 вершин.
- [LS-RQAOA на CPU](notebooks/06_ls_rqaoa_cpu.ipynb): локальные задачи, два построителя, последовательный и параллельный запуск, покрытие, энергии, локальные углы, бюджет и сохранение.
- [Математика](docs/MATHEMATICS.md): энергетическая нотация, QAOA, критерии и численные ограничения.
- [Полный API текущей версии](docs/API.md): аргументы, конфигурационные поля, атрибуты, формы массивов и ошибки.
- [Карта архитектуры и состояния переноса](docs/ARCHITECTURE_MAP_RU.md): что уже есть, чего нет и какие контракты вводить.
- [Рабочее окружение](docs/PROJECT_SETUP_RU.md): версии, команды, GPU и правила этапов.
- [Локальная проверка Windows](docs/VALIDATION_LOCAL.md): выполненные прогоны.
- [Согласованная архитектура](docs/reference/MaxCut_Library_Architecture_RU.md): целевая библиотека целиком, включая ещё не реализованные требования.
- [Проверка исходной сборки](docs/VALIDATION.md): выполненные проверки и пределы их применимости.

Исполняемые примеры модулей:

```bash
python -m QA_max_cut_lib.solvers.exact_cpu --report 1 --logs 0
python -m QA_max_cut_lib.solvers.qaoa_cpu --graph configs/graphs/complete_signed_7.json --config configs/solvers/qaoa_cpu.json --report 1
python -m QA_max_cut_lib.solvers.rqaoa_cpu --graph configs/graphs/complete_signed_7.json --config configs/solvers/rqaoa_cpu.json --report 1
python -m QA_max_cut_lib.solvers.rqaoa_cpu --graph configs/graphs/complete_signed_7.json --config configs/solvers/rqaoa_analytic_p1_cpu.json --report 1
python -m QA_max_cut_lib.solvers.ls_rqaoa_cpu --graph configs/graphs/erdos_signed_30.json --config configs/solvers/ls_rqaoa_analytic.json --report 1
python -m QA_max_cut_lib.solvers.ls_rqaoa_cpu --graph configs/graphs/erdos_signed_30.json --config configs/solvers/ls_rqaoa_shots.json --report 1
python examples/quickstart.py
```

## Разработка

```bash
python -m pip install -e ".[cpu,notebooks,dev,docs]"
python -m pytest -q
python -m build
python -m sphinx -W -b html docs docs/_build/html
```

Комментарии и docstrings написаны простым техническим английским с объяснением преобразований. Формальное подтверждение соответствия ASD-STE100 в эту сборку не входит. Публичная лицензия распространения пока не назначена; сборка не публиковалась в PyPI.
