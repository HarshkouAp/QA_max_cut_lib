# QA_max_cut_lib

Научная библиотека для взвешенного Max-Cut. Версия **0.1.0a1 — первый рабочий CPU-этап**: изменяемый граф, общий жизненный цикл солвера, полный результат, инструменты анализа, визуализация и безопасное сохранение.

Это ещё не перенос всего исследовательского архива. Сейчас доступны `ExactCPU` и `QAOACPU`; LS, RQAOA, QAOA-in-QAOA, GPU, Laboratory, серии экспериментов, профилировщик потоков и контрольные точки перечислены как следующие этапы в [статусе переноса](docs/MIGRATION_STATUS.md).

## Установка

Python 3.10 или новее. Проверенная среда этой сборки — Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[cpu,notebooks]"
```

В Windows активация окружения: `.venv\Scripts\activate`.

В этом рабочем проекте окружение уже подготовлено: `.venv` на Python 3.12.14,
полный CPU-профиль, ноутбуки и Sphinx. Подробности, версии пакетов и команды
проверки — в [описании проекта](docs/PROJECT_SETUP_RU.md). Локальная проверка
Windows — в [VALIDATION_LOCAL.md](docs/VALIDATION_LOCAL.md).

CPU-профиль не устанавливает и не импортирует CUDA или CuPy. NVIDIA-зависимости вынесены в отдельные файлы `requirements/gpu-nvidia-cu12.in` и `requirements/gpu-nvidia-cu13.in`; они подготовлены для будущего GPU-этапа и не добавляют работающий GPU-солвер в эту версию. См. [описание зависимостей](docs/INSTALLATION.md).

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
- [Математика](docs/MATHEMATICS.md): энергетическая нотация, QAOA, критерии и численные ограничения.
- [Полный API текущей версии](docs/API.md): аргументы, конфигурационные поля, атрибуты, формы массивов и ошибки.
- [Согласованная архитектура](docs/reference/MaxCut_Library_Architecture_RU.md): целевая библиотека целиком, включая ещё не реализованные требования.
- [Проверка сборки](docs/VALIDATION.md): выполненные проверки и пределы их применимости.

Исполняемые примеры модулей:

```bash
python -m QA_max_cut_lib.solvers.exact_cpu --report 1 --logs 0
python -m QA_max_cut_lib.solvers.qaoa_cpu --graph configs/graphs/complete_signed_7.json --config configs/solvers/qaoa_cpu.json --report 1
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
