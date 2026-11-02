# Проект QA_max_cut_lib — рабочее окружение

**Дата подготовки:** 2 ноября 2026 года.
**Каталог проекта:** `C:\Users\Alexey\PycharmProjects\QA_max_cut_lib`.
**Стартовая сборка:** `QA_max_cut_lib 0.1.0a1` из архива `Downloads/QA_max_cut_lib_0.1.0a1.zip`.
**Текущая версия проекта:** `0.1.0a4` (добавлен аналитический источник p=1).

## 1. Что подготовлено

1. Новый каталог проекта с `src`-раскладкой пакета `QA_max_cut_lib`.
2. Отдельное виртуальное окружение `.venv` на Python 3.12.14.
3. Полный CPU-профиль зависимостей и отдельные файлы для NVIDIA CUDA 12/13.
4. Инструменты ноутбуков и сборки документации Sphinx.
5. Исходный исследовательский архив `RQAOA_main` в `reference/RQAOA_main` — только для чтения и сверки при переносе.
6. Согласованная архитектура в `docs/reference/MaxCut_Library_Architecture_RU.md`.
7. Точное ядро парных редукций Max-Cut → Изинг в `core/reduction.py`.
8. Контрольная проверка: 56 автоматических тестов текущей сборки проходят на этой машине.
9. Git-репозиторий с начальным коммитом.

Сборка `0.1.0a1` — это **стартовая точка**, а не весь перенос. В a2 к ней
добавлено точное ядро парных редукций; в a1 были реализованы
`WeightedGraph`, `ExactCPU`, `QAOACPU`, общий жизненный цикл запуска, JSON-результат,
критерии обучения, сохранение и базовые инструменты. Остальные семейства
(RQAOA, LS-RQAOA, QAOA-in-QAOA, GPU, Laboratory, серии, профилировщик, контрольные
точки, интерактивная графика) будут добавляться по инструкциям этапов 1–10.

## 2. Структура проекта

```text
QA_max_cut_lib/
├── src/QA_max_cut_lib/
│   ├── core/           # граф, записи результата, сохранение, фабрика
│   ├── solvers/        # базовый контракт, ExactCPU, QAOACPU, критерии
│   ├── generators/     # рецепты графов и разбиения
│   ├── tools/          # численные функции, графики, отчёты, демо, ландшафты
│   └── laboratory/     # зарезервировано под экспериментальные сборки
├── tests/              # 42 проверки исходной сборки
├── configs/            # примеры JSON-конфигураций графов, солверов и графиков
├── examples/           # исполняемый quickstart
├── notebooks/          # 01_quickstart.ipynb и 02_current_api_guide.ipynb
├── docs/               # API, математика, статус переноса, локальная проверка
├── reference/RQAOA_main/  # исходный архив (89 файлов), только для сверки
├── requirements/       # профили CPU/GPU, ноутбуков, документации и lock-файл
├── .venv/              # виртуальное окружение (не коммитится)
└── pyproject.toml
```

## 3. Окружение

| Параметр | Значение |
|---|---|
| ОС | Windows 10, 10.0.19043 |
| Python | 3.12.14 (uv-managed CPython) |
| Окружение | `C:\Users\Alexey\PycharmProjects\QA_max_cut_lib\.venv` |
| NumPy / SciPy | 2.5.4 / 1.18.1 |
| Matplotlib / Seaborn / Plotly | 3.11.2 / 0.13.2 / 7.1.0 |
| CPU | 4 физических ядра, 8 логических |
| RAM | 15.7 ГБ |
| GPU | NVIDIA GeForce RTX 3050 Laptop, 4 ГБ |
| Драйвер NVIDIA | 591.74 |
| CUDA Toolkit | 13.0 (V13.0.48) |
| CuPy | 14.2.0 (`cupy-cuda13x`, системный CUDA, без `ctk`) |
| TeX | MiKTeX 25.12, pdfTeX 4.23 |
| Свободно на диске C: | около 19 ГБ на момент подготовки |

Для этапа 02 создано отдельное CPU-окружение `.venv-stage02` (без CuPy).
Основное `.venv` с CuPy 14.2.0 сохранено отдельно для будущих GPU-этапов.

Точный список установленных версий: `requirements/freeze-local-py312-win.txt`.
Разрешённые lock-файлы профилей: `requirements/lock-cpu-py312-win.txt` и
`requirements/lock-gpu-cu13-py312-win.txt`.

### Активация

PowerShell:

```powershell
cd C:\Users\Alexey\PycharmProjects\QA_max_cut_lib
.\.venv\Scripts\Activate.ps1
```

Git Bash:

```bash
cd /c/Users/Alexey/PycharmProjects/QA_max_cut_lib
source .venv/Scripts/activate
```

В PyCharm: **File → Open** → каталог проекта; интерпретатор
`.venv\Scripts\python.exe`.

Для JupyterLab зарегистрировано пользовательское ядро
`Python 3.12 (QA_max_cut_lib)`. Запуск:

```bash
.venv/Scripts/jupyter.exe lab notebooks/01_quickstart.ipynb
```

Ядро указывает на `.venv` этого проекта. Если проект переносится в другой
каталог, ядро нужно зарегистрировать заново.

### Установка с нуля

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[cpu,notebooks,dev,docs]"
```

Профиль GPU устанавливается отдельно и ровно один:

```bash
uv pip install --python .venv/Scripts/python.exe -e ".[cpu,gpu-cu13]"
```

Для CUDA 12 используется `gpu-cu12` вместо `gpu-cu13`.

## 4. Проверка

```bash
cd C:/Users/Alexey/PycharmProjects/QA_max_cut_lib
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/check_environment.py
```

Ожидаемый результат тестов: `42 passed`.

Демонстрационные запуски модулей:

```bash
.venv/Scripts/python.exe -m QA_max_cut_lib.solvers.exact_cpu --report 1 --logs 0
.venv/Scripts/python.exe -m QA_max_cut_lib.solvers.qaoa_cpu --report 1 --logs 0
.venv/Scripts/python.exe examples/quickstart.py
```

## 5. GPU-профиль

CuPy установлен в рабочее окружение: `cupy-cuda13x 14.2.0`, использующий
системный CUDA Toolkit 13.0. Вариант `ctk` не применялся, потому что системный
CUDA уже установлен и совместим с драйвером 591.74.

Проверено:

- `cupy.cuda.runtime.runtimeGetVersion()` → 13020;
- `driverGetVersion()` → 13010;
- устройство доступно, свободно около 3.3 ГБ из 4 ГБ;
- элементwise-операции и компиляция `RawKernel` через NVRTC работают.

Ограничение: 4 ГБ видеопамяти. Для GPU-этапа размер локальной задачи нужно
выбирать по фактической памяти и проверять до запуска. Профиль CuPy без `ctk`
описан в архитектуре как допустимый при явно проверенной совместимости
системного CUDA.

Файлы `requirements/gpu-nvidia-cu12.in` и `requirements/gpu-nvidia-cu13.in`
подготовлены для чистых установок с `ctk`. Для повторения локальной установки
без `ctk` достаточно `cupy-cuda13x` совместимой версии.

## 6. Правила работы по этапам

1. Каждый этап выполняется в этом проекте; исходный архив не изменяется.
2. Перед этапом рабочее дерево чистое, тесты зелёные.
3. Новые возможности не заменяются заглушками с успешным статусом.
4. После этапа: тесты, демонстрационный запуск, обновление документации и
   отдельный git-коммит.
5. Численные результаты и графики сохраняются в `results/` и не коммитятся,
   если это не эталонный пример.
6. Имена и контракты сверяются с архитектурой; отклонения фиксируются явно.

## 7. Известные ограничения стартовой сборки

Они не скрыты и не выдаются за готовые возможности:

- нет RQAOA, LS-RQAOA, QAOA-in-QAOA, GW и GPU-солверов;
- нет `RunState`, контрольных точек и продолжения запуска;
- нет общего `ExperimentRunner` и `ParallelismProfiler`;
- `num_treads` принимает только 1;
- нет интерактивного HTML и трёхмерных представлений;
- Laboratory пуста;
- `docs/VALIDATION.md` описывает прежнюю Linux-сборку; локальная проверка
  Windows находится в `docs/VALIDATION_LOCAL.md`.

Эти ограничения соответствуют состоянию `0.1.0a1`; версия `0.1.0a2` добавляет
ядро редукций, `0.1.0a3` — базовый RQAOA на CPU, `0.1.0a4` — аналитический
источник p=1, и не претендует на остальные пункты.
