# Среды запуска

## CPU

Минимальное вычислительное ядро требует `numpy>=1.26,<3` и `scipy>=1.11`.
Набор `[cpu]` дополнительно устанавливает NetworkX, pandas, jsonschema, Rich,
Matplotlib, Seaborn, Plotly, joblib, threadpoolctl, psutil и python-louvain.
Классические GW/SA и точный refiner требуют отдельного профиля `classical`.
Основной импорт не загружает графическую подсистему, CuPy, CVXPY или dimod.

Из корня проекта:

```bash
python -m pip install -e ".[cpu]"
```

Классический профиль:

```bash
python -m pip install -e ".[cpu,classical]"
```

Он добавляет CVXPY, SCS, dimod и dwave-samplers. Отсутствие этих пакетов
обнаруживается только при выборе `gw_cpu`/`simulated_annealing_cpu`; оно не
ломает импорт пакета и ExactCPU/QAOACPU. CVXPY/SCS используются только на
CPU.

В этом рабочем проекте используется `uv`:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[cpu,notebooks,dev,docs]"
```

Для Jupyter добавить `[notebooks]`, для Sphinx — `[docs]`. Для GW/SA и
серий с классическими эталонами нужен профиль `[classical]`. Исходные списки
находятся в `requirements/cpu.in`, `requirements/notebooks.in`,
`requirements/dev.in` и `requirements/docs.in`. Точные версии локальной
Windows-сборки — в `requirements/lock-cpu-py312-win.txt`.

## NVIDIA

GPU-солверы этапов 06–07 реализованы отдельно и требуют один CuPy-профиль:

```bash
python -m pip install -e ".[cpu,gpu-cu13]"
```

Альтернативный профиль для CUDA 12 — `[gpu-cu12]`. Исходные списки:
`requirements/gpu-nvidia-cu13.in` и `requirements/gpu-nvidia-cu12.in`.
Следует выбрать **один** профиль CuPy, совместимый с установленным драйвером и
устройством. Профили используют `cupy-cuda13x[ctk]` или `cupy-cuda12x[ctk]`;
дополнительный набор `ctk` устанавливает CUDA-компоненты в Python-окружение,
но NVIDIA-драйвер остаётся системной зависимостью. Не следует одновременно
устанавливать разные варианты CuPy. Основание:
[официальная инструкция CuPy](https://docs.cupy.dev/en/stable/install.html).

Локально проверены CuPy 14.2.0, системный CUDA runtime 13020, RTX 3050 Laptop
с 4 ГБ памяти и CUDA-кернелы пакета. CPU-профили не импортируют CuPy.

## Шрифты и LaTeX

`style.text_engine="mathtext"` — явно выбранный режим по умолчанию: Computer
Modern из Matplotlib, математические подписи через MathText. Он не требует
системного TeX.

`style.text_engine="latex"` включает `text.usetex=True`. Нужны работоспособный
LaTeX, его форматы и пакеты, а для растрового Agg-вывода — `dvipng`. Установка
только бинарного файла `latex` недостаточна. Дополнительные требования зависят
от формата экспорта:
[руководство Matplotlib](https://matplotlib.org/stable/users/explain/text/usetex.html).
При ошибке библиотека не меняет движок автоматически.

На локальной машине установлен MiKTeX 25.12 (`latex`, `pdflatex`). Успешный
рендер режима `latex` проверяется отдельно; учебные графики по умолчанию
используют MathText. Кириллические подписи требуют подходящих шрифтов и
настройки TeX.

## Потоки и ресурсы

Публичный аргумент сохранён в пользовательском написании `num_treads`. Первая
версия допускает только `1`; значение больше единицы вызывает ошибку. Это не
ограничивает автоматически внутренние потоки системного BLAS. Для строгих
измерений окружение должно отдельно контролировать `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` и `MKL_NUM_THREADS` до запуска Python. Профилировщик и общий бенчмарк реализованы в `0.1.0a9`; для строгого
ограничения нативных потоков используется `threadpoolctl`, а фактическая
библиотека сохраняется в measurement environment.

## Файловая система

Экспорт создаёт новые файлы через исключительное создание hard link на файловой
системе назначения. Поддержка проверена на Linux (исходная сборка) и на Windows
10/NTFS (локальная проверка, включая одновременные экспорты). Файловые системы
без hard links не проверялись. При конфликте имён нет автоматической
перезаписи или переименования.
