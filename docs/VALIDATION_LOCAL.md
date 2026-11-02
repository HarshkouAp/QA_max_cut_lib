# Локальная проверка окружения

**Дата:** 2 ноября 2026 года.
**Машина:** Windows 10 (10.0.19043), 4 физических / 8 логических ядер, 15.7 ГБ RAM.
**Python:** 3.12.14 в `.venv-stage02`.
**Проверяемая сборка:** `QA_max_cut_lib 0.1.0a2`.
**Подробный отчёт этапа:** `VALIDATION_0.1.0a2.md`.

## Выполненные проверки

| Проверка | Команда | Результат |
|---|---|---|
| Автоматические тесты | `.venv-stage02/Scripts/python.exe -m pytest -q` | **56 passed** |
| ExactCPU, CLI | `python -m QA_max_cut_lib.solvers.exact_cpu --report 1 --logs 0` | `completed`, Cut = 5 |
| QAOACPU, CLI | `python -m QA_max_cut_lib.solvers.qaoa_cpu --report 1 --logs 0` | `completed`, 20 шагов ADAM |
| Quickstart | `python examples/quickstart.py` | Exact и QAOA дали энергию −6, quality = 1.000 |
| Ноутбук 01 | `jupyter nbconvert --execute notebooks/01_quickstart.ipynb` | выполнен целиком, 5 изображений, сохранённые JSON/NPY |
| Ноутбук 02 | `jupyter nbconvert --execute notebooks/02_current_api_guide.ipynb` | выполнен целиком, 6 изображений, карта и графики |
| Ноутбук 03 | `jupyter nbconvert --execute notebooks/03_reduction_core.ipynb` | 8 кодовых ячеек, 0 ошибок |
| Сборка wheel | `python -m build` | созданы wheel и sdist `0.1.0a2` |
| Чистая установка wheel | `.venv-wheelcheck` вне проекта | импорт из `site-packages`, редукция и солверы работают |
| Сборка Sphinx | `python -m sphinx -W -b html docs docs/_build/html` | успешно, без предупреждений |
| GPU-окружение `.venv` (отдельно) | CuPy 14.2.0, CUDA 13.0, `RawKernel` | проверено ранее; в этапе 02 не используется |
| Целостность исходников | SHA-256 89 Python-файлов | 89/89 совпадают с распаковкой `RQAOA_main.rar` |
| Импорт профиля | `import QA_max_cut_lib, cvxpy, plotly, dimod, dwave.samplers, community` | успешно |

## Версии основных пакетов

| Пакет | Версия |
|---|---|
| NumPy | 2.5.4 |
| SciPy | 1.18.1 |
| pandas | 3.1.0 |
| Matplotlib | 3.11.2 |
| Seaborn | 0.13.2 |
| Plotly | 7.1.0 |
| CVXPY | 1.9.3 |
| SCS | 3.3.1 |
| NetworkX | 3.7 |
| dimod | 0.12.23 |
| dwave-samplers | 1.8.0 |
| pytest | 9.2.0 |
| Sphinx | 9.1.1 |
| CuPy | 14.2.0 (`cupy-cuda13x`) |

Полный список: `requirements/freeze-local-py312-win.txt`; разрешённые версии
профилей — `requirements/lock-cpu-py312-win.txt` и
`requirements/lock-gpu-cu13-py312-win.txt`.

## Что подтверждено

- Пакет устанавливается в отдельное окружение в режиме `-e`.
- 42 прежние проверки и 14 новых проверок ядра редукций проходят на Windows и Python 3.12.
- Импорт пакета не загружает Matplotlib и CuPy.
- Модульные CLI-примеры работают без запуска вычислений при импорте.
- Графики строятся в локальной теме и не меняют глобальные `matplotlib.rcParams`.
- Сохранение JSON/NPY, защита от перезаписи и одновременные экспорты проходят тесты.

## Что не проверялось этой командой

- Полный GPU-солвер: проверены только CuPy, устройство и компиляция ядра.
- Сборка Sphinx и wheel проверены; их содержимое не сверялось с исходным
  `ARTIFACTS.json` прежней сборки.
- Численное качество QAOA на больших графах: стартовая сборка ограничена
  полным вектором состояния и небольшими задачами.
- Режим `style.text_engine="latex"`: ноутбуки этапа 02 используют MathText;
  настоящий LaTeX не входит в приёмку этапа.
