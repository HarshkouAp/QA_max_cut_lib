# Git: этап 02

Базовая точка: `8dc86ed` (окружение и документация перед этапом 02).

## Коммиты
```
5d7cd11 Record input artifact hashes in stage 02 report
457f20e Add stage 02 git summary and patch
1b5833b Stage 02: add exact Max-Cut to Ising reduction core (0.1.0a2)
```

Основной diff `docs/STAGE02_DIFF.patch` соответствует коммиту `1b5833b`
(исходный код, тесты, документация и выполненные ноутбуки). Коммиты
`457f20e` и `5d7cd11` добавляют только отчётные материалы.

## Изменённые файлы
```
 .gitignore                                       |    1 +
 README.md                                        |   45 +-
 docs/API.md                                      |  127 +-
 docs/ARCHITECTURE_MAP_RU.md                      |   16 +-
 docs/MATHEMATICS.md                              |   80 +
 docs/MIGRATION_STATUS.md                         |   65 +-
 docs/PROJECT_SETUP_RU.md                         |   17 +-
 docs/VALIDATION.md                               |    3 +
 docs/VALIDATION_0.1.0a2.md                       |  117 +
 docs/VALIDATION_LOCAL.md                         |   19 +-
 docs/api.rst                                     |   15 +
 docs/conf.py                                     |    6 +-
 docs/index.rst                                   |   10 +-
 docs/notebook_previews/01_quickstart.html        |   51 +-
 docs/notebook_previews/02_current_api_guide.html |   77 +-
 docs/notebook_previews/03_reduction_core.html    | 8241 ++++++++++++++++++++++
 docs/validation_environment.json                 |    1 +
 docs/validation_environment_0.1.0a2.json         |   85 +
 notebooks/01_quickstart.ipynb                    |  119 +-
 notebooks/02_current_api_guide.ipynb             |  191 +-
 notebooks/03_reduction_core.ipynb                |  433 ++
 pyproject.toml                                   |    5 +-
 requirements/gpu-cu12.txt                        |    2 +-
 src/QA_max_cut_lib/__init__.py                   |   18 +-
 src/QA_max_cut_lib/_version.py                   |    8 +
 src/QA_max_cut_lib/core/__init__.py              |   11 +-
 src/QA_max_cut_lib/core/factory.py               |    4 +-
 src/QA_max_cut_lib/core/reduction.py             | 1197 ++++
 src/QA_max_cut_lib/solvers/base.py               |    7 +-
 src/QA_max_cut_lib/tools/landscape.py            |    3 +-
 src/QA_max_cut_lib/tools/plotting.py             |    3 +-
 tests/test_reduction.py                          |  399 ++
 32 files changed, 11177 insertions(+), 199 deletions(-)
```

Полный diff: `docs/STAGE02_DIFF.patch`.
