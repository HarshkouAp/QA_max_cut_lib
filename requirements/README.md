# Профили зависимостей

| Файл | Назначение |
|---|---|
| `cpu.in` | Полный CPU-профиль: массивы, графы, таблицы, JSON-схемы, графика, CPU-параллелизм, классические солверы. |
| `gpu-nvidia-cu12.in` | CPU-профиль + `cupy-cuda12x[ctk]` для CUDA 12.x. |
| `gpu-nvidia-cu13.in` | CPU-профиль + `cupy-cuda13x[ctk]` для CUDA 13.x. |
| `notebooks.in` | JupyterLab, ipykernel, nbformat, nbclient. |
| `docs.in` | Sphinx, numpydoc, pydata-sphinx-theme, myst-nb. |
| `dev.in` | pytest, build, wheel. |
| `cpu.txt`, `gpu-cu12.txt`, `gpu-cu13.txt`, `notebooks.txt`, `docs.txt`, `dev.txt` | Установка через extras пакета. |
| `lock-cpu-py312-win.txt` | Разрешённые версии CPU-профиля для Python 3.12/Windows. |
| `lock-gpu-cu13-py312-win.txt` | Разрешённые версии GPU-профиля CUDA 13 для Python 3.12/Windows. |
| `freeze-local-py312-win.txt` | Фактическое окружение этой машины. |

Устанавливается ровно один GPU-профиль. Локально использован
`cupy-cuda13x` без `ctk`, потому что системный CUDA Toolkit 13.0 уже установлен
и проверен. Для чистой машины предпочтителен профиль с `ctk`.
