# Git Stage 07

Коммит реализации: `1ebfd75` — `Stage 07: add hierarchical QAOA-in-QAOA (0.1.0a7)`.

В коммит вошли также ранее не зафиксированные файлы этапа 06 (`QAOAGPU`,
GPU-источник, GPU-RQAOA/LS, ноутбук 07 и отчёт a6), потому что локальная база
до начала этапа 07 уже содержала их, но они не были отдельным коммитом.

## Состав этапа 07

- `src/QA_max_cut_lib/solvers/hierarchy_components.py` — строгая проверка
  локального solver, нормализация рисунка, независимая проекция `W/Delta b`,
  worker-функция;
- `src/QA_max_cut_lib/solvers/qaoa_in_qaoa_cpu.py` — общий координатор,
  CPU-пул, квоты, атомарная проекция и дерево;
- `src/QA_max_cut_lib/solvers/qaoa_in_qaoa_gpu.py` — один GPU-владелец;
- конфигурации CPU/GPU/shots/random partitioner и граф N=128;
- тесты `tests/test_qaoa_in_qaoa.py` и GPU-тест;
- Tools `hierarchy_plots.py`;
- ноутбук `08_qaoa_in_qaoa.ipynb` и HTML-превью;
- документация, отчёт и JSON профилей.

## Проверки перед коммитом

```text
CPU: 124 passed, 1 skipped, 1 warning
GPU: 10 passed
Sphinx -W: успешно
wheel/sdist: 0.1.0a7
```

Артефакты сборки находятся в `dist/` и не входят в Git-коммит
(`dist/` исключён `.gitignore`); они добавляются в передаваемый zip отдельно.
