# Git Stage 08

Коммит реализации: `d31cf13` — `Stage 08: add classical baselines and exact block refinement (0.1.0a8)`.

## Состав

- `solvers/goemans_williamson_cpu.py` — CVXPY/SCS SDP, контролируемый
  спектральный ремонт и hyperplane rounding;
- `solvers/simulated_annealing_cpu.py` — dimod spin BQM и классический
  `dwave.samplers.SimulatedAnnealingSampler`;
- `solvers/exact_block_refiner.py` — условные поля, последовательный exact
  block enumeration и явные лимиты состояний;
- `solvers/refined.py` — обёртка base solver → refiner с независимыми
  `base_run` и `refinement_run`;
- `tools/classical_plots.py` — `analyze_solution` и графики GW/SA/refinement;
- конфигурации, тесты, ноутбук `09`, отчёт и профиль `classical`.

## Проверки

```text
CPU: 134 passed, 1 skipped, 1 warning
GPU: 10 passed
Sphinx -W: успешно
wheel/sdist: 0.1.0a8
CLI из установленного wheel: GW, SA, refined
```

`dist/` исключён из Git и добавляется в передаваемый zip отдельно.
