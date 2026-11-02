# Git Stage 11

Основной коммит: `e08a343` — `Stage 11: add Laboratory catalog and warm-start CPU (0.1.0a11)`.

## Состав

- versioned `laboratory/data/inventory.json` (89 files) and `catalog.json`;
- inventory/catalog JSON schemas and import-light `catalog.py`;
- `laboratory` list/show CLI;
- `WarmStartQAOACPU` and `warm_start` components;
- provided-spin, fractional-probability, GW and SA configs;
- H series (`experiments/benchmark_warm_start.json`);
- notebook 12 and HTML/PNG preview;
- `LABORATORY_RU.md`, validation report, tests and examples.

## Checks

```text
inventory: 89/89 members, 89 parsed, 0 parse errors
CPU: 161 passed, 1 skipped, 1 warning
GPU: 10 passed (no warm-start GPU claim)
Sphinx -W: success
H: 42 runs, 42 valid, 0 failed, 22848 quantum shots
cross-process pause/resume: known=160, resumed completed, shots=544
wheel/sdist: 0.1.0a11
```

Only `warm_start_product_qaoa_cpu_v1` is runnable. WS_RQAOA, LS/MK
families, fields/environments and MI diagnostics stay deferred with visible
reasons.
