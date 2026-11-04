# Проверка 0.1.0rc2

Дата: 3 ноября 2026 года.

- `218 passed, 1 skipped, 1 warning`.
- Sphinx `-W`: успешно.
- `00_quickstart.ipynb` и `01_full_guide.ipynb` выполнены.
- Benchmark297 materialized 297/297; 48 generator/plan cases = 0 shots.
- Added `BatchLSRQAOACPU` real CPU batch epoch/parity/active-endpoints
  assembly and `QuantizedQAOACPU` post-training quantization.
- GPU tests: `10 passed` in `.venv` (current source; G8 local evidence).

## Артефакты

- wheel SHA-256: `66eae140c9b6781456a10eb2914d752d4be3bc31caf67b3b2650b18cd913c709`
- sdist SHA-256: `2b428c0aa81665c59489c71861a264efa9e5330171e78dbe5cacd5ff821cb172`

## Публичный quickstart

`examples/public_quickstart.py` запущен из установленного wheel вне `src`:
точный оптимум `3.5`, expected cut `3.044365`, best-of-128 `3.5`,
optimal mass `0.805755`. SVG пересобран; Sphinx `-W` успешно.
