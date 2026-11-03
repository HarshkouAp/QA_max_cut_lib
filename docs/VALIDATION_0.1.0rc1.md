# Проверка 0.1.0rc1

Дата: 3 ноября 2026 года.

## Фактический статус

`0.1.0rc1` — **release candidate**, не финальный `0.1.0`.
Blockers: полный Stage 19 controller set, 89-source closure, G8 GPU
revalidation и полная R01–R65 evidence matrix. См.
`RELEASE_READINESS.md`.

## Проверки CPU

- `216 passed, 1 skipped, 1 warning`.
- Sphinx `-W`: успешно.
- Wheel установлен в `.venv` и `.venv-wheelcheck`; CLI/examples вне src.
- `00_quickstart.ipynb` и `01_full_guide.ipynb` выполнены в CPU kernel.
- 48-case generator/plan/analysis series: 0 quantum shots.
- Benchmark297: 297/297 materialized, без Max-Cut solves.

## Formula components

Проверены `ParityForest` (accepted/conflict/redundant), odds fusion
`15/17`, selected observation, quantization ties-to-even, restriction
proxy `6.3`, legacy/positive history mixing и Fréchet-invalid
hierarchical proxy `sqrt(.19)`.

## Ограничения

End-to-end `BatchLSRQAOACPU`, backtracking, hierarchical source,
quantized solver wrappers и полный каталог legacy branches не заявлены.
G8 GPU в этом цикле не перепроверен; CPU candidate не подменяет GPU
acceptance.

## Артефакты

- wheel: `qa_max_cut_lib-0.1.0rc1-py3-none-any.whl`, SHA-256
  `5aa80a0db4ffad03254a1bab9206b52a54f96320c6d0a19ee648fd9e0e74fda9`;
- sdist: `qa_max_cut_lib-0.1.0rc1.tar.gz`, SHA-256
  `d2d35adb7ed281cf40cecc64c5f724041a852a7c4f6fe4df3340644598561441`.
