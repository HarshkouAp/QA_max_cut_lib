# Verification 0.1.0rc1

Date: 3 November 2026.

## Actual status

`0.1.0rc1` is a **release candidate**, not the final `0.1.0`.
Blockers: the full Stage 19 controller set, 89-source closure, G8 GPU
revalidation and the complete R01–R65 evidence matrix. See
`RELEASE_READINESS.md`.

## CPU checks

- `216 passed, 1 skipped, 1 warning`.
- Sphinx `-W`: success.
- Wheel installed into `.venv` and `.venv-wheelcheck`; CLI/examples outside
  `src`.
- `00_quickstart.ipynb` and `01_full_guide.ipynb` executed in the CPU kernel.
- 48-case generator/plan/analysis series: 0 quantum shots.
- Benchmark297: 297/297 materialized, without Max-Cut solves.

## Formula components

Verified: `ParityForest` (accepted/conflict/redundant), odds fusion `15/17`,
selected observation, quantization ties-to-even, restriction proxy `6.3`,
legacy/positive history mixing and the Fréchet-invalid hierarchical proxy
`sqrt(.19)`.

## Limitations

End-to-end `BatchLSRQAOACPU`, backtracking, hierarchical source, quantized
solver wrappers and the full catalog of legacy branches are not claimed.
G8 GPU was not re-checked in this cycle; a CPU candidate does not replace GPU
acceptance.

## Artifacts

- wheel: `qa_max_cut_lib-0.1.0rc1-py3-none-any.whl`, SHA-256
  `5aa80a0db4ffad03254a1bab9206b52a54f96320c6d0a19ee648fd9e0e74fda9`;
- sdist: `qa_max_cut_lib-0.1.0rc1.tar.gz`, SHA-256
  `d2d35adb7ed281cf40cecc64c5f724041a852a7c4f6fe4df3340644598561441`.
