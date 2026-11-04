# Check `0.1.0rc2`

Date: 3 November 2026.
Status: **release candidate 2, not final `0.1.0`**.

## Results

| Check | Result |
|---|---|
| CPU suite | `218 passed, 1 skipped, 1 warning` |
| GPU suite (`.venv`, current source) | `10 passed` |
| Sphinx `-W` | success, no warnings |
| `00_quickstart.ipynb`, `01_full_guide.ipynb` | executed |
| Benchmark297 | 297/297 materialized |
| 48 generator/plan cases | 0 quantum shots |
| Wheel import outside `src` | CPU import without CuPy/CUDA |
| Documentation language | `scripts/check_docs_english.py`: English only; 20 notebooks and previews translated |

Added in this candidate: `BatchLSRQAOACPU`, the real CPU batch epoch/parity/
active-endpoint assembly, and `QuantizedQAOACPU`, post-training angle
quantization.

## Artifacts

- wheel SHA-256:
  `5c15255b889685a295ba2341881568347c2fe4a628436d46fa475e3637b81878`
- sdist SHA-256:
  `1bf830d665e9f3070039f5168acb2e4e6a043804b34f246df91d53457cd5f12f`

## Public quickstart

`examples/public_quickstart.py` was run from the installed wheel outside `src`:
exact optimum `3.5`, expected cut `3.044365`, best of 128 shots `3.5`, optimal mass
`0.805755`. The SVG was rebuilt; Sphinx `-W` succeeded.

## Limits

- `0.1.0rc2` is not `0.1.0`. R19 remains open: 18 of the 89 source records are
  reviewed and deferred, so `full_archive_migration_complete=false`. See
  `RELEASE_READINESS.md` and `SOURCE_CLOSURE.json`.
- The GPU result is local evidence on one RTX 3050 Laptop. It does not cover other
  cards, multi-GPU, CUDA Graphs, or `batch>1`.
- The CPU candidate does not replace GPU acceptance, and the GPU result does not
  extend to hardware that was not tested.
- The one warning is a Matplotlib `tight_layout` warning in a parameter-plot test;
  the figure is still produced.
- On a cold Matplotlib font cache the first full run failed once in
  `tests/test_experiments.py::test_plot_reads_profile_without_solver`; the test
  passed alone and the repeated full run returned `218 passed, 1 skipped`.
