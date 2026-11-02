# Проверка 0.1.0a18

Дата: 2 ноября 2026 года.

## База

- `QA_max_cut_lib 0.1.0a18`, Python 3.12.14.
- CPU: `211 passed, 1 skipped, 1 warning`.
- Sphinx `-W`: успешно.
- Wheel/CLI/notebook вне src проверяются после сборки.

## Recipes и benchmarks

- `connected_path_fill_v1`: N=6, m=9; N=8, m=12; path + доп. пары.
- `gnm_density_v1`: rho=0.5, N=4 → ровно 3 edges.
- Periodic grid: (2,2) → 4 edges; (3,3) → 18 edges; (1,3) → 3 edges,
  без self-loops/двойных wrap.
- Regular/BA/WS/SBM/geometric проходят preflight и materialization на
  малых профилях.
- `resolve_bench297(17)`: topology seed 50042, weight seed 50345.
- `resolve_bench297(296)`: 980042/980345.
- Все 297 IDs декодированы; `materialize_bench297` materialized 297
  instances за ~27 s в CPU validation profile, без solver calls.
- `legacy_bench16_v1` ID13 → periodic (10,12), 120 vertices.

## Structured constructions

Wishart witness W=(1,1,2)^T, t=(+1,+1,−1), labels (80,−7,10):
off-diagonal weights (1,2,2), candidate cut 4, theoretical cut 4,
residual 0; `stored_graph_optimum_verified=false`.

Hebbian counterexample N=3, patterns 5×(+1,+1,+1), last
(+1,+1,−1), coefficients `[1,1.4,1.8,2.2,2.6,6]`:
weights (−5,−1,−1), last candidate cut −2, all-plus cut 0. Ни один
pattern не объявлен optimum.

## Typed plans и analysis

- Edge partition на G с 4 labels: 3 assigned edges, vertex coverage 1,
  edge coverage 1, pair coverage 0.5; assigned sum cuts equals parent
  cut for checked assignments.
- Vertex partition: vertex coverage 1, edge coverage 1/3; boundary edges
  остаются видимыми в analysis.
- `edge_neighborhood_cover_v1`: anchor endpoints входят в каждую часть;
  `cover_neighbors` требует k≥3, `anchor_only` допускает k=2.
- `edge_coverage_scan_v1`: strict max_tasks, isolates, explicit edge
  domains; E=0 даёт null edge ratio без деления.
- `overlap_edge_partition_abs_v1`: signed +2/−2 incident edges остаются
  active; все assigned edges сохраняют weights.
- `analyze_plan`: coverage/multiplicity/abs-weight, topology descriptors,
  hop metrics; `mean_hop_distance` connected path N=3 = 4/3.

## 48-case series

`scripts/run_generation_series.py`:
6 graphs × 4 builders × 2 seeds = **48** cases; все планы complete;
**0 quantum shots**. Результат:
`results/generators_series/stage18_48_cases.json`.

## Ограничения

- Полный GraphDecomposer_light truss/repair/sampling branches не перенесён.
- Реальный Louvain/community builder не заявлен.
- Pattern recovery/adjacency-row Pearson не solver oracle.
- Edge-subset adapter проверен на typed plan и локальном payload; полный
  отдельный edge-subset LS controller не заявлен.
- 297 graphs не решались Max-Cut; это generation/decoder validation.

## Артефакты

- wheel: `qa_max_cut_lib-0.1.0a18-py3-none-any.whl`, SHA-256
  `c360c568de4f170f764a13a9ae08616a5bcc57183f63c201d2e1584faa8065f6`;
- sdist: `qa_max_cut_lib-0.1.0a18.tar.gz`, SHA-256
  `a1bdd97a6a2ca4d9a9d23791b8c09fc7847f60d04462fecb7e21adebf4076bab`.
