# Verification 0.1.0a18

Date: 2 November 2026.

## Base

- `QA_max_cut_lib 0.1.0a18`, Python 3.12.14.
- CPU: `211 passed, 1 skipped, 1 warning`.
- Sphinx `-W`: success.
- Wheel/CLI/notebook outside `src` are checked after the build.

## Recipes and benchmarks

- `connected_path_fill_v1`: N=6, m=9; N=8, m=12; path plus extra pairs.
- `gnm_density_v1`: rho=0.5, N=4 → exactly 3 edges.
- Periodic grid: (2,2) → 4 edges; (3,3) → 18 edges; (1,3) → 3 edges,
  without self-loops or double wrap.
- Regular/BA/WS/SBM/geometric pass preflight and materialization on small
  profiles.
- `resolve_bench297(17)`: topology seed 50042, weight seed 50345.
- `resolve_bench297(296)`: 980042/980345.
- All 297 IDs were decoded; `materialize_bench297` materialized 297 instances
  in about 27 s in the CPU validation profile, without solver calls.
- `legacy_bench16_v1` ID13 → periodic (10,12), 120 vertices.

## Structured constructions

Wishart witness W=(1,1,2)^T, t=(+1,+1,−1), labels (80,−7,10): off-diagonal
weights (1,2,2), candidate cut 4, theoretical cut 4, residual 0;
`stored_graph_optimum_verified=false`.

Hebbian counterexample N=3, patterns 5×(+1,+1,+1), last (+1,+1,−1),
coefficients `[1,1.4,1.8,2.2,2.6,6]`: weights (−5,−1,−1), last candidate cut
−2, all-plus cut 0. No pattern is declared optimal.

## Typed plans and analysis

- Edge partition on G with 4 labels: 3 assigned edges, vertex coverage 1,
  edge coverage 1, pair coverage 0.5; the assigned sum of cuts equals the
  parent cut for the checked assignments.
- Vertex partition: vertex coverage 1, edge coverage 1/3; boundary edges
  remain visible in the analysis.
- `edge_neighborhood_cover_v1`: anchor endpoints belong to every part;
  `cover_neighbors` requires k≥3, `anchor_only` allows k=2.
- `edge_coverage_scan_v1`: strict max_tasks, isolates, explicit edge domains;
  E=0 gives a null edge ratio without division.
- `overlap_edge_partition_abs_v1`: signed +2/−2 incident edges stay active;
  all assigned edges keep their weights.
- `analyze_plan`: coverage/multiplicity/abs-weight, topology descriptors, hop
  metrics; `mean_hop_distance` for a connected path with N=3 = 4/3.

## 48-case series

`scripts/run_generation_series.py`:
6 graphs × 4 builders × 2 seeds = **48** cases; all plans complete;
**0 quantum shots**. Result:
`results/generators_series/stage18_48_cases.json`.

## Limitations

- The full GraphDecomposer_light truss/repair/sampling branches are not
  ported.
- No real Louvain/community builder is claimed.
- Pattern recovery/adjacency-row Pearson is not a solver oracle.
- The edge-subset adapter is verified on a typed plan and a local payload; a
  full separate edge-subset LS controller is not claimed.
- The 297 graphs were not solved as Max-Cut; this is generation/decoder
  validation.

## Artifacts

- wheel: `qa_max_cut_lib-0.1.0a18-py3-none-any.whl`, SHA-256
  `c360c568de4f170f764a13a9ae08616a5bcc57183f63c201d2e1584faa8065f6`;
- sdist: `qa_max_cut_lib-0.1.0a18.tar.gz`, SHA-256
  `a1bdd97a6a2ca4d9a9d23791b8c09fc7847f60d04462fecb7e21adebf4076bab`.
