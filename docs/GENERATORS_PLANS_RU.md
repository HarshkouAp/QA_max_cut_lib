# Generators, typed plans и coverage (a18)

## Recipes

`schema_version=1` recipes `complete`, `gnm`, `erdos_renyi` сохранены без
изменения семантики. Для новых семейств используется `schema_version=2`:
`gnm_density_v1`, `connected_path_fill_v1`, `regular_v1`,
`barabasi_albert_v1`, `watts_strogatz_v1`, `sbm_v1`, `geometric_v1`,
`periodic_grid_v1`.

`gnm_density_v1` — ровно `m=floor(rho*n(n-1)/2)` рёбер, не Bernoulli
`G(n,p)`. `connected_path_fill` начинает с path и добавляет случайные
недостающие пары; это не равномерное распределение всех connected
graphs. Grid coordinates и внешние labels — metadata; self-loops и
дубли wrap-рёбер не создаются.

Resolver `legacy_bench16_v1` принимает IDs 0..15. Resolver
`benchmark297_legacy_v1` декодирует 11 families × 9 slots × 3 weights
без generation; `resolve_bench297(17)` даёт seeds 50042/50345.

## Структурированные веса

Wishart: `W=sqrt(N/(N-1))(I-tt^T/N)Z`, `B=WW^T`, веса — off-diagonal
`B_ij`. Диагональ в graph не добавляется. `theoretical_optimum_cut`,
`numerically_checked` и `stored_graph_optimum_verified` — независимые
статусы; residual не является сертификатом.

Hebbian: `w_ij=-(1/N)sum_mu a_mu xi_i xi_j`. Default profile
`linspace(1,N,M)` с последним `2N`; это именованный profile, не
доказательство optimum. Pattern candidate всегда `not_certified`, пока
ExactCPU/proof не подтвердил stored graph.

## Typed plans

`SubproblemPlan` имеет `plan_kind`, parent hash, builder recipe и parts.
`SubproblemSpec` хранит `nodes`, `edge_domain`, `edges`, anchors и
provenance.

- `vertex_partition` — непересекающееся полное покрытие vertices;
- `induced_cover` — перекрытие допустимо, покрытие измеряется отдельно;
- `edge_partition` — явные назначенные edges, pairwise exclusive; полный
  план покрывает все parent edges.

Для edge partition список vertices не восстанавливает local Hamiltonian.
`validate_plan` проверяет parent hash, веса и edge domain. `plan_to_graph`
возвращает ровно объявленный local graph.

## Builders

- `connected_bfs_partition_v1` — BFS-блоки, optional merge singletons;
- `adaptive_dense_induced_v1` — corrected rating/growth/repair с
  explicit overlap policy;
- `edge_neighborhood_cover_v1` — anchor endpoints + cover neighbors;
- `edge_coverage_scan_v1` — atomic episodes и strict max_tasks;
- `overlap_edge_partition_abs_v1` — edge-exclusive parts с signed
  weights и active-by-unassigned-edge semantics.

## Tools

`analyze_plan` независимо считает vertex/edge/pair coverage,
multiplicities, repeated incidences, abs-weight coverage, topology
descriptors и hop metrics. Hop distance — BFS по числу рёбер; signed cost
weights не являются длинами. `mean_hop_distance` для disconnected graph
равен null, отдельно доступны reachable mean, reachability и global
efficiency.

## Ограничения

Полный GraphDecomposer_light (truss/repair/sampling branches),
Louvain-like community detection, pattern recovery oracle и
adjacency-row Pearson как ZZ-correlation не заявлены. Edge-subset adapter
явно требует `local_problem=edge_subset_no_fields`; induced-only
LS не должен принимать edge partition.
