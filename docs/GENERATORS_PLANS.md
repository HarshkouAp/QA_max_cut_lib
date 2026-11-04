# Generators, typed plans, and coverage

This page covers the input side of the library: how a graph is produced from a
recipe, how a structured weight matrix is turned into a graph, and how a decomposition
is represented so that a local solver can be told exactly which edges it owns. Given a
recipe or a plan, each section states what the object promises, what it deliberately
does not promise, and which coverage number to check. The running example is a
four-vertex graph decomposed into an edge partition, because that case exposes the
differences between vertex, edge, and pair coverage at once.

Two rules apply throughout. Generated metadata — grid coordinates, external labels,
benchmark family names — never enters the objective. And a plan is a description of a
subproblem, not a claim that solving the parts solves the whole.

## Recipe families

`schema_version=1` recipes (`complete`, `gnm`, `erdos_renyi`) keep their original
semantics. New families use `schema_version=2`: `gnm_density_v1`,
`connected_path_fill_v1`, `regular_v1`, `barabasi_albert_v1`, `watts_strogatz_v1`,
`sbm_v1`, `geometric_v1`, `periodic_grid_v1`, plus the structured recipes
`wishart_projected_v1` and `hebbian_patterns_v1`.

The families differ in what is fixed and what is random, and the fixed part matters
for reproduction:

- `gnm_density_v1` produces exactly `m = floor(density * n(n-1)/2)` edges. It is not a
  Bernoulli `G(n, p)` draw, so the edge count is deterministic given `n` and
  `density`.
- `connected_path_fill_v1` starts from a path on the labels in order and adds random
  missing pairs until `n_edges` is reached. The result is connected, but it is not a
  uniform sample of connected graphs: the path edges are always present.
- `periodic_grid_v1` wraps coordinates; it never creates a self-loop and never emits
  the same wrapped edge twice.

`GraphFactory.from_recipe(recipe)` returns a `WeightedGraph`; `graph.snapshot()`
records the resolved topology seed, weight seed, and requested sizes, so a run can be
described after the fact without replaying the generator.

Benchmark resolvers decode an existing benchmark instead of generating one.
`legacy_bench16_v1` accepts identifiers `0..15`; `benchmark297_legacy_v1` decodes an
identifier into a family, a slot, an instance size, and a weight profile, with no
graph generation at resolve time. For example, `resolve_bench297(17)` reports family
`er_sparse`, slot `5`, `n_vertices=200`, and the seeds `50042` (topology) and `50345`
(weights). Identifiers outside `0..296` raise `ValueError`.

## Structured weights

`wishart_projected_v1` builds a low-rank factor and projects it:

\[
W=\sqrt{\tfrac{N}{N-1}}\,(I-tt^{\mathsf T}/N)Z,\qquad B=WW^{\mathsf T},
\]

and takes the off-diagonal entries `B_ij` as weights. The diagonal is not added to
the graph, because a self-loop would change the cut without changing the partition.

The record carries three independent statuses: `theoretical_optimum_cut`,
`numerically_checked`, and `stored_graph_optimum_verified`. They are not
interchangeable. A small residual between a candidate and a stored graph is not a
certificate, and `stored_graph_optimum_verified=False` means exactly that, even when
the candidate equals the theoretical value.

`hebbian_patterns_v1` uses

\[
w_{ij}=-\frac1N\sum_\mu a_\mu\,\xi_i^\mu\xi_j^\mu .
\]

The default profile is `linspace(1, N, M)` with the last coefficient set to `2N`.
This is a named profile, not evidence about an optimum. Every pattern candidate is
reported as `not_certified` until `ExactCPU` or an explicit proof confirms the stored
graph.

## Typed plans

A `SubproblemPlan` carries `plan_kind`, the parent hash, the builder recipe, and the
parts. Each `SubproblemSpec` carries `nodes`, `edge_domain`, `edges`, anchors, and
provenance. Three plan kinds exist, and they answer different questions:

- `vertex_partition` — disjoint, complete cover of the vertices. Edges between parts
  are not part of any subproblem, so the sum of part objectives is not the parent
  objective.
- `induced_cover` — overlap is allowed and the coverage is measured separately; a
  vertex may appear in several parts.
- `edge_partition` — explicit assigned edges, pairwise exclusive. A complete plan
  covers every parent edge exactly once.

For an edge partition the vertex list does not reconstruct the local Hamiltonian: two
parts may share a vertex while owning different edges, and an induced subgraph on
those vertices would contain edges belonging to another part. `validate_plan` checks
the parent hash, the weights, and the edge domain, and `plan_to_graph` returns exactly
the declared local graph — no induced closure.

## Builders

- `connected_bfs_partition_v1` — BFS blocks, with optional merging of singletons.
- `adaptive_dense_induced_v1` — corrected rating, growth, and repair with an explicit
  overlap policy.
- `edge_neighborhood_cover_v1` — anchor endpoints plus a cover of their neighbours.
- `edge_coverage_scan_v1` — atomic episodes under a strict `max_tasks` cap.
- `overlap_edge_partition_abs_v1` — edge-exclusive parts with signed weights and
  active-by-unassigned-edge semantics.

## Coverage and hops

`analyze_plan(graph, plan)` computes vertex, edge, and pair coverage independently,
together with multiplicities, repeated incidences, absolute-weight coverage, topology
descriptors, and hop metrics. The three coverage numbers are not derivations of one
another: a plan can cover every vertex and every edge while covering only some of the
vertex pairs — in the example below, `3` of `6`.

Hop distance is a BFS count of edges. Signed cost weights are not lengths, so a
negative weight does not shorten a hop. `mean_hop_distance` is `null` for a
disconnected graph; the reachable mean, reachability, and global efficiency are
reported separately instead of being folded into one misleading average.

## Running example

The recipes first, with real counts: `gnm_density_v1` at `n=6`, `density=0.5` gives
`floor(0.5*15) = 7` edges; `connected_path_fill_v1` at `n=6`, `n_edges=9` gives `9`
edges on `6` vertices; `periodic_grid_v1` at `3 x 3` with both axes periodic gives
`18` edges.

Now decompose the four-vertex graph with edges `(-7,10,2)`, `(10,30,-3)`,
`(-7,30,5)` and an isolated vertex `80`:

```python
from QA_max_cut_lib import WeightedGraph
from QA_max_cut_lib.generators.plans import build_overlap_edge_partition, validate_plan
from QA_max_cut_lib.tools.partition_analysis import analyze_plan

graph = WeightedGraph([(-7, 10, 2), (10, 30, -3), (-7, 30, 5)],
                      nodes=[-7, 10, 30, 80])
plan = build_overlap_edge_partition(graph, max_size=3, seed=0)
print(plan.plan_kind, plan.completion_status)
print(plan.coverage)
print(validate_plan(graph, plan), analyze_plan(graph, plan)["pair_coverage"])
```

The output is

```text
edge_partition complete
{'vertices': 4, 'vertices_total': 4, 'vertex_fraction': 1.0, 'edges': 3, 'edges_total': 3, 'edge_fraction': 1.0, 'pairs': 3, 'pairs_total': 6, 'pair_fraction': 0.5, 'pair_multiplicity_max': 1, 'edge_multiplicity_max': 1, 'vertex_complete': True, 'edge_complete': True, 'pair_complete': False, 'uncovered_vertices': [], 'uncovered_edges': []}
True 0.5
```

The plan has two parts, `(-7, 10, 30)` and `(80,)`: the isolated vertex is carried as
its own part rather than dropped. The plan is complete in the sense that matters for
an edge partition: all three edges are owned exactly once
(`edge_multiplicity_max=1`). It covers every vertex, yet only `3` of the `6` vertex
pairs, so `pair_coverage` is `0.5` — a number a vertex-only view would hide. The same
plan round-trips through `SubproblemPlan.to_dict`/`from_dict` without changing its
parts.

## Limits

- The full `GraphDecomposer_light` (truss, repair, and sampling branches) is not
  implemented. Only the builders listed above exist.
- No Louvain-like community detection and no pattern-recovery oracle.
- Adjacency-row Pearson correlation is not a `ZZ` correlation and is not offered as
  one.
- The edge-subset adapter requires `local_problem=edge_subset_no_fields`. An
  induced-only LS recipe must reject an edge partition rather than silently take the
  induced closure; `validate_capability` raises `LocalProblemCapabilityError` in that
  case.

## Where next

- [API reference](API.md) — recipe fields, plan fields, and the `partition` helper.
- [Concepts](CONCEPTS.md) — labels, `node_order`, and cut values.
- [Choosing an algorithm](CHOOSE_ALGORITHM.md) — which solver accepts which plan kind.
