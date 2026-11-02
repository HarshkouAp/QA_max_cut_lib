from Generators.GraphGenerator import ConnectedRandGraph
from Tools.PartitionEfficiency import *

"""
FastDenseSubgraphGenerator — ускорённая версия генератора плотных подграфов.
Основные принципы: precompute, set-based operations, greedy local-repair,
lightweight heuristics вместо тяжёлых комбинаторных поисков.
Compatible with networkx graphs.
"""

from dataclasses import dataclass
from typing import Optional, Set, Dict, List, Tuple
import random
import time
from collections import deque, defaultdict

import networkx as nx

# -------------------- CONFIG --------------------

@dataclass
class FastConfig:
    max_size: int = 15
    min_size: int = 3
    num_subgraphs: int = 8

    # rating weights
    alpha: float = 1.0        # links to selected
    beta: float = 0.6         # links to candidates
    clustering_w: float = 0.25
    triangle_w: float = 0.8
    proximity_w: float = 0.08

    temperature: float = 0.5

    # coverage / overlap
    allow_vertex_overlap: bool = True   # True: vertices can belong to multiple subgraphs
    max_edge_overlap_ratio: float = 0.0

    # performance knobs
    use_truss: bool = False             # expensive; keep False for large graphs
    truss_k_max: int = 6

    boundary_sample_limit: int = 200    # if boundary large, sample top by degree
    distance_bfs_threshold: int = 5     # when selected >= threshold use multi-source BFS

    # swap/repair parameters (greedy mode)
    max_swap_remove: int = 2
    max_swap_add: int = 3
    candidate_pool_size: int = 10
    repair_attempts_per_subgraph: int = 25

    # coverage
    allow_small_coverage: bool = True

    # strict mode: if True, raise error on any found edge reuse at final audit
    strict_edge_exclusivity_audit: bool = False

    random_seed: Optional[int] = 42

# -------------------- UTIL --------------------

def _edge_key(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u <= v else (v, u)

# -------------------- GENERATOR --------------------

class FastDenseSubgraphGenerator:
    def __init__(self, G: nx.Graph, cfg: Optional[FastConfig] = None):
        self.G = G
        self.cfg = cfg or FastConfig()
        if self.cfg.random_seed is not None:
            random.seed(self.cfg.random_seed)

        # precompute fast structures (sets, degrees, triangles, clustering)
        self._precompute()

        # used edges are exclusive; store as canonical tuples
        self.used_edges: Set[Tuple[int, int]] = set()
        # adjacency map for used edges for faster conflict checks
        self.used_adj: Dict[object, Set[object]] = defaultdict(set)

        # quick membership for edges in original graph
        self._edge_exists = { _edge_key(u, v) for u, v in G.edges() }

    def _precompute(self) -> None:
        G = self.G
        self.nodes = list(G.nodes())
        self.n = len(self.nodes)
        # adjacency sets
        self.neigh: Dict[int, Set[int]] = {v: set(G.neighbors(v)) for v in self.nodes}
        self.degree: Dict[int, int] = {v: len(self.neigh[v]) for v in self.nodes}
        # triangles per node (NetworkX provides optimized routine)
        try:
            self.triangles = nx.triangles(G)
        except Exception:
            # fallback: compute by set intersections (still fairly fast)
            self.triangles = {v: 0 for v in self.nodes}
            for u in self.nodes:
                for v in self.neigh[u]:
                    if v <= u: continue
                    common = self.neigh[u].intersection(self.neigh[v])
                    self.triangles[u] += len(common)
                    self.triangles[v] += len(common)
            # each triangle counted twice per node above for neighbors; but it's a fallback heuristic
        self.max_triangles = max(self.triangles.values()) if self.triangles else 1

        # clustering coefficient precompute
        try:
            self.clustering = nx.clustering(G)
        except Exception:
            self.clustering = {v: 0.0 for v in self.nodes}

    # ---------- Edge exclusivity helpers ----------

    def _mark_edges(self, edges: Set[Tuple[int, int]]) -> None:
        """Atomically mark a set of canonical edges as used."""
        for e in edges:
            a, b = e
            # ensure canonical ordering
            e_norm = _edge_key(a, b)
            if e_norm not in self.used_edges:
                self.used_edges.add(e_norm)
                self.used_adj[e_norm[0]].add(e_norm[1])
                self.used_adj[e_norm[1]].add(e_norm[0])

    def _unmark_edges(self, edges: Set[Tuple[int, int]]) -> None:
        """Remove marks for a set of canonical edges (if present)."""
        for e in edges:
            a, b = e
            e_norm = _edge_key(a, b)
            if e_norm in self.used_edges:
                self.used_edges.remove(e_norm)
            # remove from adjacency if present (use discard to avoid KeyError)
            self.used_adj[e_norm[0]].discard(e_norm[1])
            self.used_adj[e_norm[1]].discard(e_norm[0])

    def _edges_conflict(self, edges: Set[Tuple[int, int]]) -> bool:
        """Return True if any edge in the set is already marked used."""
        for e in edges:
            if _edge_key(e[0], e[1]) in self.used_edges:
                return True
        return False

    def _audit_edge_exclusivity(self, results_nodes_edges: List[Tuple[Set[int], Set[Tuple[int,int]]]]) -> List[Tuple[Tuple[int,int], List[int]]]:
        """
        Audit final partition for edge exclusivity.
        Returns list of conflicts: [(edge, [list of subgraph indices that use it]), ...]
        """
        edge_to_subs: Dict[Tuple[int,int], List[int]] = defaultdict(list)
        for idx, (_, edges) in enumerate(results_nodes_edges):
            for e in edges:
                e_norm = _edge_key(e[0], e[1])
                edge_to_subs[e_norm].append(idx)
        conflicts = [(e, subs) for e, subs in edge_to_subs.items() if len(subs) > 1]
        return conflicts

    # ---------- multi-source BFS distances for set of targets (fast, pure-Python) ----------
    def _multi_bfs_distances(self, sources: Set[int], targets: Set[int]) -> Dict[int, int]:
        if not targets:
            return {}
        INF = 10**6
        dist = {t: INF for t in targets}
        if not sources:
            return dist
        q = deque()
        visited = set()
        for s in sources:
            q.append((s, 0))
            visited.add(s)
            if s in targets:
                dist[s] = 0
        while q and any(dist[t] >= INF for t in targets):
            v, d = q.popleft()
            nd = d + 1
            for nb in self.neigh.get(v, ()):
                if nb in visited: continue
                visited.add(nb)
                if nb in targets:
                    dist[nb] = nd
                q.append((nb, nd))
        return dist

    # ---------- start node selection: prefer nodes with many triangles and free incident edges ----------
    def _select_start_node(self) -> int:
        # quick scoring: triangle count + degree; prefer free incident edges
        best = None; best_score = -1.0
        for v in self.nodes:
            free_incident = any((_edge_key(v, nb) not in self.used_edges) for nb in self.neigh[v])
            score = (self.triangles.get(v, 0) + 0.5 * self.degree.get(v,0))
            if free_incident:
                score += 0.5
            if score > best_score:
                best_score = score; best = v
        return best if best is not None else random.choice(self.nodes)

    # ---------- compute rating for a single node — optimized to use sets and precomputed metrics ----------
    def _node_rating(self, node: int, selected: Set[int], candidates: Set[int],
                     distances: Dict[int, int] = None) -> float:
        neigh = self.neigh[node]
        r1 = self.cfg.alpha * len(neigh & selected)
        r2 = self.cfg.beta * len(neigh & candidates)
        r3 = self.cfg.clustering_w * self.clustering.get(node, 0.0)
        t = self.triangles.get(node, 0)
        r4 = self.cfg.triangle_w * (t / (self.max_triangles or 1))
        r5 = 0.0
        if selected:
            if distances is not None and node in distances:
                avgd = distances[node]
            else:
                # fallback: distance ~ 2 if connected via neighbor else larger
                avgd = 2.0
            r5 = self.cfg.proximity_w * (1.0 / (1.0 + avgd))
        return r1 + r2 + r3 + r4 + r5

    # ---------- select nodes adaptively (top-1 deterministic + weighted sampling) ----------
    def _select_adaptive(self, ratings: Dict[int, float], k: int) -> Set[int]:
        if not ratings or k <= 0:
            return set()
        nodes = list(ratings.keys())
        vals = [ratings[n] for n in nodes]
        if len(nodes) <= k:
            return set(nodes)
        # normalize
        minv, maxv = min(vals), max(vals)
        if maxv > minv:
            norm = [(v - minv) / (maxv - minv) for v in vals]
        else:
            norm = [1.0] * len(vals)
        # top1
        top_idx = max(range(len(norm)), key=lambda i: norm[i])
        chosen = {nodes[top_idx]}
        remaining = [nodes[i] for i in range(len(nodes)) if i != top_idx]
        rem_vals = [norm[i] for i in range(len(nodes)) if i != top_idx]
        # softmax sampling
        if sum(rem_vals) <= 0:
            # fallback uniform
            picks = random.sample(remaining, k-1)
            chosen.update(picks)
            return chosen
        # compute probabilities
        import math
        temps = [math.exp(v / self.cfg.temperature) for v in rem_vals]
        S = sum(temps)
        probs = [t / S for t in temps]
        # draw without replacement
        num = min(k-1, len(remaining))
        # simple weighted sampling without replacement (reservoir-like)
        picked = []
        rem_pool = list(zip(remaining, probs))
        for _ in range(num):
            r = random.random()
            acc = 0.0
            for j, (node, p) in enumerate(rem_pool):
                acc += p
                if r <= acc:
                    picked.append(node)
                    # remove and renormalize
                    rem_pool.pop(j)
                    if rem_pool:
                        s = sum(p for _, p in rem_pool)
                        rem_pool = [(n, p/s) for n, p in rem_pool]
                    break
        chosen.update(picked)
        return chosen

    # ---------- generate one dense candidate as node set (no nx.Graph built inside) ----------
    def generate_dense_candidate(self, start_node: Optional[int] = None) -> Tuple[Set[int], Set[Tuple[int,int]]]:
        if start_node is None:
            start_node = self._select_start_node()
        selected = {start_node}
        boundary = set(self.neigh[start_node]) - selected

        it = 0
        max_it = self.cfg.max_size * 4
        while len(selected) < self.cfg.max_size and boundary and it < max_it:
            it += 1
            # sample large boundary by degree or by triangle importance
            b_list = list(boundary)
            if len(b_list) > self.cfg.boundary_sample_limit:
                b_list.sort(key=lambda x: (self.triangles.get(x,0), self.degree.get(x,0)), reverse=True)
                b_list = b_list[:self.cfg.boundary_sample_limit]

            distances = None
            if len(selected) >= self.cfg.distance_bfs_threshold:
                distances = self._multi_bfs(selected, set(b_list))

            # compute ratings
            ratings = {}
            cand_set = set(b_list)
            for v in b_list:
                ratings[v] = self._node_rating(v, selected, cand_set, distances)

            num_to_add = min(self.cfg.max_size - len(selected), max(1, max(1, len(ratings)//3)))
            new_nodes = self._select_adaptive(ratings, num_to_add)
            if not new_nodes:
                break
            # update
            selected |= new_nodes
            for v in new_nodes:
                for nb in self.neigh[v]:
                    if nb not in selected:
                        boundary.add(nb)
            boundary.difference_update(new_nodes)

        # after nodes selected, compute edges allowed (free edges only)
        edges = set()
        for u in selected:
            for v in self.neigh[u]:
                if v in selected:
                    e = _edge_key(u,v)
                    if e in self._edge_exists and e not in self.used_edges:
                        edges.add(e)

        # Final conflict check (safety): if any edge in `edges` is already used -> remove candidate
        if self._edges_conflict(edges):
            # return empty edge set to indicate candidate conflict (caller will skip)
            return selected, set()

        # ensure connectivity: take largest connected component if required
        if self.cfg.min_size <= len(selected):
            # build small adjacency using edges set for connectivity check
            if edges:
                # BFS from start_node over edges
                comp_nodes = set()
                q = deque([start_node])
                visited = {start_node}
                while q:
                    x = q.popleft()
                    comp_nodes.add(x)
                    for y in self.neigh[x]:
                        if y in visited: continue
                        if y in selected and _edge_key(x,y) in edges:
                            visited.add(y); q.append(y)
                if len(comp_nodes) < len(selected):
                    # reduce selected/edges to this component
                    selected = selected & comp_nodes
                    edges = {e for e in edges if e[0] in selected and e[1] in selected}
        return selected, edges

    # ---------- helper multi-source BFS wrapper using neighbor dict ----------
    def _multi_bfs(self, sources: Set[int], targets: Set[int]) -> Dict[int, int]:
        if not targets:
            return {}
        INF = 10**6
        dist = {t: INF for t in targets}
        if not sources:
            return dist
        q = deque()
        vis = set()
        for s in sources:
            q.append((s,0)); vis.add(s)
            if s in targets: dist[s]=0
        while q and any(dist[t] >= INF for t in targets):
            v,d = q.popleft()
            nd = d+1
            for nb in self.neigh[v]:
                if nb in vis: continue
                vis.add(nb)
                if nb in targets:
                    dist[nb]=nd
                q.append((nb, nd))
        return dist

    # ---------- attach uncovered nodes greedily ----------
    def _attach_uncovered(self, sub_node_edge_list: List[Tuple[Set[int], Set[Tuple[int,int]]]]) -> List[Tuple[Set[int], Set[Tuple[int,int]]]]:
        # compute covered
        covered_nodes = set()
        for nodes, _ in sub_node_edge_list:
            covered_nodes |= nodes
        uncovered = set(self.nodes) - covered_nodes
        if not uncovered:
            return sub_node_edge_list

        # build node->sub index
        node_to_subs = defaultdict(list)
        for idx, (nodes, _) in enumerate(sub_node_edge_list):
            for v in nodes:
                node_to_subs[v].append(idx)

        # process uncovered by descending triangles (prefer cyclic verts)
        for u in sorted(uncovered, key=lambda x: (self.triangles.get(x,0), self.degree.get(x,0)), reverse=True):
            if u in covered_nodes:
                continue
            # free incident neighbors
            free_nbs = [v for v in self.neigh[u] if _edge_key(u,v) in self._edge_exists and _edge_key(u,v) not in self.used_edges]
            if not free_nbs:
                # create small singleton if allowed
                if self.cfg.allow_small_coverage:
                    sub_node_edge_list.append(({u}, set()))
                    covered_nodes.add(u)
                continue
            # try attach to best existing subgraph
            sub_scores = {}
            for nb in free_nbs:
                for si in node_to_subs.get(nb, []):
                    if len(sub_node_edge_list[si][0]) < self.cfg.max_size:
                        sub_scores[si] = sub_scores.get(si,0) + 1
            if sub_scores:
                best = max(sub_scores.items(), key=lambda x: x[1])[0]
                # add edges for connection (only free ones)
                to_add_edges = []
                for nb in free_nbs:
                    if nb in sub_node_edge_list[best][0]:
                        e = _edge_key(u, nb)
                        if e not in self.used_edges:
                            to_add_edges.append(e)
                if to_add_edges:
                    # commit
                    nodes, edges = sub_node_edge_list[best]
                    nodes = set(nodes); edges = set(edges)
                    nodes.add(u)
                    # final conflict check
                    new_edges_set = set(to_add_edges)
                    if self._edges_conflict(new_edges_set):
                        # skip attaching here (someone else took edges)
                        continue
                    # mark edges
                    self._mark_edges(new_edges_set)
                    for e in new_edges_set:
                        if e in self._edge_exists:
                            edges.add(e)
                    sub_node_edge_list[best] = (nodes, edges)
                    node_to_subs[u].append(best)
                    covered_nodes.add(u)
                    continue
            # else grow small subgraph from u
            nodes_new, edges_new = self._grow_from_seed(u)
            if not edges_new and len(nodes_new) == 1 and not self.cfg.allow_small_coverage:
                continue
            # filter edges_new by free
            edges_new = {e for e in edges_new if e not in self.used_edges}
            # if no free edges and nodes>1 -> skip
            if not edges_new and len(nodes_new) > 1:
                continue
            # final conflict check
            if self._edges_conflict(edges_new):
                continue
            # commit (mark edges)
            self._mark_edges(edges_new)
            sub_node_edge_list.append((nodes_new, edges_new))
            for v in nodes_new:
                node_to_subs[v].append(len(sub_node_edge_list)-1)
                covered_nodes.add(v)
        return sub_node_edge_list

    # ---------- grow from seed ----------
    def _grow_from_seed(self, seed: int) -> Tuple[Set[int], Set[Tuple[int,int]]]:
        # BFS growth using free edges, prefer nodes with higher triangle counts
        selected = {seed}
        edges = set()
        q = deque([seed])
        it = 0
        max_it = self.cfg.max_size * 5
        while q and len(selected) < self.cfg.max_size and it < max_it:
            it += 1
            v = q.popleft()
            candidates = []
            for nb in self.neigh[v]:
                e = _edge_key(v, nb)
                if e not in self._edge_exists or e in self.used_edges or e in edges:
                    continue
                candidates.append(nb)
            # sort by triangle count and degree
            candidates.sort(key=lambda x: (self.triangles.get(x,0), self.degree.get(x,0)), reverse=True)
            for nb in candidates:
                if len(selected) >= self.cfg.max_size:
                    break
                e = _edge_key(v, nb)
                if e in edges or e in self.used_edges:
                    continue
                selected.add(nb); edges.add(e); q.append(nb)
        # optionally drop isolated singletons
        if not edges and len(selected)==1:
            return selected, set()
        return selected, edges

    # ---------- greedy local repair (fast): try removing a few weakest nodes and replacing by best uncovered neighbors ----------
    def _greedy_repair(self, sub_node_edge_list: List[Tuple[Set[int], Set[Tuple[int,int]]]]) -> List[Tuple[Set[int], Set[Tuple[int,int]]]]:
        # compute covered
        covered = set().union(*[nodes for nodes, _ in sub_node_edge_list]) if sub_node_edge_list else set()
        uncovered = set(self.nodes) - covered
        if not uncovered:
            return sub_node_edge_list

        # node->sub map
        node_to_subs = defaultdict(list)
        for idx, (nodes, _) in enumerate(sub_node_edge_list):
            for v in nodes:
                node_to_subs[v].append(idx)

        # iterate subgraphs and attempt limited greedy swaps
        for idx in range(len(sub_node_edge_list)):
            nodes, edges = sub_node_edge_list[idx]
            if not nodes:
                continue
            # compute embeddedness quickly: neighbors inside fraction
            emb = {v: (len(self.neigh[v] & nodes)/ (len(self.neigh[v]) or 1)) for v in nodes}
            weak = sorted(nodes, key=lambda x: emb.get(x,0.0))
            # candidate uncovered neighbors (limit)
            cand_uncovered = set()
            for v in nodes:
                for nb in self.neigh[v]:
                    if nb in uncovered:
                        cand_uncovered.add(nb)
            if not cand_uncovered:
                continue
            # rank candidates by triangle count and free incident edges
            def cand_score(x):
                free_inc = sum(1 for nb in self.neigh[x] if _edge_key(x,nb) not in self.used_edges)
                return (self.triangles.get(x,0), free_inc, self.degree.get(x,0))
            cand_list = sorted(list(cand_uncovered), key=cand_score, reverse=True)[:self.cfg.candidate_pool_size]

            attempts = 0
            while attempts < self.cfg.repair_attempts_per_subgraph and cand_list:
                attempts += 1
                # choose up to max_swap_add top candidates
                to_add = set(cand_list[:self.cfg.max_swap_add])
                # choose to remove up to max_swap_remove weakest
                to_remove = set(weak[:self.cfg.max_swap_remove])
                # feasibility: new_size within limits
                new_size = len(nodes) - len(to_remove) + len(to_add)
                if new_size > self.cfg.max_size:
                    # reduce to_add greedily
                    to_add = set(list(to_add)[:max(1, self.cfg.max_swap_add - 1)])
                    new_size = len(nodes) - len(to_remove) + len(to_add)
                    if new_size > self.cfg.max_size:
                        break
                # check edges: edges among new nodes must be free or belong to freed edges
                freed_edges = {e for e in edges if (e[0] in to_remove or e[1] in to_remove)}
                candidate_new_edges = set()
                feasible = True
                # edges between remain and to_add
                remain = nodes - to_remove
                for a in remain:
                    for b in to_add:
                        if _edge_key(a,b) in self._edge_exists:
                            e = _edge_key(a,b)
                            if e in self.used_edges and e not in freed_edges:
                                feasible = False; break
                            candidate_new_edges.add(e)
                    if not feasible:
                        break
                if not feasible:
                    # drop worst candidate and continue
                    cand_list = cand_list[1:]
                    continue
                # edges among to_add
                for a in to_add:
                    for b in to_add:
                        if a >= b: continue
                        if _edge_key(a,b) in self._edge_exists:
                            e = _edge_key(a,b)
                            if e in self.used_edges and e not in freed_edges:
                                feasible = False; break
                            candidate_new_edges.add(e)
                    if not feasible:
                        break
                if not feasible:
                    cand_list = cand_list[1:]; continue

                # Final conflict check BEFORE committing (ensure no other subgraph used these edges)
                # If candidate_new_edges include any edge marked and not in freed_edges => infeasible
                clash = False
                for e in candidate_new_edges:
                    if e in self.used_edges and e not in freed_edges:
                        clash = True; break
                if clash:
                    cand_list = cand_list[1:]; continue

                # net_gain in covered nodes
                removed_loss = sum(1 for v in to_remove if not any((v in other_nodes) for j,(other_nodes,_) in enumerate(sub_node_edge_list) if j!=idx))
                added_new = sum(1 for v in to_add if v in uncovered)
                net_gain = added_new - removed_loss
                if net_gain <= 0:
                    # not beneficial now
                    break

                # apply swap: unmark freed edges, then mark new edges (atomic sequence)
                # 1) unmark freed edges
                self._unmark_edges(freed_edges)
                # 2) mark candidate_new_edges
                self._mark_edges(candidate_new_edges)

                # rebuild nodes and edges
                new_nodes = (nodes - to_remove) | to_add
                new_edges = {e for e in edges if e[0] in new_nodes and e[1] in new_nodes}
                new_edges |= candidate_new_edges
                # ensure connectivity: simple check by BFS on new_edges
                if new_edges:
                    vis = set()
                    start = next(iter(new_nodes))
                    q = deque([start]); vis.add(start)
                    while q:
                        x = q.popleft()
                        for y in self.neigh[x]:
                            if y in new_nodes and _edge_key(x,y) in new_edges and y not in vis:
                                vis.add(y); q.append(y)
                    if len(vis) < len(new_nodes):
                        # re-evaluate: keep largest component only
                        comp = vis
                        new_nodes = new_nodes & comp
                        new_edges = {e for e in new_edges if e[0] in new_nodes and e[1] in new_nodes}
                # commit
                sub_node_edge_list[idx] = (new_nodes, new_edges)
                # update local maps
                nodes, edges = new_nodes, new_edges
                covered |= new_nodes
                uncovered = set(self.nodes) - covered
                # recompute weak / cand_list
                emb = {v: (len(self.neigh[v] & nodes) / (len(self.neigh[v]) or 1)) for v in nodes}
                weak = sorted(nodes, key=lambda x: emb.get(x,0.0))
                weak = weak
                cand_uncovered = set()
                for v in nodes:
                    for nb in self.neigh[v]:
                        if nb in uncovered:
                            cand_uncovered.add(nb)
                cand_list = sorted(list(cand_uncovered), key=lambda x: (self.triangles.get(x,0), sum(1 for nb in self.neigh[x] if _edge_key(x,nb) not in self.used_edges)), reverse=True)[:self.cfg.candidate_pool_size]
                # break to let next subgraph be considered
                break
        return sub_node_edge_list

    # ---------- main method: generate subgraphs (returns list of networkx.Graph) ----------
    def generate_multiple_subgraphs(self) -> List[nx.Graph]:
        t0 = time.time()
        results_nodes_edges: List[Tuple[Set[int], Set[Tuple[int,int]]]] = []
        attempts = 0
        max_attempts = max(200, self.cfg.num_subgraphs * 20)

        while len(results_nodes_edges) < self.cfg.num_subgraphs and attempts < max_attempts:
            attempts += 1
            start = None
            if self.cfg.use_truss:
                # optional truss-based seed: pick node with high triangle count and free edges
                start = max(self.nodes, key=lambda v: (self.triangles.get(v,0), self.degree.get(v,0)))
            else:
                start = self._select_start_node()
            nodes_sel, edges_sel = self.generate_dense_candidate(start)
            if len(nodes_sel) < self.cfg.min_size:
                continue
            # filter out edges that are already used
            edges_free = {e for e in edges_sel if e not in self.used_edges}
            if not edges_free and len(nodes_sel) > 1:
                continue
            # compute overlap ratio
            overlap = sum(1 for e in edges_sel if e in self.used_edges)
            if edges_sel:
                ratio = overlap / len(edges_sel)
            else:
                ratio = 0.0
            if ratio > self.cfg.max_edge_overlap_ratio:
                continue
            # reduce to nodes incident to free edges
            nodes_inc = set()
            for (a,b) in edges_free:
                nodes_inc.add(a); nodes_inc.add(b)
            if not nodes_inc:
                if self.cfg.allow_small_coverage:
                    # allow singletons
                    results_nodes_edges.append(({start}, set()))
                    continue
                else:
                    continue
            # keep only largest connected component of nodes_inc w.r.t. free edges
            # build adjacency (lightweight)
            adj = defaultdict(list)
            for a,b in edges_free:
                adj[a].append(b); adj[b].append(a)
            # BFS from arbitrary node
            root = next(iter(nodes_inc))
            comp = set([root])
            q = deque([root])
            while q:
                v = q.popleft()
                for nb in adj[v]:
                    if nb not in comp:
                        comp.add(nb); q.append(nb)
            nodes_final = comp
            edges_final = {e for e in edges_free if e[0] in comp and e[1] in comp}

            # Final conflict check (safety)
            if self._edges_conflict(edges_final):
                # conflict found — skip this candidate
                continue

            # check size constraint
            if len(nodes_final) < self.cfg.min_size:
                continue
            # commit edges as used (atomically)
            self._mark_edges(edges_final)
            results_nodes_edges.append((nodes_final, edges_final))

        # coverage phase
        results_nodes_edges = self._attach_uncovered(results_nodes_edges)
        # greedy repair phase
        results_nodes_edges = self._greedy_repair(results_nodes_edges)
        # final coverage attempt
        results_nodes_edges = self._attach_uncovered(results_nodes_edges)

        # Final audit: check that no edge is assigned to more than one subgraph
        conflicts = self._audit_edge_exclusivity(results_nodes_edges)
        if conflicts:
            # print summary of conflicts
            print("WARNING: edge exclusivity violations detected! Number of conflicting edges:", len(conflicts))
            for e, subs in conflicts[:10]:  # show up to first 10 conflicts
                print(f" Edge {e} used in subgraphs {subs}")
            if self.cfg.strict_edge_exclusivity_audit:
                raise RuntimeError("Edge exclusivity audit failed: conflicts detected.")
            # otherwise continue but keep used_edges consistent (we trust used_edges marking)

        # build final networkx.Graph objects
        graphs = []
        for nodes, edges in results_nodes_edges:
            Gsub = nx.Graph()
            Gsub.add_nodes_from(nodes)
            # add edges that exist in original graph
            Gsub.add_edges_from([e for e in edges if e in self._edge_exists])
            graphs.append(Gsub)

        t1 = time.time()
        covered = set().union(*[set(g.nodes()) for g in graphs]) if graphs else set()
        print(f"[fastgen] produced {len(graphs)} subgraphs; covered {len(covered)}/{self.n} nodes; time {t1-t0:.2f}s")
        return graphs

# -------------------- QUICK DEMO --------------------

if __name__ == "__main__":
    # Example usage and quick benchmark
    import sys

    G = ConnectedRandGraph(number_of_nodes=1000,
                           number_of_edges=100000,
                           weights='ones',
                           seed=42)
    cfg = FastConfig(max_size=10, min_size=6, num_subgraphs=30,
                     boundary_sample_limit=1000, distance_bfs_threshold=100,
                     use_truss=True,
                     max_swap_add=30, max_swap_remove=30,
                     candidate_pool_size=10, repair_attempts_per_subgraph=30,
                     random_seed=None)
    gen = FastDenseSubgraphGenerator(G, cfg)
    sgs = gen.generate_multiple_subgraphs()
    covered = set().union(*[set(sg.nodes()) for sg in sgs]) if sgs else set()
    uncovered = set(G.nodes()) - covered
    print(f"Generated {len(sgs)} subgraphs")
    print(f"Covered vertices: {len(covered)} / {G.number_of_nodes()}")
    print(f"Uncovered vertices: {len(uncovered)}")

    total_edges = 0
    for i, sg in enumerate(sgs, 1):
        print(f"SG{i}: nodes={sg.number_of_nodes()}, edges={sg.number_of_edges()}, density={nx.density(sg):.4f}")
        total_edges += sg.number_of_edges()

    print(total_edges)

    pe = PartitionEfficiency(G, sgs)
    pe.efficiency(sample_threshold=200, sample_pairs=800, random_seed=1)
    pe.plot_statistic(bins=25, figsize=(14, 10), title="Demo partition statistics")
