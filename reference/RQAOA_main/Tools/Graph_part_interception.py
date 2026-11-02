from Generators.GraphGenerator import ConnectedRandGraph

import networkx as nx
import heapq
import random
import numpy as np
from collections import defaultdict, deque

def _edge_key(u, v):
    return (u, v) if u <= v else (v, u)

def build_internal_structures(G):
    """
    Построить компактное представление:
      - edge_u, edge_v, edge_w : списки длины m
      - edge_id: dict (u,v)->eid
      - adj: dict node -> list of (nbr, eid)
    """
    edge_u, edge_v, edge_w = [], [], []
    edge_id = {}
    adj = defaultdict(list)
    m = 0
    for u, v, data in G.edges(data=True):
        k = _edge_key(u, v)
        if k in edge_id:
            continue
        w = data.get('weight', 1.0)
        edge_id[k] = m
        edge_u.append(k[0]); edge_v.append(k[1]); edge_w.append(w)
        adj[k[0]].append((k[1], m))
        adj[k[1]].append((k[0], m))
        m += 1
    return edge_u, edge_v, edge_w, edge_id, adj

class LazyMaxHeap:
    """Max-heap с lazy-обновлениями: в heap храним (-score, counter, node).
       Для проверки актуальности перед извлечением пересчитываем текущ скор и сравниваем."""
    def __init__(self):
        self.heap = []
        self.counter = 0

    def push(self, score, node):
        # score — числовой (чем больше — тем лучше)
        heapq.heappush(self.heap, (-score, self.counter, node))
        self.counter += 1

    def pop_valid(self, valid_score_fn):
        """
        Достаёт элемент с наибольшим score, проверяя через valid_score_fn(node) -> current_score.
        Если score устарел, пропускаем (lazy) и продолжаем.
        Возвращает (node, current_score) или (None, None) если пусто.
        """
        while self.heap:
            neg_score, _, node = heapq.heappop(self.heap)
            current = valid_score_fn(node)
            if current is None:
                continue
            # отрицательная запись в куче — сравним с current
            if abs(current + neg_score) < 1e-12 or current > -neg_score - 1e-12:
                # приемлемо (heap может содержать чуть меньший скор из-за округления) — возвращаем
                return node, current
            # иначе — устарело, пропускаем (и возможно в будущем новый актуальный скор будет в куче)
        return None, None

    def push_if_possible(self, score, node):
        self.push(score, node)

def partition_graph_max_overlap_fast(
    G, M, min_size=4, seed_strategy='max_overlap', random_state=None,
    w_internal=1.0, w_overlap=1.0
):
    """
    Быстрая версия partition_graph_max_overlap для больших графов.
    Вход:
      - G: networkx.Graph (независимо взвешенный через 'weight')
      - M: максимальное число вершин в подграфе
      - min_size: порог для постобработки малых кластеров
      - seed_strategy: 'max_overlap' (рекомендуем) или 'random'
      - w_internal, w_overlap: веса для комбинированной оценки
    Возвращает:
      - list of networkx.Graph — каждый граф содержит только рёбра, закреплённые за этим кластером.
    """
    rnd = random.Random(random_state)

    # --- постройка внутренних структур ---
    edge_u, edge_v, edge_w, edge_id, adj = build_internal_structures(G)
    m = len(edge_u)
    # булев флаг нераспределённости ребра
    unassigned = [True] * m
    # суммарный вес нераспр. инцидентных рёбер у вершины
    node_unassigned_weight = defaultdict(float)
    for eid in range(m):
        node_unassigned_weight[edge_u[eid]] += edge_w[eid]
        node_unassigned_weight[edge_v[eid]] += edge_w[eid]

    # counts of how many clusters each vertex participates
    vertex_cluster_count = defaultdict(int)

    # вспомог: активные вершины (имеют хотя бы одно нераспр. ребро)
    def active_vertices():
        return {u for u, w in node_unassigned_weight.items() if w > 0.0}

    # хипы для выбора seed (используем lazy-heap)
    used_heap = LazyMaxHeap()   # вершины с vertex_cluster_count > 0
    fresh_heap = LazyMaxHeap()  # другие активные вершины

    # функция для получения текущего скорa для узла
    def current_score(node):
        w_inc = node_unassigned_weight.get(node, 0.0)
        ov = vertex_cluster_count.get(node, 0)
        # node может временно исчезнуть (нет инцидентных ребер) -> None
        if w_inc <= 0.0 and ov == 0:
            return None
        return w_overlap * ov + w_internal * w_inc

    # инициализация свежих узлов
    for node, w in node_unassigned_weight.items():
        if w > 0.0:
            sc = current_score(node)
            fresh_heap.push(sc, node)

    clusters_edges = []   # list of set(eid)
    clusters_verts = []   # list of set(nodes)

    # --- основной цикл: пока есть нераспр. ребра ---
    remaining_edges = m
    while remaining_edges > 0:
        # обновляем и выбираем seed
        # предпочтение — вершина, уже участвующая в кластерах (used) и активная
        seed = None
        if seed_strategy == 'random':
            active = list(active_vertices())
            if not active:
                break
            seed = rnd.choice(active)
        else:
            # попробуем извлечь из used_heap
            def valid_score_used(n): return current_score(n) if vertex_cluster_count.get(n, 0) > 0 else None
            u_node, u_score = used_heap.pop_valid(valid_score_used)
            if u_node is not None:
                seed = u_node
            else:
                # извлекаем лучшую свежую
                def valid_score_fresh(n): return current_score(n)
                f_node, f_score = fresh_heap.pop_valid(valid_score_fresh)
                if f_node is None:
                    # если нет активных вершин — выходим
                    break
                seed = f_node

        # Если выбранный seed не активен (no unassigned incident edges) — пропускаем
        if node_unassigned_weight.get(seed, 0.0) <= 0.0 and vertex_cluster_count.get(seed, 0) == 0:
            # поиск произвольного нераспр. ребра
            # find any unassigned edge
            found_eid = None
            for eid in range(m):
                if unassigned[eid]:
                    found_eid = eid
                    break
            if found_eid is None:
                break
            seed = edge_u[found_eid]

        # --- локальное расширение вокруг seed ---
        # cluster: set of nodes
        cluster_nodes = set([seed])
        # considered — узлы, которые мы уже исследовали/учитывали как кандидаты
        considered = set()
        # текущий слой — начальная выбранная группа соседей seed
        # соберём соседей seed с нераспр. ребрами
        neighbors = [nbr for (nbr, eid) in adj[seed] if unassigned[eid]]
        if not neighbors:
            # fallback — взять одно ребро инцидентное seed
            any_eid = None
            for (nbr, eid) in adj[seed]:
                if unassigned[eid]:
                    any_eid = eid; break
            if any_eid is None:
                # попытаемся найти любое нераспр. ребро в графе
                found_eid = None
                for eid in range(m):
                    if unassigned[eid]:
                        found_eid = eid; break
                if found_eid is None:
                    break
                neighbors = [edge_u[found_eid], edge_v[found_eid]]
            else:
                neighbors = [edge_u[any_eid] if edge_v[any_eid]==seed else edge_v[any_eid]]
        considered.update(neighbors)

        # helper: compute combined score for a candidate v relative to ref_set
        def candidate_score(v, ref_set):
            # internal: суммарный вес нераспр. ребер v -> ref_set
            internal = 0.0
            for (nbr, eid) in adj[v]:
                if nbr in ref_set and unassigned[eid]:
                    internal += edge_w[eid]
            ov = vertex_cluster_count.get(v, 0)
            return w_internal * internal + w_overlap * ov

        # первый шаг: выбрать до needed вершин из neighbors
        needed = M - 1
        if neighbors:
            scored = [(candidate_score(v, set(neighbors)), v) for v in neighbors]
            scored.sort(reverse=True)
            to_add = [v for (_, v) in scored[:needed]]
            cluster_nodes.update(to_add)

        # расширение по слоям
        current_layer = set([v for v in cluster_nodes if v != seed])
        ref_set = set(neighbors)  # union of considered neighbor layers (start)
        while len(cluster_nodes) < M and current_layer:
            # собрать новые кандидаты: соседи вершин current_layer, у которых есть нераспр. ребра к current_component
            new_candidates = set()
            for u in current_layer:
                for (nbr, eid) in adj[u]:
                    if not unassigned[eid]:
                        continue
                    if nbr in cluster_nodes or nbr in considered:
                        continue
                    new_candidates.add(nbr)
            if not new_candidates:
                break
            # рассчитать score относительно ref_set ∪ new_candidates (как вы просили в оригинале)
            ref = ref_set | new_candidates
            scored2 = [(candidate_score(v, ref), v) for v in new_candidates]
            scored2.sort(reverse=True)
            need = M - len(cluster_nodes)
            to_add2 = [v for (_, v) in scored2[:need]]
            cluster_nodes.update(to_add2)
            ref_set |= new_candidates
            considered |= new_candidates
            current_layer = set(to_add2)

        # --- назначаем рёбра, инцидентные вершинам cluster_nodes, и которые ещё unassigned ---
        assigned_eids = set()
        for v in cluster_nodes:
            for (nbr, eid) in adj[v]:
                if not unassigned[eid]:
                    continue
                if nbr in cluster_nodes:
                    assigned_eids.add(eid)
        if not assigned_eids:
            # fallback: назначаем 1 ребро инцидентное seed
            for (nbr, eid) in adj[seed]:
                if unassigned[eid]:
                    assigned_eids.add(eid)
                    break

        # пометить рёбра как присвоенные и обновить node_unassigned_weight
        for eid in assigned_eids:
            if not unassigned[eid]:
                continue
            unassigned[eid] = False
            remaining_edges -= 1
            u = edge_u[eid]; v = edge_v[eid]; w = edge_w[eid]
            node_unassigned_weight[u] -= w
            node_unassigned_weight[v] -= w
            # push updated nodes в heap (lazy)
            sc_u = current_score(u)
            if sc_u is not None:
                if vertex_cluster_count.get(u, 0) > 0:
                    used_heap.push(sc_u, u)
                else:
                    fresh_heap.push(sc_u, u)
            sc_v = current_score(v)
            if sc_v is not None:
                if vertex_cluster_count.get(v, 0) > 0:
                    used_heap.push(sc_v, v)
                else:
                    fresh_heap.push(sc_v, v)

        # сформируем фактические множества вершин из назначенных рёбер (это вершины кластера)
        real_nodes = set()
        for eid in assigned_eids:
            real_nodes.add(edge_u[eid]); real_nodes.add(edge_v[eid])
        clusters_edges.append(assigned_eids)
        clusters_verts.append(real_nodes)
        for node in real_nodes:
            vertex_cluster_count[node] += 1
            # обновим и положим в used_heap
            sc = current_score(node)
            if sc is not None:
                used_heap.push(sc, node)

    # --- Постобработка: дополнение кластеров до M, максимизируя overlap и внутренний gain ---
    # перебираем кластеры в порядке убывания размера
    cluster_indices = sorted(range(len(clusters_edges)), key=lambda i: len(clusters_verts[i]), reverse=True)
    for idx in cluster_indices:
        while len(clusters_verts[idx]) < M:
            # кандидаты: вершины, у которых есть unassigned ребро к текущему кластеру
            candidates = {}
            for v in list(clusters_verts[idx]):
                for (nbr, eid) in adj[v]:
                    if not unassigned[eid]:
                        continue
                    if nbr in clusters_verts[idx]:
                        continue
                    # вычислим прирост внутренних ребёр если добавить nbr
                    # собираем нераспр. ребра (nbr <-> existing cluster vertex)
                    gain_edges = []
                    gain_weight = 0.0
                    for (nbr2, eid2) in adj[nbr]:
                        if not unassigned[eid2]:
                            continue
                        if nbr2 in clusters_verts[idx]:
                            gain_edges.append(eid2)
                            gain_weight += edge_w[eid2]
                    if gain_edges:
                        # score combines gain_weight и overlap
                        ov = vertex_cluster_count.get(nbr, 0)
                        total_score = w_internal * gain_weight + w_overlap * ov
                        # сохраняем лучший вариант для данного nbr
                        if nbr not in candidates or candidates[nbr][0] < total_score:
                            candidates[nbr] = (total_score, gain_edges)
            if not candidates:
                break
            # выберем лучшее добавление
            best_nbr, (best_score, best_edges) = max(candidates.items(), key=lambda kv: kv[1][0])
            # назначим ребра best_edges
            for eid in best_edges:
                if not unassigned[eid]:
                    continue
                unassigned[eid] = False
                remaining_edges -= 1
                u = edge_u[eid]; v = edge_v[eid]; w = edge_w[eid]
                node_unassigned_weight[u] -= w
                node_unassigned_weight[v] -= w
            # добавим вершину в кластер
            clusters_verts[idx].add(best_nbr)
            clusters_edges[idx].update(best_edges)
            vertex_cluster_count[best_nbr] += 1
            # обновим хипы для этой вершины и её соседей
            sc = current_score(best_nbr)
            if sc is not None:
                used_heap.push(sc, best_nbr)

    # --- удаление пустых и формирование networkx.Graph результатов ---
    result = []
    for eds in clusters_edges:
        if not eds:
            continue
        H = nx.Graph()
        for eid in eds:
            u = edge_u[eid]; v = edge_v[eid]; w = edge_w[eid]
            H.add_edge(u, v, weight=w)
        result.append(H)
    return result



if __name__ == "__main__":
    G = ConnectedRandGraph(number_of_nodes=1000,
                           number_of_edges=100000,
                           weights='ones')


    subgraphs = partition_graph_max_overlap_fast(G, M=10, min_size=8,
                                           seed_strategy='max_overlap', random_state=42,
                                           w_internal=1.0, w_overlap=2.0)


    print(len(subgraphs))
