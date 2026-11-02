from Generators.GraphGenerator import ConnectedRandGraph

from typing import Iterable, List, Set, Tuple, Dict, Optional
import networkx as nx
from collections import defaultdict



def _precompute_neighbors(G: nx.Graph) -> Dict[int, Set[int]]:
    return {n: set(G.neighbors(n)) for n in G.nodes()}

def _edge_weight(G: nx.Graph, u: int, v: int, weight_attr: Optional[str]) -> float:
    if weight_attr is None:
        return 1.0
    w = G.get_edge_data(u, v, default={}).get(weight_attr, None)
    return float(w) if w is not None else 1.0

def _connections_to_set(G: nx.Graph, node: int, nodes_set: Set[int],
                        neighbors_map: Dict[int, Set[int]],
                        weight_attr: Optional[str]) -> float:
    """Суммарная 'сила' соединений node → nodes_set (количество или суммарный вес)."""
    neigh = neighbors_map.get(node, set())
    common = neigh & nodes_set
    if not common:
        return 0.0
    if weight_attr is None:
        return float(len(common))
    s = 0.0
    for v in common:
        s += _edge_weight(G, node, v, weight_attr)
    return s

def _grow_subgraph_from_seed(G: nx.Graph, seed_nodes: Set[int], allowed_pool: Set[int],
                             neighbors_map: Dict[int, Set[int]], max_size: int,
                             weight_attr: Optional[str]) -> Set[int]:
    """
    Жадный рост: добавляем по одному узлу, который максимизирует связи с selected.
    Кандидаты — соседи selected, фильтруемые allowed_pool (если задан).
    """
    selected = set(seed_nodes)
    boundary = set()
    for u in seed_nodes:
        boundary.update(neighbors_map.get(u, set()))
    boundary.difference_update(selected)

    while len(selected) < max_size and boundary:
        best_node = None
        best_score = -1.0
        # перебираем boundary; это можно оптимизировать heap-структурой, если нужно
        for cand in list(boundary):
            if allowed_pool and (cand not in allowed_pool) and (cand not in selected):
                continue
            score = _connections_to_set(G, cand, selected, neighbors_map, weight_attr)
            # небольшой бонус для кандидатов из allowed_pool — дает приоритет покрытию нужной области
            if allowed_pool and cand in allowed_pool:
                score += 1e-6
            if score > best_score:
                best_score = score
                best_node = cand
        if best_node is None or best_score <= 0.0:
            break
        selected.add(best_node)
        for nb in neighbors_map.get(best_node, set()):
            if nb not in selected:
                boundary.add(nb)
        boundary.discard(best_node)

    return selected

def _build_one_when_small(G: nx.Graph, center: int, base_neighbors: Set[int],
                          neighbors_map: Dict[int, Set[int]],
                          max_size: int, weight_attr: Optional[str]) -> Set[int]:
    """
    Если базовый набор + центр умещается в max_size — дополняем жадно ближайшими вершинами.
    """
    selected = set(base_neighbors) | {center}
    if len(selected) >= max_size:
        return set(list(selected)[:max_size])

    boundary = set()
    for u in list(selected):
        boundary.update(neighbors_map.get(u, set()))
    boundary.difference_update(selected)

    while len(selected) < max_size and boundary:
        best_node = None
        best_score = -1.0
        for cand in list(boundary):
            score = _connections_to_set(G, cand, selected, neighbors_map, weight_attr)
            if score > best_score:
                best_score = score
                best_node = cand
        if best_node is None or best_score <= 0.0:
            break
        selected.add(best_node)
        for nb in neighbors_map.get(best_node, set()):
            if nb not in selected:
                boundary.add(nb)
        boundary.discard(best_node)

    return selected

def generate_subgraphs_for_vertex(G: nx.Graph, v: int, max_size: int,
                                  neighbors_map: Optional[Dict[int, Set[int]]] = None,
                                  weight_attr: Optional[str] = None,
                                  include_center: bool = True) -> List[nx.Graph]:
    """
    Для одной вершины v строит список подграфов (nx.Graph), каждый размером ≤ max_size.
    Параметры:
      - neighbors_map: если передан — используется (для ускорения большого количества v).
      - weight_attr: имя атрибута ребра для суммирования весов (опционально).
      - include_center: если True, в каждый подграф включается сама вершина v.
    Возвращает список networkx.Graph (копии подграфов).
    """
    if neighbors_map is None:
        neighbors_map = _precompute_neighbors(G)

    if v not in G:
        return []

    base_neighbors = set(neighbors_map.get(v, set()))
    center_set = {v} if include_center else set()

    # тривиальный случай: степень 0
    if not base_neighbors:
        return [G.subgraph(center_set).copy()]

    # случай помещается в один подграф
    if len(base_neighbors) + len(center_set) <= max_size:
        selected = _build_one_when_small(G, v, base_neighbors, neighbors_map, max_size, weight_attr)
        return [G.subgraph(selected).copy()]

    # иначе — разбиваем базовый набор на несколько подграфов, чтобы покрыть всех соседей
    uncovered = set(base_neighbors)
    result_subgraphs: List[nx.Graph] = []

    # pool_of_interest = base_neighbors + их 1-hop соседство (для расширения кандидатов)
    pool_of_interest = set(uncovered)
    for node in list(uncovered):
        pool_of_interest.update(neighbors_map.get(node, set()))
    pool_of_interest.difference_update(center_set)

    degree = dict(G.degree())

    while uncovered:
        # seed — непокрытая вершина с максимальной степенью (можно сменить на более сложную метрику)
        seed = max(uncovered, key=lambda x: degree.get(x, 0))
        seed_nodes = set(center_set) | {seed}

        # allowed_pool — хотим покрыть uncovered, но допускаем ближайшее окружение для связности
        allowed_pool = set(uncovered) | (pool_of_interest - set(center_set))

        selected = _grow_subgraph_from_seed(G, seed_nodes, allowed_pool, neighbors_map, max_size, weight_attr)

        # если выбранный не покрывает ничего из uncovered (крайний случай), добавляем самые "важные" непокрытые
        covered_now = set(selected) & base_neighbors
        if not covered_now:
            need = max_size - len(selected)
            top_take = sorted(uncovered, key=lambda x: -degree.get(x, 0))[:need]
            for t in top_take:
                selected.add(t)
            covered_now = set(selected) & base_neighbors

        uncovered.difference_update(covered_now)

        if len(selected) > max_size:
            selected = set(list(selected)[:max_size])

        result_subgraphs.append(G.subgraph(selected).copy())

    return result_subgraphs

def generate_all_vertex_subgraphs(G: nx.Graph, vertices: Iterable[int],
                                  max_size: int, weight_attr: Optional[str] = None,
                                  include_center: bool = True) -> List[nx.Graph]:
    """
    Для всех вершин из vertices возвращает общий список всех подграфов.
    """
    neighbors_map = _precompute_neighbors(G)
    all_subs: List[nx.Graph] = []
    for v in vertices:
        subs = generate_subgraphs_for_vertex(G, v, max_size, neighbors_map, weight_attr, include_center)
        all_subs.extend(subs)
    return all_subs

# Пример быстрой проверки:
if __name__ == "__main__":

    G = ConnectedRandGraph(number_of_nodes=60,
                           number_of_edges=800,
                           weights='ones',
                           seed=1000)

    verts = list(G.nodes())[:20]
    subs = generate_all_vertex_subgraphs(G, verts, max_size=15, weight_attr='w', include_center=True)
    print("Generated subgraphs:", len(subs))
    for i, sg in enumerate(subs[:10], 1):
        print(i, "nodes", sg.number_of_nodes(), "edges", sg.number_of_edges(), "density", nx.density(sg))
