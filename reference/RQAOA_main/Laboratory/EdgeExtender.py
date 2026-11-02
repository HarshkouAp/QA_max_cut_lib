from typing import Iterable, List, Set, Tuple, Dict, Optional
import networkx as nx
import random

from Generators.GraphGenerator import ConnectedRandGraph


def _precompute_neighbors(G: nx.Graph) -> Dict[int, Set[int]]:
    return {n: set(G.neighbors(n)) for n in G.nodes()}

def _edge_weight(G: nx.Graph, u: int, v: int, weight_attr: Optional[str]) -> float:
    if weight_attr is None:
        return 1.0
    w = G.get_edge_data(u, v, default={}).get(weight_attr, None)
    return float(w) if w is not None else 1.0

def _connections_to_set(G: nx.Graph, node: int, nodes_set: Set[int], neighbors_map: Dict[int, Set[int]],
                        weight_attr: Optional[str]) -> float:
    """Количество (или суммарный вес) рёбер между node и nodes_set."""
    # iterate over smaller set for efficiency
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

def _grow_subgraph_for_seed(G: nx.Graph, seed_nodes: Set[int], allowed_pool: Set[int],
                            neighbors_map: Dict[int, Set[int]], max_size: int,
                            weight_attr: Optional[str]) -> Set[int]:
    """
    Жадно растём от seed_nodes добавляя вершины, которые имеют наибольшую
    сумму связей (весов) к текущему выбранному множеству. Кандидаты —
    соседи выбранных вершин, ограниченные allowed_pool.
    Гарантируем связность (в процессе роста выбираем только смежные вершины).
    """
    selected = set(seed_nodes)
    # boundary: кандидаты, смежные к selected, но не входящие в selected
    boundary = set()
    for u in seed_nodes:
        boundary.update(neighbors_map.get(u, set()))
    boundary.difference_update(selected)

    # priority: we will recompute score each iteration (deg to selected)
    while len(selected) < max_size and boundary:
        # оценить кандидатов
        best_node = None
        best_score = -1.0
        for cand in list(boundary):
            if cand not in allowed_pool and cand not in selected:
                # допускаем кандидатов из allowed_pool преимущественно,
                # но если allowed_pool пуст — допускаем любые соседние вершины
                continue
            score = _connections_to_set(G, cand, selected, neighbors_map, weight_attr)
            # небольшой бонус если cand принадлежит allowed_pool (чтобы быстрее покрывать исходный набор)
            if cand in allowed_pool:
                score += 0.001
            if score > best_score:
                best_score = score
                best_node = cand
        if best_node is None or best_score <= 0.0:
            # если нет кандидатов с положительной связностью, можем добавить
            # вершин из allowed_pool (не обязательно смежных) с наибольшей общей степенью
            # но это ухудшит связность — выходим
            break
        selected.add(best_node)
        # обновить границу
        for nb in neighbors_map.get(best_node, set()):
            if nb not in selected:
                boundary.add(nb)
        boundary.discard(best_node)

    return selected

def _build_one_subgraph_when_small(G: nx.Graph, k: int, j: int, base_neighbors: Set[int],
                                  neighbors_map: Dict[int, Set[int]],
                                  max_size: int, weight_attr: Optional[str]) -> Set[int]:
    """
    Случай: базовый набор (включая endpoints) умещается в max_size.
    Дополняем из ближайшего окружения вершинами, максимально увеличивающими
    число связей с уже выбранными (жадно), до достижения max_size.
    """
    selected = set(base_neighbors) | {k, j}
    if len(selected) >= max_size:
        return set(list(selected)[:max_size])  # защитный обрез в редком случае

    # кандидаты: соседи текущего множества, исключая уже выбранные
    boundary = set()
    for u in list(selected):
        boundary.update(neighbors_map.get(u, set()))
    boundary.difference_update(selected)

    while len(selected) < max_size and boundary:
        # выбрать вершину с макс. связями в selected (учитывая веса при наличии)
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
        # обновить boundary
        for nb in neighbors_map.get(best_node, set()):
            if nb not in selected:
                boundary.add(nb)
        boundary.discard(best_node)

    return selected

def generate_subgraphs_for_edge(G: nx.Graph, edge: Tuple[int, int], max_size: int,
                                neighbors_map: Optional[Dict[int, Set[int]]] = None,
                                weight_attr: Optional[str] = None) -> List[nx.Graph]:
    """
    Для одного ребра (k, j) строит список подграфов (nx.Graph),
    каждый ≤ max_size, покрывающих соседей k и j (и их ближайшее окружение).
    Параметры:
      - weight_attr: если указано, используется суммарный вес при ранжировании.
    Возвращает: список подграфов networkx (subgraph.copy()).
    """
    if neighbors_map is None:
        neighbors_map = _precompute_neighbors(G)

    k, j = edge
    if k not in G or j not in G:
        return []

    # базовый набор соседей (исключаем сами k и j из base_neighbors)
    nk = neighbors_map.get(k, set()) - {j}
    nj = neighbors_map.get(j, set()) - {k}
    base_neighbors = set(nk | nj)
    # включаем сами концы ребра в каждый подграф
    # если нужно исключить концы — измените ниже
    endpoint_set = {k, j}

    # Быстрая обработка: если помещаемся в один подграф
    if len(base_neighbors) + len(endpoint_set) <= max_size:
        selected_nodes = _build_one_subgraph_when_small(G, k, j, base_neighbors, neighbors_map, max_size, weight_attr)
        return [G.subgraph(selected_nodes).copy()]

    # Иначе — нужно разбить base_neighbors на несколько подграфов,
    # чтобы покрыть всех соседей (coverage). Подход: жадное покрытие
    # — пока остались непокрытые base_neighbors, создаём подграф, стартуя
    # с seed из непокрытых (наиболее 'центральная'), и растём до max_size,
    # отдавая приоритет добавлению непокрытых соседей.
    uncovered = set(base_neighbors)
    result_subgraphs = []

    # Доп. пул кандидатов: ближайшее окружение base_neighbors (1-hop)
    pool_of_interest = set(uncovered)
    for v in list(uncovered):
        pool_of_interest.update(neighbors_map.get(v, set()))
    # уберём сами концы
    pool_of_interest.discard(k)
    pool_of_interest.discard(j)

    # предвычислим сортировку по степени для выбора seed (больше шансов покрыть много вершин)
    degree = dict(G.degree())
    # while uncovered not empty
    while uncovered:
        # выбрать seed — непокрытая вершина с наибольшей степени (внутри base_neighbors предпочтение)
        seed = max(uncovered, key=lambda x: degree.get(x, 0))
        # стартовый набор включает endpoints + seed
        seed_nodes = set(endpoint_set) | {seed}

        # ограничим allowed_pool при росте: стремимся покрыть uncovered, но допускаем ближайшее окружение
        allowed_pool = set(uncovered) | (pool_of_interest - set(endpoint_set))
        # разрешаем кандидатов быть любыми из allowed_pool (чтобы обеспечить связность)
        selected = _grow_subgraph_for_seed(G, seed_nodes, allowed_pool, neighbors_map, max_size, weight_attr)

        # если в выбранном нет ни одной непокрытой вершины (редкий случай) — вручную добавим несколько непокрытых
        covered_now = set(selected) & base_neighbors
        if not covered_now:
            # добавим top-(max_size - len(selected)) непокрытых по степени, не обязательно смежных
            need = max_size - len(selected)
            take = sorted(uncovered, key=lambda x: -degree.get(x, 0))[:need]
            for t in take:
                selected.add(t)
            covered_now = set(selected) & base_neighbors

        # отметим покрытые
        uncovered.difference_update(covered_now)
        # сохраняем подграф (приводим к ограничению max_size на всякий случай)
        if len(selected) > max_size:
            selected = set(list(selected)[:max_size])
        result_subgraphs.append(G.subgraph(selected).copy())

    return result_subgraphs

def generate_all_edge_subgraphs(G: nx.Graph, edges: Iterable[Tuple[int, int]],
                                max_size: int, weight_attr: Optional[str] = None) -> List[nx.Graph]:
    """
    Для всех edges возвращает список всех получившихся подграфов (concatenated list).
    (можно группировать/индексировать результаты по ребрам на уровне вызывающего кода).
    """
    neighbors_map = _precompute_neighbors(G)
    all_subs: List[nx.Graph] = []
    for e in edges:
        subs = generate_subgraphs_for_edge(G, e, max_size, neighbors_map, weight_attr)
        all_subs.extend(subs)
    return all_subs



if __name__ == "__main__":

    G = ConnectedRandGraph(number_of_nodes=100,
                           number_of_edges=3000,
                           weights='random',
                           seed=None)

    edges = list(G.edges())
    random.shuffle(edges)
    edges = edges[:10]

    subgraphs = generate_all_edge_subgraphs(G, edges, max_size=16, weight_attr='w')
    print("Total subgraphs:", len(subgraphs))
    for i, sg in enumerate(subgraphs[:10], 1):
        print(i, "nodes:", sg.number_of_nodes(), "edges:", sg.number_of_edges(), "density:", nx.density(sg))
