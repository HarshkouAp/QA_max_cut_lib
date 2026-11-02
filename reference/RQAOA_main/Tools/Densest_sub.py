import random
import networkx as nx


def densest_k_subgraph_fast(
    G: nx.Graph,
    k: int,
    n_restarts: int = 20,
    local_iter: int = 500,
    candidate_pool: int = 64,
    seed: int = 42,
):
    """
    Быстрая эвристика для поиска индуцированного подграфа на k вершинах
    с максимальным числом внутренних рёбер.

    Параметры:
        G              : исходный неориентированный граф NetworkX
        k              : требуемое число вершин подграфа
        n_restarts     : число перезапусков эвристики
        local_iter     : максимум локальных swap-улучшений на один запуск
        candidate_pool : число лучших кандидатов на удаление/добавление
        seed           : seed для воспроизводимости

    Возвращает:
        H          : найденный индуцированный подграф
        best_nodes : список выбранных вершин
        best_edges : число внутренних рёбер в H
    """

    if G.is_directed():
        raise ValueError("Функция рассчитана на неориентированный граф nx.Graph.")

    N = G.number_of_nodes()

    if not (0 < k <= N):
        raise ValueError(f"Нужно 0 < k <= N, но получено k={k}, N={N}.")

    rng = random.Random(seed)

    nodes = list(G.nodes())
    node_to_id = {node: i for i, node in enumerate(nodes)}

    # Переводим граф во внутреннее представление через множества соседей.
    # Это быстрее, чем постоянно обращаться к NetworkX внутри циклов.
    adj = [set() for _ in range(N)]

    for u, v in G.edges():
        if u == v:
            continue

        i = node_to_id[u]
        j = node_to_id[v]

        adj[i].add(j)
        adj[j].add(i)

    degrees = [len(adj[i]) for i in range(N)]

    def count_internal_edges(S):
        return sum(len(adj[u] & S) for u in S) // 2

    def greedy_initial_solution(start):
        S = {start}

        # gain[v] = число рёбер из вершины v в текущее множество S
        gain = [0] * N

        for w in adj[start]:
            gain[w] += 1

        while len(S) < k:
            best_v = max(
                (v for v in range(N) if v not in S),
                key=lambda v: (gain[v], degrees[v], rng.random())
            )

            S.add(best_v)

            for w in adj[best_v]:
                gain[w] += 1

        return S, gain

    def local_search(S, gain):
        S = set(S)
        outside = set(range(N)) - S

        edge_count = count_internal_edges(S)

        for _ in range(local_iter):
            remove_candidates = sorted(
                S,
                key=lambda u: gain[u]
            )[:min(candidate_pool, len(S))]

            add_candidates = sorted(
                outside,
                key=lambda v: gain[v],
                reverse=True
            )[:min(candidate_pool, len(outside))]

            best_delta = 0
            best_swap = None

            for u in remove_candidates:
                lost_edges = gain[u]

                for v in add_candidates:
                    # gain[v] считает связи v со всем S.
                    # Если v смежна с u, то после удаления u это ребро исчезнет.
                    edge_uv = 1 if u in adj[v] else 0

                    gained_edges = gain[v] - edge_uv
                    delta = gained_edges - lost_edges

                    if delta > best_delta:
                        best_delta = delta
                        best_swap = (u, v)

            if best_swap is None:
                break

            u, v = best_swap

            S.remove(u)
            outside.add(u)

            for w in adj[u]:
                gain[w] -= 1

            outside.remove(v)
            S.add(v)

            for w in adj[v]:
                gain[w] += 1

            edge_count += best_delta

        return S, edge_count

    # Часть запусков стартует из самых высокостепенных вершин,
    # часть — из случайных вершин.
    starts = sorted(range(N), key=lambda i: degrees[i], reverse=True)
    starts = starts[:min(n_restarts, N)]

    while len(starts) < n_restarts:
        starts.append(rng.randrange(N))

    best_S = None
    best_edges = -1

    for start in starts:
        S, gain = greedy_initial_solution(start)
        S, edge_count = local_search(S, gain)

        if edge_count > best_edges:
            best_edges = edge_count
            best_S = S

    best_nodes = [nodes[i] for i in best_S]

    # G.subgraph(nodes) в NetworkX возвращает индуцированный подграф на заданных вершинах;
    # .copy() делает независимую копию, а не view.
    H = G.subgraph(best_nodes).copy()

    return H, best_nodes, best_edges