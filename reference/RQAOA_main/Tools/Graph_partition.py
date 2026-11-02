import networkx as nx
from networkx.algorithms import community
import community as community_louvain
import random
from collections import deque
import matplotlib.pyplot as plt


def random_connected_communities(graph, max_size, seed=None):

    if seed is not None:
        random.seed(seed)

    # Инициализация
    nodes = list(graph.nodes())
    random.shuffle(nodes)  # Случайный порядок вершин
    visited = set()
    communities = []

    for node in nodes:
        if node in visited:
            continue

        # Создаем новый связный подграф
        community = set()
        queue = deque([node])

        while queue and len(community) < max_size:
            current = queue.popleft()
            if current in visited:
                continue

            visited.add(current)
            community.add(current)

            # Добавляем случайных соседей
            neighbors = list(graph.neighbors(current))
            random.shuffle(neighbors)

            for neighbor in neighbors:
                if neighbor not in visited and len(community) < max_size:
                    queue.append(neighbor)

        communities.append(community)

    return communities


def split_graph_into_subgraphs(graph, max_size):


    # communities = list(community.greedy_modularity_communities(graph))
    communities = random_connected_communities(graph, max_size)

    subgraphs = []

    # Проходим по каждому сообществу и разбиваем его на подграфы, если оно слишком большое
    for comm in communities:
        comm_subgraph = graph.subgraph(comm)

        # Если размер сообщества меньше или равен subgraph_size, добавляем его как есть
        if len(comm_subgraph) <= max_size:
            subgraphs.append(comm_subgraph)
        else:
            # Если сообщество слишком большое, разбиваем его на части
            # Используем алгоритм k-связных компонент или просто разбиваем на равные части
            # В данном примере разбиваем на равные части
            nodes = list(comm_subgraph.nodes())
            for i in range(0, len(nodes), max_size):
                subgraph_nodes = nodes[i:i + max_size]
                subgraph = graph.subgraph(subgraph_nodes)
                subgraphs.append(subgraph)

    # Убедимся, что подграфы не содержат изолированных вершин (если это возможно)
    # Если есть подграфы с одной вершиной, попробуем объединить их с соседними подграфами
    final_subgraphs = []
    single_node_subgraphs = []

    for subgraph in subgraphs:
        if len(subgraph) == 1:
            single_node_subgraphs.append(subgraph)
        else:
            final_subgraphs.append(subgraph.copy())

    # Объединяем подграфы с одной вершиной с соседними подграфами
    for single_subgraph in single_node_subgraphs:
        node = list(single_subgraph.nodes())[0]
        neighbors = list(graph.neighbors(node))

        # Ищем соседний подграф, к которому можно добавить вершину
        merged = False
        for i, final_subgraph in enumerate(final_subgraphs):
            if any(neighbor in final_subgraph for neighbor in neighbors):
                # Добавляем вершину к этому подграфу, если это не нарушает subgraph_size
                if len(final_subgraph) < max_size:
                    final_subgraphs[i] = (graph.subgraph(list(final_subgraph.nodes()) + [node])).copy()
                    merged = True
                    break

        # Если не удалось объединить, добавляем как отдельный подграф
        if not merged:
            final_subgraphs.append(single_subgraph.copy())

    return final_subgraphs.copy()


def graph_partition(graph, max_size, visualise=False):

    subgraphs_list = split_graph_into_subgraphs(graph, max_size)
    max_node_name = max(list(graph.nodes()))
    subgraphs_names = list(range(max_node_name + 1, max_node_name + len(subgraphs_list) + 1, 1))

    for subgraph, name in zip(subgraphs_list, subgraphs_names):
        graph.add_node(name)
        nx.set_node_attributes(graph, {name: {"subgraph": subgraph}})
        union_list = {}
        for subnode in subgraph.nodes():
            union_list[subnode] = {"union": name}
        nx.set_node_attributes(graph, union_list)

    cuted_edges = {}
    for name_1 in subgraphs_names:
        for name_2 in subgraphs_names:
            if name_1 == name_2:
                continue
            if name_1 < name_2:
                cuted_edges[(name_1, name_2)] = []


    node_unions = nx.get_node_attributes(graph, "union")
    for subgraph, name in zip(subgraphs_list, subgraphs_names):
        for node in subgraph.nodes():

            for neighbor, weight in graph[node].items():
                if neighbor not in subgraph.nodes():
                    if name < node_unions[neighbor]:
                        cuted_edges[(name, node_unions[neighbor])].append((node, neighbor, weight["weight"]))
                    else:
                        cuted_edges[(node_unions[neighbor], name)].append((neighbor, node, weight["weight"]))
                        graph.remove_edge(node, neighbor)
            graph.remove_node(node)

    graph.add_edges_from(cuted_edges.keys())
    nx.set_edge_attributes(graph, cuted_edges, "cutted_edges")

    if visualise:
        options = {'node_size': 1000, 'width': 1}
        nx.draw(graph, nx.circular_layout(graph), with_labels=True, **options)
        nx.draw_networkx_edges(graph, nx.circular_layout(graph), edgelist=graph.edges(), width=1)
        edge_labels = nx.get_edge_attributes(graph, "weight")
        nx.draw_networkx_edge_labels(graph, nx.circular_layout(graph), edge_labels)

        ax = plt.gca()
        ax.margins(0.08)
        plt.axis("off")
        plt.show()

    return graph.copy()

# Пример использования
# G = generate_graph(40, 0.7, visualise=False)  # Пример графа
# G1 = graph_partition(G, 5, visualise=False)
# print(G1)
# print(nx.get_node_attributes(G1, "subgraph"))
# print(nx.get_edge_attributes(G1, "cuted_edges"))
# G2 = graph_partition(G1, 5, visualise=False)
# print(G2)
# print(nx.get_node_attributes(G2, "subgraph"))
# print(nx.get_edge_attributes(G2, "cuted_edges"))
# F = nx.get_node_attributes(G2, "subgraph")[24]
# print(nx.get_node_attributes(F, "subgraph"))