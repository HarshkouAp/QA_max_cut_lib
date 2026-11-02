import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
import numpy as np
from sympy.codegen.fnodes import dimension


def ConnectedRandGraph(number_of_nodes, number_of_edges, weights="ones", seed=None):

    if number_of_edges < number_of_nodes:
        raise Exception("Number of edges must be greater than number of nodes or equal to number of nodes")

    if seed is None:
        seed = round(1000 * np.random.random())

    graph = nx.path_graph(number_of_nodes)
    edges = number_of_nodes - 1

    pairs = list(product(range(number_of_nodes), repeat=2))
    np.random.seed(seed)
    np.random.shuffle(pairs)

    for j_node, k_node in pairs:

        if edges == number_of_edges:
            break

        if (j_node not in nx.neighbors(graph, k_node)) & (j_node != k_node):
            graph.add_edge(int(j_node), int(k_node))
            edges += 1

    if weights == 'random':
        np.random.seed(seed)
        weights = dict(zip(graph.edges(), np.round(2 * np.random.random_sample(int(number_of_edges)), 4)))
    elif weights == 'pos_neg':
        np.random.seed(seed)
        rand_int = np.random.randint(1, 11, size=number_of_edges)
        np.random.seed(seed)
        rand_bit = 1 - (2 * np.random.randint(0, 2, size=number_of_edges))
        weights = dict(zip(graph.edges(), rand_int * rand_bit))
    elif weights == 'ones':
        weights = dict(zip(graph.edges(), np.ones(int(number_of_edges))))

    nx.set_edge_attributes(graph, weights, "weight")

    return graph

def DensRandGraph(number_of_nodes, density, weights="ones", seed=None):

    if seed is None:
        seed = round(1000 * np.random.random())

    number_of_edges = int(number_of_nodes * (number_of_nodes - 1) * density / 2)
    graph = nx.dense_gnm_random_graph(number_of_nodes, int(number_of_edges), seed)

    if weights == 'random':
        np.random.seed(seed)
        weights = dict(zip(graph.edges(), np.round(0.01 + np.random.random_sample(int(number_of_edges)), 4)))
    elif weights == 'pos_neg':
        np.random.seed(seed)
        rand_int = np.random.randint(1, 11, size=number_of_edges)
        np.random.seed(seed)
        rand_bit = 1 - (2 * np.random.randint(0, 2, size=number_of_edges))
        weights = dict(zip(graph.edges(), rand_int * rand_bit))
    elif weights == 'ones':
        weights = dict(zip(graph.edges(), np.ones(int(number_of_edges))))


    nx.set_edge_attributes(graph, weights, "weight")

    return graph

def EdgesRandGraph(number_of_nodes, number_of_edges, weights="ones", seed=None):

    if number_of_edges < number_of_nodes:
        raise Exception("Number of edges must be greater than number of nodes or equal to number of nodes")

    if seed is None:
        seed = round(1000 * np.random.random())

    graph = nx.dense_gnm_random_graph(number_of_nodes, number_of_edges, seed)

    if weights == 'random':
        np.random.seed(seed)
        weights = dict(zip(graph.edges(), np.round(2 * np.random.random_sample(int(number_of_edges)), 4)))
    elif weights == 'ones':
        weights = dict(zip(graph.edges(), np.ones(int(number_of_edges))))

    nx.set_edge_attributes(graph, weights, "weight")

    return graph

def RegularGraph(number_of_nodes, degree, weights="ones", seed=None):

    if degree < 0 or degree >= number_of_nodes:
        raise Exception("Degree must be greater than or equal to 0 and less than number of nodes")

    if number_of_nodes * degree % 2 != 0:
        raise Exception("The product of number_of_nodes and degree must be even")

    if seed is None:
        seed = round(1000 * np.random.random())

    graph = nx.random_regular_graph(degree, number_of_nodes, seed=seed)
    number_of_edges = graph.number_of_edges()

    if weights == 'random':
        np.random.seed(seed)
        weights = dict(zip(graph.edges(), np.round(2 * np.random.random_sample(int(number_of_edges)), 4)))
    elif weights == 'pos_neg':
        np.random.seed(seed)
        rand_int = np.random.randint(1, 11, size=number_of_edges)
        np.random.seed(seed)
        rand_bit = 1 - (2 * np.random.randint(0, 2, size=number_of_edges))
        weights = dict(zip(graph.edges(), rand_int * rand_bit))
    elif weights == 'ones':
        weights = dict(zip(graph.edges(), np.ones(int(number_of_edges))))

    nx.set_edge_attributes(graph, weights, "weight")

    return graph

def GridGraph(dimentions, weights="ones", seed=None):

    if seed is None:
        seed = round(1000 * np.random.random())

    graph = nx.grid_graph(dim=dimentions, periodic=True)

    number_of_nodes = 1
    for d in dimentions:
        number_of_nodes *= d

    number_of_edges = graph.number_of_edges()
    mapping = dict(zip(graph.nodes(), range(number_of_nodes)))
    nx.relabel_nodes(graph, mapping, copy=False)

    if weights == 'random':
        np.random.seed(seed)
        weights = dict(zip(graph.edges(), np.round(2 * np.random.random_sample(int(number_of_edges)), 4)))
    elif weights == 'pos_neg':
        np.random.seed(seed)
        rand_int = np.random.randint(1, 11, size=number_of_edges)
        np.random.seed(seed)
        rand_bit = 1 - (2 * np.random.randint(0, 2, size=number_of_edges))
        weights = dict(zip(graph.edges(), rand_int * rand_bit))
    elif weights == 'ones':
        weights = dict(zip(graph.edges(), np.ones(int(number_of_edges))))

    nx.set_edge_attributes(graph, weights, "weight")

    return graph

def BenchGraph(seed):

    if seed == 0:
        return DensRandGraph(number_of_nodes=120, density=0.2, weights="random", seed=seed)
    elif seed == 1:
        return DensRandGraph(number_of_nodes=120, density=0.4, weights="random", seed=seed)
    elif seed == 2:
        return DensRandGraph(number_of_nodes=120, density=0.6, weights="random", seed=seed)
    elif seed == 3:
        return DensRandGraph(number_of_nodes=120, density=0.8, weights="random", seed=seed)
    elif seed == 4:
        return DensRandGraph(number_of_nodes=120, density=1.0, weights="random", seed=seed)

    elif seed == 5:
        return DensRandGraph(number_of_nodes=120, density=0.2, weights="pos_neg", seed=seed)
    elif seed == 6:
        return DensRandGraph(number_of_nodes=120, density=0.4, weights="pos_neg", seed=seed)
    elif seed == 7:
        return DensRandGraph(number_of_nodes=120, density=0.6, weights="pos_neg", seed=seed)
    elif seed == 8:
        return DensRandGraph(number_of_nodes=120, density=0.8, weights="pos_neg", seed=seed)
    elif seed == 9:
        return DensRandGraph(number_of_nodes=120, density=1.0, weights="pos_neg", seed=seed)


    elif seed == 10:
        return DensRandGraph(number_of_nodes=250, density=0.1, weights="random", seed=seed)
    elif seed == 11:
        return DensRandGraph(number_of_nodes=500, density=0.05, weights="random", seed=seed)
    elif seed == 12:
        return DensRandGraph(number_of_nodes=1000, density=0.025, weights="random", seed=seed)

    elif seed == 13:
        return GridGraph(dimentions=(10, 12), weights="random", seed=seed)
    elif seed == 14:
        return GridGraph(dimentions=(5, 5, 5), weights="random", seed=seed)
    elif seed == 15:
        return GridGraph(dimentions=(4, 3, 3, 3), weights="random", seed=seed)


    return None


def DrawGraph(self):

    options = {'node_size': 1000, 'width': 1}
    pos = nx.spring_layout(self)
    nx.draw(self, pos=pos, with_labels=True, **options)
    nx.draw_networkx_edges(self, pos=pos, edgelist=self.edges(), width=1)
    edge_labels = nx.get_edge_attributes(self, "weight")
    nx.draw_networkx_edge_labels(self, pos=pos, edge_labels=edge_labels)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.show()



if __name__ == '__main__':

    U = DensRandGraph(number_of_nodes=10,
                             density=0.7,
                             weights="ones",
                             seed=42)
    DrawGraph(U)





