import networkx as nx

from Diplom.QAOA_СF import QAOA
from Tools.Graph_partition import *
from Generators.GraphGenerator import DensRandGraph
import numpy as np
import warnings
import time
warnings.filterwarnings('ignore')

global q_energy, c_energy, o_graph, c_cut


class QAOAinQAOA:

    def __init__(self, graph, max_size, depth=1, init_amp=0.1, repetition=1,
                 max_iter=300, callback=False):

        self.graph = nx.Graph(graph)
        self.max_size = max_size
        self.depth = depth
        self.init_amp = init_amp
        self.repetition = repetition
        self.max_iter = max_iter
        self.callback = callback

        self.init_graph = nx.Graph(graph)
        self.meta_graph = nx.Graph(graph)

        self.processing_time = 0
        self.solution_energy = 0
        self.solution_cut = 0
        self.solution_state = []
        self.solution_dct = {}
        self.corr_dct = {}


    def solve(self):

        self.processing_time = time.time()

        while self.graph.number_of_nodes() > 1:

            self.meta_graph = graph_partition(self.graph.copy(), self.max_size)
            self._local_QAOA()
            self._reformulate_QAOA()
            self.graph = self.meta_graph

        nx.set_node_attributes(self.graph, {k: {"bit": 1} for k in self.graph.nodes()})
        self.graph = self._merge_solution(self.graph)
        self._classical_solution()
        self.corr_dct = nx.get_edge_attributes(self.graph, "corr")
        self.processing_time = time.time() - self.processing_time
        self._callback()

    def _local_QAOA(self):

        bit_dct, corr_dct, spin_dct = {}, {}, {}
        subgraphs = nx.get_node_attributes(self.meta_graph, "subgraph")
        values = self.init_amp * np.random.uniform(-1, 1, size=self.graph.number_of_nodes())
        spins = dict(zip(self.graph.nodes(), np.array(values)))
        nx.set_node_attributes(self.graph, spins, "spin")

        for _ in range(self.repetition):
            for node in self.meta_graph.nodes():

                subgraph = subgraphs[node]

                qaoa = QAOA(
                    graph=subgraph,
                    full_graph=self.graph,
                    field_amp=1,
                    n_layers=1,
                    method="L-BFGS-B",
                    max_iter=100,
                    samples=25,
                    callback=False,
                    logs=False,
                )

                qaoa.solve()

                corr = np.einsum(
                    "s,si,sj->ij",
                    qaoa.probs,
                    qaoa.state_list,
                    qaoa.state_list,
                    optimize=True
                )

                nx.set_node_attributes(self.graph, qaoa.q_spins_dct, "spin")
                mapping = {node: i for i, node in enumerate(list(subgraph.nodes()))}
                bit_dct |= {node: {'bit': bit} for node, bit in qaoa.q_solution_dct.items()}
                spin_dct |= {node: {'spin': spin} for node, spin in qaoa.q_spins_dct.items()}
                corr_dct |= {edge: {'corr': corr[mapping[edge[0]], mapping[edge[1]]]} for edge in subgraph.edges()}

                nx.set_node_attributes(subgraph, bit_dct)
                nx.set_node_attributes(subgraph, spin_dct)
                nx.set_edge_attributes(subgraph, corr_dct)
                nx.set_node_attributes(self.meta_graph, {node: {"subgraph": subgraph}})

        nx.set_node_attributes(self.graph, bit_dct)
        nx.set_edge_attributes(self.graph, corr_dct)

        return self.graph

    def _reformulate_QAOA(self):

        bits = nx.get_node_attributes(self.graph, "bit")
        cutted_edges = nx.get_edge_attributes(self.meta_graph, "cutted_edges")

        for k, j in self.meta_graph.edges():
            weight = 0
            for x, y, w in cutted_edges[(k, j)]:
                if bits[x] == bits[y]:
                    weight += w
                else:
                    weight -= w

            nx.set_edge_attributes(self.meta_graph, {(k, j): {"weight": weight}})

    def _merge_solution(self, graph):

        bits = nx.get_node_attributes(graph, "bit")
        spins = nx.get_node_attributes(graph, "spin")
        subs = nx.get_node_attributes(graph, "subgraph")
        corrs = nx.get_edge_attributes(graph, "corr")
        cuts = nx.get_edge_attributes(graph, "cutted_edges")
        print(cuts)

        rec_graph = nx.Graph()
        bit_dct, corr_dct, spin_dct, sub_dct, cut_dct, edges  = {}, {}, {}, {}, {}, []

        for node in graph.nodes():

            sub_bits = nx.get_node_attributes(subs[node], "bit")
            sub_subs = nx.get_node_attributes(subs[node], "subgraph")
            sub_spins = nx.get_node_attributes(subs[node], "spin")
            sub_corrs = nx.get_edge_attributes(subs[node], "corr")
            sub_cuts = nx.get_edge_attributes(subs[node], "cutted_edges")

            bit_dct |= {k: {"bit": bit * bits[node]} for k, bit in sub_bits.items()}
            sub_dct |= {k: {"subgraph": sub} for k, sub in sub_subs.items()}
            spin_dct |= {k: {"spin": spin} for k, spin in sub_spins.items()}
            corr_dct |= {edge : {"corr" : corr} for edge, corr in sub_corrs.items()}
            cut_dct |= {edge : {"cutted_edges" : edges} for edge, edges in sub_cuts.items()}

            edges += [(k, j) for k, j in subs[node].edges()]

        for u, v in graph.edges():

            cut = cuts[(u, v)]
            edges += [(k, j) for k, j, w in cut]

            for k, j, w in cut:

                sk, sj = spin_dct[k]['spin'], spin_dct[j]['spin']
                su, sv = spins[u], spins[v]
                kj = sk * sj
                uv = su * sv
                norm_kj = np.sqrt((1 - sk**2) * (1 - sj**2))
                norm_uv = np.sqrt((1 - su ** 2) * (1 - sv ** 2))
                e = 0.000001

                corr_dct[(k, j)] = {"corr" : kj + (corrs[(u, v)] - uv) * (norm_kj / (norm_uv + e))}


        rec_graph.add_edges_from(edges)
        nx.set_edge_attributes(rec_graph, corr_dct)
        nx.set_edge_attributes(rec_graph, cut_dct)
        nx.set_node_attributes(rec_graph, bit_dct)
        nx.set_node_attributes(rec_graph, sub_dct)
        nx.set_node_attributes(rec_graph, spin_dct)

        print(rec_graph.number_of_nodes())

        if len(list(sub_dct.items())) == 0:
            return rec_graph
        else:
            rec_graph = self._merge_solution(rec_graph)
            return rec_graph

    def _classical_solution(self):

        self.solution_dct = nx.get_node_attributes(self.graph, "bit")
        edges_weights = nx.get_edge_attributes(self.init_graph, "weight")
        self.solution_cut = 0.5 * np.sum([edges_weights[(k, j)] * (1 - self.solution_dct[k] * self.solution_dct[j])
                                    for k, j in self.init_graph.edges()])

    def _callback(self):

        if self.callback:
            print("\033[33m {}".format(f"====================================="))
            print(f"Subgraph size : {self.max_size}   Depth : {self.depth}   Max iter : {self.max_iter}")
            print(f"Number of nodes : {nx.number_of_nodes(self.init_graph)}   Number of edges : {nx.number_of_edges(self.init_graph)}")
            print(f"QAOA2  ::  Q_energy : {round(self.solution_energy, 6)}")
            print(f"Classical energy for QAOA2 state: {self.solution_cut}")
            print(f"Processing time : {round(self.processing_time, 6)} s")
            print("\033[33m {}".format(f"====================================="))


def plot_corr_heatmap(
    G: nx.Graph,
    attr: str = "corr",
    nodes=None,
    cmap: str = "coolwarm",
    vmin: float = -1.0,
    vmax: float = 1.0,
    figsize=(8, 8),
    diagonal_value: float = 0.0,
    title: str = "Карта корреляций",
):
    if nodes is None:
        nodes = list(G.nodes())

    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    matrix = np.full((n, n), np.nan, dtype=float)
    np.fill_diagonal(matrix, diagonal_value)

    for (u, v), corr in nx.get_edge_attributes(G, attr).items():
        if u not in node_to_idx or v not in node_to_idx:
            continue

        i = node_to_idx[u]
        j = node_to_idx[v]

        matrix[i, j] = float(corr)

        if not G.is_directed():
            matrix[j, i] = float(corr)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("lightgray")

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        matrix,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )

    fig.colorbar(im, ax=ax, label=attr)

    ax.set_title(title)
    ax.set_xlabel("Вершины")
    ax.set_ylabel("Вершины")

    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()

    return fig, ax, matrix


if __name__ == "__main__":

    G = DensRandGraph(number_of_nodes=80,
                      density=1,
                      seed=1001,
                      weights='random')

    qaoa2 = QAOAinQAOA(graph=G,
                       max_size=10,
                       depth=1,
                       repetition=5,
                       max_iter=300,
                       init_amp=1,
                       callback=True)
    qaoa2.solve()

    fig, ax, corr_matrix = plot_corr_heatmap(
        qaoa2.graph,
        figsize=(10, 10),
        title="Карта корреляций рёбер",
    )

    plt.show()