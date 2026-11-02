import time
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from Diplom.QAOA_СFR import QAOA
from Tools.Graph_partition import *
from Generators.GraphGenerator import DensRandGraph

warnings.filterwarnings("ignore")


global q_energy, c_energy, o_graph, c_cut


class QAOAinQAOA:

    def __init__(self, max_size, depth=1, init_amp=0.1, repetition=1,
                 max_iter=300, callback=False):

        self.graph = nx.Graph
        self.max_size = max_size
        self.depth = depth
        self.init_amp = init_amp
        self.repetition = repetition
        self.max_iter = max_iter
        self.callback = callback

        self.init_graph = nx.Graph
        self.meta_graph = nx.Graph

        self.processing_time = 0
        self.solution_energy = 0
        self.solution_cut = 0
        self.solution_state = []
        self.solution_dct = {}
        self.corr_dct = {}

    @staticmethod
    def _edge_key(u, v):
        return (u, v) if u <= v else (v, u)

    @staticmethod
    def _as_edge_attr_dict(edge_value_dict, attr_name):
        return {
            edge: {attr_name: value}
            for edge, value in edge_value_dict.items()
        }

    def solve(self, graph):

        self.graph = nx.Graph(graph)
        self.init_graph = nx.Graph(graph)
        self.processing_time = time.time()

        while self.graph.number_of_nodes() > 1:

            self.meta_graph = graph_partition(self.graph.copy(), self.max_size)
            self._local_QAOA()
            self._reformulate_QAOA()
            self.graph = self.meta_graph

        nx.set_node_attributes(
            self.graph,
            {node: {"bit": 1} for node in self.graph.nodes()}
        )

        self.graph = self._merge_solution(self.graph)
        self._classical_solution()
        self.corr_dct = nx.get_edge_attributes(self.graph, "corr")
        self.processing_time = time.time() - self.processing_time
        self._callback()

    def _local_QAOA(self):

        bit_dct = {}
        corr_dct = {}
        spin_dct = {}

        subgraphs = nx.get_node_attributes(self.meta_graph, "subgraph")

        spins = nx.get_node_attributes(self.meta_graph, "spin")
        if len(spins.keys()) == 0:
            values = self.init_amp * np.random.uniform(
                -1,
                1,
                size=self.graph.number_of_nodes(),
            )
            spins = dict(zip(self.graph.nodes(), np.asarray(values)))
            nx.set_node_attributes(self.graph, spins, "spin")

        for _ in range(self.repetition):
            for meta_node in self.meta_graph.nodes():

                subgraph = subgraphs[meta_node]

                qaoa = QAOA(
                    graph=subgraph,
                    full_graph=self.graph,
                    field_amp=1,
                    n_layers=self.depth,
                    method="L-BFGS-B",
                    max_iter=self.max_iter,
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
                    optimize=True,
                )

                nx.set_node_attributes(self.graph, qaoa.q_spins_dct, "spin")

                mapping = {
                    node: idx
                    for idx, node in enumerate(list(subgraph.nodes()))
                }

                local_bit_dct = {
                    node: {"bit": bit}
                    for node, bit in qaoa.q_solution_dct.items()
                }

                local_spin_dct = {
                    node: {"spin": spin}
                    for node, spin in qaoa.q_spins_dct.items()
                }

                local_corr_dct = {
                    self._edge_key(u, v): {
                        "corr": corr[mapping[u], mapping[v]]
                    }
                    for u, v in subgraph.edges()
                }

                bit_dct |= local_bit_dct
                spin_dct |= local_spin_dct
                corr_dct |= local_corr_dct

                nx.set_node_attributes(subgraph, local_bit_dct)
                nx.set_node_attributes(subgraph, local_spin_dct)
                nx.set_edge_attributes(subgraph, local_corr_dct)

                nx.set_node_attributes(
                    self.meta_graph,
                    {meta_node: {"subgraph": subgraph}},
                )

        nx.set_node_attributes(self.graph, bit_dct)
        nx.set_node_attributes(self.graph, spin_dct)
        nx.set_edge_attributes(self.graph, corr_dct)

        return self.graph

    def _reformulate_QAOA(self):
        """
        Строит веса мета-графа после локального QAOA.

        Важное исправление:
        если ребро (x, y), попавшее в cutted_edges мета-ребра, само уже
        является мета-ребром предыдущего уровня, то его собственный
        атрибут cutted_edges сохраняется как четвертый элемент кортежа:

            (x, y, w, child_cuts)

        Это не дает потерять иерархическую информацию при глубокой
        рекурсии. Если child_cuts отсутствует, ребро считается листовым.
        """

        bits = nx.get_node_attributes(self.graph, "bit")

        for u, v in self.meta_graph.edges():

            weight = 0
            enriched_cuts = []
            raw_cuts = self.meta_graph.edges[u, v].get("cutted_edges", [])

            for item in raw_cuts:

                x, y, w = item[:3]

                if bits[x] == bits[y]:
                    weight += w
                else:
                    weight -= w

                child_cuts = None
                if self.graph.has_edge(x, y):
                    child_cuts = self.graph.edges[x, y].get("cutted_edges", None)

                if child_cuts is None:
                    enriched_cuts.append((x, y, w))
                else:
                    enriched_cuts.append((x, y, w, child_cuts))

            self.meta_graph.edges[u, v]["weight"] = weight
            self.meta_graph.edges[u, v]["cutted_edges"] = enriched_cuts

    def _merge_solution(self, graph):
        bits = nx.get_node_attributes(graph, "bit")
        spins = nx.get_node_attributes(graph, "spin")
        subs = nx.get_node_attributes(graph, "subgraph")

        rec_graph = nx.Graph()

        node_list = set()
        edge_list = []

        bit_dct = {}
        spin_dct = {}
        sub_dct = {}
        corr_dct = {}
        cut_dct = {}

        for node in graph.nodes():

            node_bit = bits.get(node, 1)

            # ВАЖНО: если это уже листовая вершина, её нельзя терять.
            if node not in subs:
                node_list.add(node)
                bit_dct[node] = {"bit": node_bit}

                if node in spins:
                    spin_dct[node] = {"spin": spins[node]}

                continue

            subgraph = subs[node]

            sub_bits = nx.get_node_attributes(subgraph, "bit")
            sub_spins = nx.get_node_attributes(subgraph, "spin")
            sub_subs = nx.get_node_attributes(subgraph, "subgraph")
            sub_corrs = nx.get_edge_attributes(subgraph, "corr")
            sub_cuts = nx.get_edge_attributes(subgraph, "cutted_edges")

            # ВАЖНО: переносим все вершины подграфа, даже если у них нет рёбер.
            for k in subgraph.nodes():
                node_list.add(k)
                bit_dct[k] = {"bit": sub_bits.get(k, 1) * node_bit}

                if k in sub_spins:
                    spin_dct[k] = {"spin": sub_spins[k]}

                if k in sub_subs:
                    sub_dct[k] = {"subgraph": sub_subs[k]}

            for (u, v), corr in sub_corrs.items():
                corr_dct[self._edge_key(u, v)] = {"corr": corr}

            for (u, v), cuts in sub_cuts.items():
                cut_dct[self._edge_key(u, v)] = {"cutted_edges": cuts}

            for u, v in subgraph.edges():
                edge_list.append(self._edge_key(u, v))

        for u, v in graph.edges():

            edge_data = graph.edges[u, v]
            edge_corr = edge_data.get("corr", None)
            edge_cuts = edge_data.get("cutted_edges", None)

            if edge_cuts is None:
                edge = self._edge_key(u, v)
                edge_list.append(edge)
                node_list.update(edge)

                if edge_corr is not None:
                    corr_dct[edge] = {"corr": edge_corr}

                continue

            if edge_corr is None:
                raise ValueError(
                    f"У мета-ребра {(u, v)} есть 'cutted_edges', "
                    f"но отсутствует 'corr'."
                )

            su = spins.get(u, 0.0)
            sv = spins.get(v, 0.0)
            uv = su * sv
            norm_uv = np.sqrt(max(0.0, (1 - su ** 2) * (1 - sv ** 2)))
            eps = 1e-6

            for item in edge_cuts:

                k, j, w = item[:3]
                child_cuts = item[3] if len(item) > 3 else None

                new_edge = self._edge_key(k, j)
                edge_list.append(new_edge)
                node_list.update(new_edge)

                if k not in spin_dct or j not in spin_dct:
                    raise ValueError(
                        f"Не найден spin для вершины {k} или {j} "
                        f"при разворачивании ребра {(u, v)}."
                    )

                sk = spin_dct[k]["spin"]
                sj = spin_dct[j]["spin"]

                kj = sk * sj
                norm_kj = np.sqrt(max(0.0, (1 - sk ** 2) * (1 - sj ** 2)))

                corr_dct[new_edge] = {
                    "corr": kj + (edge_corr - uv) * (norm_kj / (norm_uv + eps))
                }

                if child_cuts is not None:
                    cut_dct[new_edge] = {"cutted_edges": child_cuts}

        # ВАЖНО: сначала добавляем вершины, потом рёбра, потом атрибуты.
        rec_graph.add_nodes_from(node_list)
        rec_graph.add_edges_from(edge_list)

        nx.set_node_attributes(rec_graph, bit_dct)
        nx.set_node_attributes(rec_graph, spin_dct)
        nx.set_node_attributes(rec_graph, sub_dct)

        nx.set_edge_attributes(rec_graph, corr_dct)
        nx.set_edge_attributes(rec_graph, cut_dct)

        if len(sub_dct) == 0:
            return rec_graph

        return self._merge_solution(rec_graph)

    def _classical_solution(self):

        self.solution_dct = nx.get_node_attributes(self.graph, "bit")
        edges_weights = nx.get_edge_attributes(self.init_graph, "weight")

        self.solution_cut = 0.5 * np.sum([
            edges_weights[(u, v)] * (1 - self.solution_dct[u] * self.solution_dct[v])
            for u, v in self.init_graph.edges()
        ])

    def _callback(self):

        if self.callback:
            print("\033[33m {}".format("====================================="))
            print(
                f"Subgraph size : {self.max_size}   "
                f"Depth : {self.depth}   "
                f"Max iter : {self.max_iter}"
            )
            print(
                f"Number of nodes : {nx.number_of_nodes(self.init_graph)}   "
                f"Number of edges : {nx.number_of_edges(self.init_graph)}"
            )
            print(f"QAOA2  ::  Q_energy : {round(self.solution_energy, 6)}")
            print(f"Classical energy for QAOA2 state: {self.solution_cut}")
            print(f"Processing time : {round(self.processing_time, 6)} s")
            print("\033[33m {}".format("====================================="))


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
        np.ma.masked_invalid(matrix),
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

    G = DensRandGraph(
        number_of_nodes=300,
        density=1,
        seed=1001,
        weights="random",
    )

    qaoa2 = QAOAinQAOA(
        max_size=10,
        depth=2,
        repetition=5,
        max_iter=300,
        init_amp=0.5,
        callback=True,
    )
    qaoa2.solve()

    fig, ax, corr_matrix = plot_corr_heatmap(
        qaoa2.graph,
        figsize=(10, 10),
        title="Карта корреляций рёбер",
    )

    plt.show()
