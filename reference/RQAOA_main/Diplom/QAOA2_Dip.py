from Solvers.QAOA import *
import warnings
from Tools.Graph_partition import *
from Generators.GraphGenerator import DensRandGraph
import time
warnings.filterwarnings('ignore')

global q_energy, c_energy, o_graph, c_cut


class QAOAinQAOA:

    def __init__(self, graph, max_size, depth=1, max_iter=300, callback=False):

        self.graph = nx.Graph(graph)
        self.max_size = max_size
        self.depth = depth
        self.max_iter = max_iter
        self.callback = callback

        self.init_graph = nx.Graph(graph)
        self.meta_graph = nx.Graph(graph)

        self.processing_time = 0
        self.solution_energy = 0
        self.solution_cut = 0
        self.solution_state = []
        self.solution_dct = {}


    def solve(self):

        self.processing_time = time.time()

        weights = sum(nx.get_edge_attributes(self.graph, "weight").values()) * 0.5
        self.solution_energy = weights

        while self.graph.number_of_nodes() > 1:

            self.meta_graph = graph_partition(self.graph.copy(), self.max_size)
            self._local_QAOA()
            self._reformulate_QAOA()
            self.graph = self.meta_graph

        subgraphs = list(nx.get_node_attributes(self.graph, "subgraph").values())[0]
        self._merge_solution(subgraphs)
        self._classical_solution()
        self.processing_time = time.time() - self.processing_time
        self._callback()

    def _local_QAOA(self):

        bit_dct = {}
        subgraphs= nx.get_node_attributes(self.meta_graph, "subgraph")

        for node in self.meta_graph.nodes():

            subgraph = subgraphs[node]
            qaoa = QAOA(subgraph,
                        n_layers=self.depth,
                        max_iter=self.max_iter,
                        logs=False,
                        callback=False)

            qaoa.solve()
            bit_dct = bit_dct | {node:{'bit': bit} for node, bit in qaoa.q_solution_dct.items()}

            weights = nx.get_edge_attributes(subgraph, "weight")

            for k, j in subgraph.edges():
                if bit_dct[k] == bit_dct[j]:
                    self.solution_energy -= weights[k, j] * 0.5
                else:
                    self.solution_energy += weights[k, j] * 0.5

            nx.set_node_attributes(subgraph, bit_dct)
            nx.set_node_attributes(self.meta_graph, {node: {"subgraph": subgraph}})
        nx.set_node_attributes(self.graph, bit_dct)


        return self.graph

    def _reformulate_QAOA(self):

        bits = nx.get_node_attributes(self.graph, "bit")
        cutted_edges = nx.get_edge_attributes(self.meta_graph, "cuted_edges")

        for k, j in self.meta_graph.edges():
            weight = 0
            for x, y, w in cutted_edges[(k, j)]:
                if bits[x] == bits[y]:
                    weight += w
                else:
                    weight -= w

            nx.set_edge_attributes(self.meta_graph, {(k, j): {"weight": weight}})

    def _merge_solution(self, graph):

        state = nx.get_node_attributes(graph, "bit")
        subgraphs = nx.get_node_attributes(graph, "subgraph")

        for node in subgraphs.keys():

            subgraph = subgraphs[node]
            sub_state = nx.get_node_attributes(subgraph, "bit")

            if state[node] == 1:
                for sub_node in sub_state.keys():
                    nx.set_node_attributes(subgraph, {sub_node: {"bit": -1 * sub_state[sub_node]}})

            if len(list(nx.get_node_attributes(subgraph, "subgraph").keys())) != 0:
                self._merge_solution(subgraph)

            else:
                sub_state = nx.get_node_attributes(subgraph, "bit")
                for sub_node in sub_state.keys():
                    nx.set_node_attributes(self.init_graph, {sub_node: {"bit": sub_state[sub_node]}})

    def _classical_solution(self):

        self.solution_dct = nx.get_node_attributes(self.init_graph, "bit")
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




if __name__ == "__main__":

    G = DensRandGraph(number_of_nodes=100,
                      density=1,
                      seed=1000,
                      weights='random')

    qaoa2 = QAOAinQAOA(graph=G,
                       max_size=10,
                       depth=2,
                       max_iter=300,
                       callback=True)
    qaoa2.solve()
    print(qaoa2.solution_dct)


