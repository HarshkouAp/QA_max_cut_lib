from Solvers.QAOA import *
import warnings
from Tools.Graph_partition import *
from Generators.GraphGenerator import DensRandGraph
from Generators.maxcut_benchmark_generator import *
from Solvers.GoemansWilliamson import GoemansWilliamsonMaxcut
import time
warnings.filterwarnings('ignore')

global q_energy, c_energy, o_graph, c_cut

def QAOA2(graph, max_size, depth=10, max_iter=3000, callback=False, method='L-BFGS-B'):

    def recursion(g):

        global q_energy, c_energy

        weights = sum(nx.get_edge_attributes(g, "weight").values()) * 0.5
        q_energy = weights
        c_energy = weights

        while g.number_of_nodes() > 1:

            g1 = graph_partition(g.copy(), max_size)
            g, g1 = local_QAOA(g, g1)
            g1 = reformulate_QAOA(g, g1)
            g = g1

        g = list(nx.get_node_attributes(g, "subgraph").values())[0]
        merge_solution(g)
        classical_solution()

    def local_QAOA(pre_graph, graph, mode="sol", method='L-BFGS-B'):

        global q_energy, c_energy
        bit_list = {}
        subgraphs= nx.get_node_attributes(graph, "subgraph")

        for node in graph.nodes():

            subgraph = subgraphs[node]
            old_nodes = list(subgraph.nodes())
            new_nodes = list(range(len(old_nodes)))
            mapping = dict(zip(old_nodes, new_nodes))
            subgraph = nx.relabel_nodes(subgraph, mapping)

            qaoa = QAOA(subgraph,
                        n_layers=depth,
                        max_iter=max_iter,
                        logs=False,
                        callback=False,
                        method=method)

            qaoa.solve()

            mapping = dict(zip(new_nodes, old_nodes))
            subgraph = nx.relabel_nodes(subgraph, mapping)
            q_sol = "".join(['1' if bit == 1 else '0' for bit in qaoa.q_solution])
        
            for subnode, bit in zip(list(subgraph.nodes()), q_sol):
                bit_list[subnode] = {"bit": bit}

            weights = nx.get_edge_attributes(subgraph, "weight")
            for k, j in subgraph.edges():
                if bit_list[k] == bit_list[j]:
                    q_energy -= weights[k, j] * 0.5
                    c_energy -= weights[k, j] * 0.5
                else:
                    q_energy += weights[k, j] * 0.5
                    c_energy += weights[k, j] * 0.5

            nx.set_node_attributes(subgraph, bit_list)
            nx.set_node_attributes(graph, {node: {"subgraph": subgraph}})
            nx.set_node_attributes(pre_graph, bit_list)

        return pre_graph, graph

    def reformulate_QAOA(pre_graph, graph):

        bits = nx.get_node_attributes(pre_graph, "bit")
        cuted_edges = nx.get_edge_attributes(graph, "cuted_edges")

        for k, j in graph.edges():
            weight = 0
            for x, y, w in cuted_edges[(k, j)]:
                if bits[x] == bits[y]:
                    weight += w
                else:
                    weight -= w

            nx.set_edge_attributes(graph, {(k, j): {"weight": weight}})

        return graph

    def invert(solution):
        inv_solution = ""
        for bit in solution:
            inv_solution += str((int(bit) + 1) % 2)
        return inv_solution

    def merge_solution(graph):

        global o_graph

        state = nx.get_node_attributes(graph, "bit")
        subgraphs = nx.get_node_attributes(graph, "subgraph")

        for node in subgraphs.keys():

            subgraph = subgraphs[node]
            sub_state = nx.get_node_attributes(subgraph, "bit")

            if state[node] == "1":
                for sub_node in sub_state.keys():
                    nx.set_node_attributes(subgraph, {sub_node: {"bit": invert(sub_state[sub_node]) }})

            if len(list(nx.get_node_attributes(subgraph, "subgraph").keys())) != 0:
                merge_solution(subgraph)

            else:
                sub_state = nx.get_node_attributes(subgraph, "bit")
                for sub_node in sub_state.keys():
                    nx.set_node_attributes(o_graph, {sub_node: {"bit": sub_state[sub_node]}})

    def classical_solution():
        global c_cut, o_graph
        c_cut = 0
        state = nx.get_node_attributes(o_graph, "bit")
        edges_weights = nx.get_edge_attributes(o_graph, "weight")
        c_cut = np.sum([edges_weights[(k, j)] if (state[k] != state[j]) else 0
                                    for k, j in o_graph.edges()])

    global q_energy, c_energy, o_graph, c_cut

    start = time.time()
    o_graph = graph.copy()
    recursion(graph)
    end = time.time()

    if callback:

        print("\033[33m {}".format(f"====================================="))
        print(f"Subgraph size : {max_size}   Depth : {depth}   Max iter : {max_iter}")
        print(f"Number of nodes : {nx.number_of_nodes(o_graph)}   Number of edges : {nx.number_of_edges(o_graph)}")
        print(f"QAOA2  ::  Q_energy : {round(q_energy, 6)}   BF-in-BF : {c_energy}")
        print(f"Classical energy for QAOA2 state: {c_cut}")
        print(f"Processing time : {round(end - start, 6)} s")
        print("\033[33m {}".format(f"====================================="))

    return q_energy, c_energy, c_cut


class QAOAinQAOA:

    def __init__(self, graph, max_size, n_layers, max_iter, method):
        self.graph = nx.Graph(graph)
        self.max_size = max_size
        self.n_layers = n_layers
        self.max_iter = max_iter
        self.method = method

        self.q_energy = 0
        self.c_energy = 0
        self.c_cut = 0

    def solve(self):

        self.q_energy, self.c_energy, self.c_cut = QAOA2(graph=self.graph,
                                                         max_size=self.max_size,
                                                         depth=self.n_layers,
                                                         max_iter=self.max_iter,
                                                         method=self.method)

if __name__ == "__main__":

    G = DensRandGraph(number_of_nodes=100,
                      density=0.5,
                      seed=1000,
                      weights='random')

    G = get_benchmark_graph(129)

    qaoa2 = QAOAinQAOA(graph=G,
                       max_size=5,
                       n_layers=1,
                       max_iter=300,
                       method='ADAM')
    qaoa2.solve()
    print(qaoa2.q_energy)
    print(qaoa2.c_cut)


