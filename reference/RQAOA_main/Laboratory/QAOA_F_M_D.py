import numpy as np
from ENV.ProblemGenerator import WishartMaxCutGenerator
from Laboratory.QAOA_F_M import QAOA
from Generators.GraphGenerator import DensRandGraph
from Tools.GraphDecomposer import *
from Solvers.GoemansWilliamson import *


class DistQAOA:

    def __init__(self, graph, subgraphs, n_layers, local_iter, global_iter, method='ADAM'):
        self.graph = nx.Graph(graph)
        self.subgraphs = subgraphs
        self.n_layers = n_layers
        self.local_iter = local_iter
        self.global_iter = global_iter
        self.method = method

        self.params = dict(zip(range(len(self.subgraphs)), [[] for _ in range(len(self.subgraphs))]))
        self.flag = 'no ones'
        self.solution = {}
        self.cut = 0

    def _cut_value(self):

        edges_weights = nx.get_edge_attributes(self.graph, "weight")
        self.cut = -1 * np.sum([edges_weights[(k, j)] if (self.solution[k] != self.solution[j]) else 0
                           for k, j in self.graph.edges()])


    def solve(self):

        bits = dict(zip(self.graph.nodes(), np.zeros(int(self.graph.number_of_nodes()))))
        nx.set_node_attributes(self.graph, bits, "bit")

        spins = dict(zip(self.graph.nodes(), np.zeros(int(self.graph.number_of_nodes()))))
        nx.set_node_attributes(self.graph, spins, "spin")


        for step in range(self.global_iter):

            if step > 0:
                self.flag = 'params'

            for idx, subgraph in enumerate(self.subgraphs):

                qaoa = QAOA(graph=subgraph,
                            full_graph=self.graph,
                            n_layers=self.n_layers,
                            max_iter=self.local_iter,
                            method='L-BFGS-B',
                            callback=False,
                            logs=False,
                            samples=25,
                            params=self.params[idx],
                            initialization=self.flag)
                qaoa.solve()

                nx.set_node_attributes(self.graph, qaoa.q_spin_dct, "spin")
                nx.set_node_attributes(self.graph, qaoa.q_solution_dct, "bit")
                self.params[idx] = qaoa.params

            self.solution = nx.get_node_attributes(self.graph, "bit")
            self._cut_value()

            print(f'Step: {step}   Cut: {self.cut}   Params: {self.params}')






if __name__ == '__main__':
    rng = np.random.default_rng(7)

    G = DensRandGraph(number_of_nodes=100,
                      weights='pos_neg',
                      density=1,
                      seed=1000)

    # gw = GoemansWilliamsonMaxcut(graph=G,
    #                              no_hyperplane_samples=100,
    #                              callback=True)
    # gw.solve()


    config = SubgraphConfig(
        max_size=10,
        num_subgraphs=15,
        allow_overlap=True,
        random_seed=42)

    generator = OptimizedDenseSubgraphGenerator(G, config)
    partition = generator.generate_multiple_subgraphs()


    dqaoa = DistQAOA(
        graph=G,
        subgraphs=partition,
        n_layers=1,
        local_iter=1,
        global_iter=100,
    )
    dqaoa.solve()



