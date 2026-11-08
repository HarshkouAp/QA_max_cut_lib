from Laboratory.LS_RQAOA_fast_1 import LargeScaleRQAOA_1
from Laboratory.EdgeExtender import *

import networkx as nx
import numpy as np



class FineTuning:

    def __init__(self, graph, solution, corr_dict, max_size, number_of_edges, number_of_corrections=-1):
        self.graph = graph
        self.solution = np.array([1 if bit == '1' else -1 for bit in solution])
        self.corr_dict = corr_dict
        self.max_size = max_size
        self.number_of_edges = number_of_edges
        self.number_of_corrections = number_of_corrections

        self.adj = nx.adjacency_matrix(self.graph)
        self.n_nodes = self.graph.number_of_nodes()
        self.weird_edges = []
        self.c_solution = [1 if bit == '1' else -1 for bit in solution]
        self.c_energy = self._compute_cut_value(self.solution)


    def _compute_cut_value(self, solution):

        cut = 0.0
        for u, v in self.graph.edges():
            w = self.adj[u][v]
            if solution[u] != solution[v]:
                cut += w
        return cut


    def _fine_tuning_hamiltonian(self, subgraph):

        n_nodes = subgraph.number_of_nodes()
        nodes = list(subgraph.nodes())
        total_states = 1 << n_nodes
        hamiltonian = np.zeros(1 << n_nodes)

        states = self.solution
        for _ in range(n_nodes):
            states = np.vstack((states, states))

        combs = 2 * ((np.arange(2 ** n_nodes)[:, None] >> np.arange(n_nodes-1, -1, -1)) & 1) - 1
        states[:, nodes] = combs[:, :]

        batch_size = min(1024, total_states)

        for i in range(0, total_states, batch_size):
            end_idx = min(i + batch_size, total_states)
            batch_states = states[i:end_idx]
            temp = batch_states @ self.adj
            batch_energies = -0.5 * np.einsum('ij,ij->i', temp, batch_states)
            hamiltonian[i:end_idx] = batch_energies

        hamiltonian = -1 * (hamiltonian + 0.5 * np.sum(self.adj)) / 2
        best_energy = np.min(hamiltonian)


        if best_energy < self.c_energy:
            self.c_energy = best_energy
            self.solution = states[hamiltonian == best_energy][0]


    def solve(self):


        corrs = list(self.corr_dict.keys())
        weird_corrs = np.sort(corrs)[:self.number_of_edges]
        weird_edges = [self.corr_dict[corr] for corr in weird_corrs]

        subgraphs = generate_all_edge_subgraphs(G=self.graph,
                                                edges=weird_edges,
                                                max_size=self.max_size,
                                                weight_attr='w')

        for subgraph in subgraphs[:self.number_of_corrections]:
            self._fine_tuning_hamiltonian(subgraph)

        self.solution = ''.join(['1' if bit == 1 else '1' for bit in self.solution])




if __name__ == '__main__':
    G = ConnectedRandGraph(number_of_nodes=100,
                           number_of_edges=4950,
                           weights='pos_neg',
                           seed=1000)

    bits = dict(zip(G.nodes(), np.zeros(int(G.number_of_edges()))))
    nx.set_node_attributes(G, bits, "bit")

    lsq = LargeScaleRQAOA_1(graph=G,
                           max_size=10,
                           part_iter=1,
                           treshold=3,
                           n_layers=1,
                           w_scale=1,
                           logs=False,
                           callback=True,
                           max_iter=500,
                           random_seed=1000)
    lsq.solve()


    ft = FineTuning(graph=G,
                    solution=lsq.lsr_sol,
                    corr_dict=lsq.max_corr_dict,
                    max_size=10,
                    number_of_edges=100)
    ft.solve()