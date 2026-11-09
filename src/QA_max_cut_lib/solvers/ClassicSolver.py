import time
import numpy as np
import networkx as nx
from Generators.GraphGenerator import DensRandGraph



class ClassicSolver:

    def __init__(self, graph):

        self.graph = nx.Graph(graph)
        self.n_nodes = graph.number_of_nodes()
        self.state_list = 2 * ((np.arange(2 ** (self.n_nodes - 1))[:, None]
                                >> np.arange(self.n_nodes - 1, -1, -1)) & 1) - 1
        self.hamiltonian = np.zeros(1 << self.n_nodes - 1)
        self.c_energy = 0
        self.max_energy = 0
        self.c_solution = []
        self.solution_dct = {}
        self.c_time = 0
        self.incorrect_energy = 0


    def solve(self):

        start_time = time.time()
        total_states = 1 << (self.n_nodes - 1)
        adj = nx.adjacency_matrix(self.graph)
        batch_size = min(1000, total_states)

        for i in range(0, total_states, batch_size):
            end_idx = min(i + batch_size, total_states)
            batch_states = self.state_list[i:end_idx]
            temp = batch_states @ adj
            batch_energies = -0.5 * np.einsum('ij,ij->i', temp, batch_states)
            self.hamiltonian[i:end_idx] = batch_energies

        self.hamiltonian = -1 * (self.hamiltonian + 0.5 * np.sum(adj)) / 2
        self.c_energy = np.min(self.hamiltonian)
        self.max_energy = np.max(self.hamiltonian)
        self.c_solution = self.state_list[self.hamiltonian == self.c_energy][0]
        self.solution_dct = dict(zip(self.graph.nodes(), [bit for bit in self.c_solution]))
        self.c_time = time.time() - start_time
        self.incorrect_energy = np.max(self.hamiltonian)


if __name__ == '__main__':

    G = DensRandGraph(
        number_of_nodes=25,
        density=0.5,
        random_weights=False,
        seed=42)

    c = ClassicSolver(G)
    c.solve()
    print(c.c_energy)
    print(c.c_time)