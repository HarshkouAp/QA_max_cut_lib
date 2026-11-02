import numpy as np
from scipy.optimize import minimize
import networkx as nx
from Solvers.GoemansWilliamson import *
from Generators.GraphGenerator import ConnectedRandGraph
from Tools.GraphDecomposer import *


import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif"})


class LocalQAOA:

    def __init__(self, graph, n_layers, max_iter=1, method='L-BFGS-B', initialization='ones'):
        self.graph = nx.Graph(graph)
        self.n_layers = n_layers
        self.max_iter = max_iter
        self.method = method
        self.initialization = initialization

        self.n_nodes = nx.number_of_nodes(self.graph)
        self.nodes = list(self.graph.nodes())

        self.state_list = 2 * ((np.arange(2 ** (self.n_nodes - 1))[:, None]
                                >> np.arange(self.n_nodes - 1, -1, -1)) & 1) - 1
        self.init_state = np.ones(1 << (self.n_nodes - 1), dtype=complex) / (np.sqrt(2) ** self.n_nodes)

        self.hamiltonian = np.array([])
        self.mix_operator = np.array([])
        self.state = np.array([])
        self.probs = np.array([])
        self.angles = np.ones(2 * self.n_layers)
        self.energy = 0.0

        self._mix_operator_preset()

    def _mix_operator_preset(self):
        state_list = np.array([("0" * self.n_nodes)[:self.n_nodes - len(bin(s)[2:])] + bin(s)[2:]
                               for s in range(1 << (self.n_nodes - 1))])
        preset = []
        for qubit in range(1, self.n_nodes + 1):
            permutations = []
            for x in state_list:
                if x[qubit - 1] == "0":
                    perm = int(x, 2) + 2 ** (self.n_nodes - qubit)
                    if perm >= 2 ** (self.n_nodes - 1):
                        permutations.append(2 ** self.n_nodes - perm - 1)
                    else:
                        permutations.append(perm)
                else:
                    perm = int(x, 2) - 2 ** (self.n_nodes - qubit)
                    if perm >= 2 ** (self.n_nodes - 1):
                        permutations.append(2 ** self.n_nodes - perm - 1)
                    else:
                        permutations.append(perm)
            preset.append(permutations)
        self.mix_operator = np.array(preset)

    def build_hamiltonian(self, field=None):
        adj = nx.to_numpy_array(self.graph, nodelist=self.nodes, weight='weight')
        h = np.zeros(self.n_nodes) if field is None else np.asarray(field, dtype=float)

        total_states = 1 << (self.n_nodes - 1)
        self.hamiltonian = np.zeros(total_states)
        batch_size = min(1000, total_states)

        const = -0.25 * np.sum(adj)

        for i in range(0, total_states, batch_size):
            j = min(i + batch_size, total_states)
            batch_states = self.state_list[i:j]
            temp = batch_states @ adj
            internal = 0.25 * np.einsum('ij,ij->i', temp, batch_states)
            external = batch_states @ h
            self.hamiltonian[i:j] = const + internal + external

    def _cost_function(self, depth, final=False):
        def function(angles):
            gamma = angles[:depth]
            beta = angles[depth:2 * depth]
            state = np.copy(self.init_state)

            for k in range(depth):
                vector = np.exp(-1j * gamma[k] * self.hamiltonian) * state
                for j in range(self.n_nodes):
                    vector = vector * np.cos(beta[k]) - 1j * vector[self.mix_operator[j]] * np.sin(beta[k])
                state = vector

            energy = 2 * np.sum(np.conj(state) * self.hamiltonian * state)

            if final:
                return state, np.real(energy)
            return np.real(energy)

        return function

    def optimize(self):
        result = minimize(
            self._cost_function(self.n_layers),
            self.angles,
            method=self.method,
            options={'maxiter': self.max_iter, 'maxfun': self.max_iter, 'gtol': 1e-5}
        )
        self.angles = result.x
        self.state, self.energy = self._cost_function(self.n_layers, final=True)(self.angles)
        self.probs = np.abs(self.state) ** 2

    def spin_means(self):
        return 2 * np.sum(self.state_list * self.probs.reshape(-1, 1), axis=0)

    def bit_means(self):
        return 0.5 * (1 - self.spin_means())

    def most_probable_state(self):
        return self.state_list[np.argmax(self.probs)]


class DistributedQAOA:

    def __init__(self, graph, partition, n_layers, max_iter=1, outer_max_iter=20,
                 tol=1e-4, method='L-BFGS-B', initialization='ones', seed=0):
        self.graph = nx.Graph(graph)
        self.partition = [list(part) for part in partition]
        self.n_layers = n_layers
        self.max_iter = max_iter
        self.outer_max_iter = outer_max_iter
        self.tol = tol
        self.method = method
        self.initialization = initialization
        self.rng = np.random.default_rng(seed)

        self.blocks = []
        for part in self.partition:
            block = LocalQAOA(
                graph=self.graph.subgraph(part).copy(),
                n_layers=self.n_layers,
                max_iter=self.max_iter,
                method=self.method,
                initialization=self.initialization,
            )
            self.blocks.append(block)

        self.spin_mean_dct = {node: float(self.rng.choice([-1, 1])) for node in self.graph.nodes()}
        self.bit_mean_dct = {}
        self.q_solution_dct = {}
        self.history = []
        self.block_energy_history = [[] for _ in self.blocks]
        self.total_energy_history = []

    def _external_field(self, part):
        part_set = set(part)
        field = np.zeros(len(part))
        for i, u in enumerate(part):
            for v in self.graph.neighbors(u):
                if v not in part_set:
                    w = self.graph[u][v].get('weight', 1.0)
                    field[i] += 0.5 * w * self.spin_mean_dct[v]
        return field

    def _update_solution(self):
        self.bit_mean_dct = {u: 0.5 * (1 - self.spin_mean_dct[u]) for u in self.graph.nodes()}
        self.q_solution_dct = {u: int(self.spin_mean_dct[u] < 0) for u in self.graph.nodes()}

    def cut_value(self):
        return sum(
            data.get('weight', 1.0)
            for u, v, data in self.graph.edges(data=True)
            if self.q_solution_dct[u] != self.q_solution_dct[v]
        )

    def solve(self):
        for _ in range(self.outer_max_iter):
            new_spin_mean_dct = dict(self.spin_mean_dct)
            total_energy = 0.0

            for k, (block, part) in enumerate(zip(self.blocks, self.partition)):
                field = self._external_field(part)
                block.build_hamiltonian(field=field)
                block.optimize()
                self.block_energy_history[k].append(-block.energy)
                total_energy += -block.energy

                local_spin_means = block.spin_means()
                for u, m in zip(block.nodes, local_spin_means):
                    new_spin_mean_dct[u] = m

            self.total_energy_history.append(total_energy)
            delta = max(abs(new_spin_mean_dct[u] - self.spin_mean_dct[u]) for u in self.graph.nodes())
            self.history.append(delta)
            self.spin_mean_dct = new_spin_mean_dct

            if delta < self.tol:
                break

        self._update_solution()

    def plot_convergence(self):
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        x = np.arange(1, len(self.total_energy_history) + 1)

        for k, energies in enumerate(self.block_energy_history):
            ax1.plot(x, energies, linewidth=1.2, label=f'Subgraph {k + 1}')

        ax1.grid(True, which='both', linestyle='--', alpha=0.5)
        ax1.set_title(
            f'Distributed QAOA convergence by subgraphs\n'
            f'Number of nodes: {self.graph.number_of_nodes()}   '
            f'Number of edges: {self.graph.number_of_edges()}   '
            f'Layers: {self.n_layers}',
            fontsize=16,
            pad=20,
        )
        ax1.set_xlabel('Outer iteration', fontsize=16, labelpad=10)
        ax1.set_ylabel('Subgraph energy', fontsize=16, labelpad=10)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.legend(loc='best', fontsize=12)
        sns.despine()

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(x, self.total_energy_history, linewidth=1.2, label='Total energy')
        ax2.grid(True, which='both', linestyle='--', alpha=0.5)
        ax2.set_title(
            f'Distributed QAOA convergence by total energy\n'
            f'Number of nodes: {self.graph.number_of_nodes()}   '
            f'Number of edges: {self.graph.number_of_edges()}   '
            f'Layers: {self.n_layers}',
            fontsize=16,
            pad=20,
        )
        ax2.set_xlabel('Outer iteration', fontsize=16, labelpad=10)
        ax2.set_ylabel('Total energy', fontsize=16, labelpad=10)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2.legend(loc='best', fontsize=12)
        sns.despine()
        plt.show()


if __name__ == '__main__':
    rng = np.random.default_rng(7)

    G = ConnectedRandGraph(number_of_nodes=80,
                           number_of_edges=4000,
                           weights='random',
                           seed=42)

    config = SubgraphConfig(
        max_size=8,
        num_subgraphs=15,
        allow_overlap=True,
        random_seed=42)

    generator = OptimizedDenseSubgraphGenerator(G, config)
    partition = generator.generate_multiple_subgraphs()


    dqaoa = DistributedQAOA(
        graph=G,
        partition=partition,
        n_layers=1,
        max_iter=1,
        outer_max_iter=500,
        tol=1e-4,
        seed=1
    )
    dqaoa.solve()
    dqaoa.plot_convergence()

    print('history:', dqaoa.history)
    print('spin means:', dqaoa.spin_mean_dct)
    print('bit means:', dqaoa.bit_mean_dct)
    print('solution:', dqaoa.q_solution_dct)
    print('cut value:', dqaoa.cut_value())

    gw = GoemansWilliamsonMaxcut(graph=G,
                                 no_hyperplane_samples=100,
                                 callback=True)
    gw.solve()



