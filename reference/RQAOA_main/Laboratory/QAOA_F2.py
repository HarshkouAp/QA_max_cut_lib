import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import time
import warnings
plt.rcParams.update({"font.family": "serif"})
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '8'

from Generators.GraphGenerator import ConnectedRandGraph


class QAOA:

    def __init__(self, graph, full_graph, n_layers, max_iter=300, samples=25,
                 alpha=1., beta=1., gamma=1.,
                 logs=False, callback=False, method='L-BFGS-B', opt_method='layerwise'):

        self.graph = nx.Graph(graph)
        self.full_graph = nx.Graph(full_graph)
        self.n_layers = n_layers

        self.max_iter = max_iter
        self.samples = samples
        self.logs = logs
        self.callback = callback
        self.method = method
        self.opt_method = opt_method
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.energies = np.array([])
        self.layers_borders = np.array([])
        self.result, self.iterations = 0, 0

        self.n_nodes = nx.number_of_nodes(self.graph)
        self.state_list = 2 * ((np.arange(2**(self.n_nodes-1))[:, None]
                                >> np.arange(self.n_nodes-1, -1, -1)) & 1) - 1
        self.init_state = (np.ones(1 << (self.n_nodes - 1)) / (np.sqrt(2) ** self.n_nodes))

        self.c_solution = []
        self.q_solution = []
        self.q_solution_dct = {}
        self.q_bits_probs = {}

        self.c_energy, self.q_energy = 0., 0.
        self.cut_value = 0
        self.max_energy = 0
        self.ned = 0
        self.c_time, self.q_time = 0., 0.
        self.hamiltonian, self.mix_operator = np.array([]), np.array([])
        self.state, self.probs = np.array([]), np.array([])


    @staticmethod
    def sigmoid(x):
        return 2 * (1 / (1 + np.exp(-x)) - 0.5)

    def field(self, alpha, beta, gamma):

        print("==" * 30)

        field = np.zeros(self.n_nodes)
        w_int = np.zeros(self.n_nodes)
        bits = nx.get_node_attributes(self.full_graph, "bit")
        graph_nodes = list(self.graph.nodes())

        for node_id, node in enumerate(self.graph.nodes()):
            items = list(self.graph[node].items())
            data = np.array([
                [
                    attrs.get("weight"),
                    attrs.get("corr"),
                    bits[neighbor],
                ]
                for neighbor, attrs in items
            ])

            W = np.sum(data[::, 0])
            w_int[node_id] = W


        for node_id, node in enumerate(self.graph.nodes()):
            items = list(self.full_graph[node].items())
            data = np.array([
                [
                    attrs.get("weight"),
                    attrs.get("corr"),
                    bits[neighbor],
                ]
                for neighbor, attrs in items if neighbor not in graph_nodes
            ])

            if len(data) == 0:
                field[node_id] = 0.
                continue

            P = beta * data[::, 1] * data[::, 2]
            W = np.sum(data[::, 0])
            N = data.shape[0]
            D_plus = np.prod(1 + P)
            D_minus = np.prod(1 - P)
            F1 = (W / N) * (2 * (D_plus / (D_plus + D_minus)) - 1)

            if W == 0:
                F2 = 0
            elif np.abs(W) < np.abs(w_int[node_id]):
                F2 = gamma * np.sum(np.abs(data[::, 0]) * data[::, 1] * data[::, 2])
            else:
                F2 = (gamma * np.abs(w_int[node_id] / W) *
                      np.sum(np.abs(data[::, 0]) * data[::, 1] * data[::, 2]))

            field[node_id] = alpha * (F1 + F2)

            print(f"w: {w_int[node_id]}   W: {W}   F1: {F1:.3f}   F2: {F2:.3f}   F: {(F1 + F2):.3f}")

        return field

    def _maxcut_classical_solver(self):

        start_time = time.time()
        total_states = 1 << (self.n_nodes - 1)
        adj = nx.adjacency_matrix(self.graph)
        self.hamiltonian = np.zeros(1 << self.n_nodes - 1)

        batch_size = min(1024, total_states)
        for i in range(0, total_states, batch_size):
            end_idx = min(i + batch_size, total_states)
            batch_states = self.state_list[i:end_idx]
            temp = batch_states @ adj
            batch_energies = -0.5 * np.einsum('ij,ij->i', temp, batch_states)
            self.hamiltonian[i:end_idx] = batch_energies

        self.hamiltonian = -1 * (self.hamiltonian + 0.5 * np.sum(adj)) / 2
        field = self.field(1, 0, 1)
        self.hamiltonian -= np.sum(self.state_list * field, axis=1)

        if self.n_nodes == self.graph.number_of_nodes():
            self.max_energy = np.max(self.hamiltonian)
            self.c_energy = np.min(self.hamiltonian)
            self.c_solution = self.state_list[self.hamiltonian == self.c_energy][0]
            self.c_time = time.time() - start_time

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

    def _cost_function(self, depth, final=False, mode='optimization'):

        def function(angles):
            gamma = angles[0:depth]
            beta = angles[depth:2 * depth]
            state = np.copy(self.init_state)
            for k in range(depth):
                vector = np.exp(-1j * gamma[k] * self.hamiltonian) * state
                for j in range(self.n_nodes):
                    vector = (vector * np.cos(beta[k]) - 1j * vector[self.mix_operator[j]] * np.sin(beta[k]))
                state = vector
            energy = 2 * np.sum(np.conj(state) * self.hamiltonian * state)

            if mode == 'optimization':
                self.energies = np.append(self.energies, np.real(energy))
                self.iterations += 1

            if final:
                return state, np.abs(energy)
            else:
                return np.real(energy)

        return function

    def _simultaneous_optimization(self):

        start_opt = time.time()
        init_params = np.ones(2 * self.n_layers)

        result_min = minimize(self._cost_function(self.n_layers),
                              init_params,
                              method=self.method,
                              options={'maxiter': self.max_iter,
                                       'maxfun': self.max_iter,
                                       'gtol': 1e-5})

        optimal_angles = result_min.x
        fin_function = self._cost_function(self.n_layers, final=True)
        self.state, self.q_energy = fin_function(optimal_angles)
        self.layers_borders = np.append(self.layers_borders, self.iterations)

        self.q_time = time.time() - start_opt

    def _layerwise_optimization(self):

        start_opt = time.time()
        init_params = np.ones(2)

        for p in range(1, self.n_layers + 1):

            start_time = time.time()
            result_min = minimize(self._cost_function(p),
                                  init_params,
                                  method=self.method,
                                  options={'maxiter': self.max_iter,
                                           'maxfun': self.max_iter,
                                           'gtol': 1e-5})
            optimal_angles = result_min.x
            init_params = np.insert(optimal_angles, round(len(optimal_angles) / 2), 1)
            init_params = np.append(init_params, 1)
            fin_function = self._cost_function(p, final=True)
            self.state, self.q_energy = fin_function(optimal_angles)
            self.layers_borders = np.append(self.layers_borders, self.iterations)

            if self.logs:
                print("\033[37m {}".format(
                      f"Layer : {p:<2d}  "
                      f"Energy : {self.q_energy:<4f}   "
                      f"Processing time : {(time.time() - start_time):<4f}"))

        self.q_time = time.time() - start_opt

    def solve(self):

        self._maxcut_classical_solver()
        self._mix_operator_preset()

        if self.opt_method == 'layerwise':
            self._layerwise_optimization()
        elif self.opt_method == 'simultaneous':
            self._simultaneous_optimization()
        else:
            raise ValueError(f"Unknown optimization method: {self.opt_method}")

        self.probs = 2 * np.abs(self.state) ** 2
        self.q_solution = self.state_list[np.argmax(self.probs)]
        self.q_solution_dct = dict(zip(self.graph.nodes(), [bit for bit in self.q_solution]))
        self.q_bits_probs = dict(zip(self.graph.nodes(), np.sum(self.state_list * self.probs[:, np.newaxis], axis=0)))
        self.result = 1 if (False not in (self.c_solution == self.q_solution)
                            or False not in (self.c_solution == -self.q_solution)) else 0
        self._cut_value()
        self.ned = self.NED(-1 * self.q_energy)
        QAOA._callback(self)

    def _cut_value(self):

        edges_weights = nx.get_edge_attributes(self.graph, "weight")
        mapping = dict(zip(self.graph.nodes(), range(self.graph.number_of_nodes())))
        self.cut_value = -1 * np.sum([edges_weights[(k, j)]
                                      if (self.q_solution[mapping[k]] != self.q_solution[mapping[j]]) else 0
                                      for k, j in self.graph.edges()])

    def _callback(self):

        if self.callback:

            print("\033[37m {}".format(f"-------------------------------------"))

            if self.result == 1:
                print("\033[32m {}".format("***SUCCESS***"))
            else:
                print("\033[31m {}".format("***FAIL***"))

            print(f"Graph       ||   number of nodes : {self.n_nodes}  "
                  f"number of edges : {nx.number_of_edges(self.graph)} ")
            print(f"QAOA        ||   number of layers : {self.n_layers}   "
                  f"optimization method : {self.method}")
            print(f"Classical   ||   solution : {''.join([str(int((bit + 1) / 2)) for bit in self.c_solution])}    "
                  f"Energy : {np.round(self.c_energy, 4)}")
            print(f"Quantum     ||   solution : {''.join([str(int((bit + 1) / 2)) for bit in self.q_solution])}    "
                  f"Cut : {self.cut_value}   "
                  f"Energy : {np.round(-1 * self.q_energy, 4)}")
            print(f"Processing  ||   QAOA time : {np.round(self.q_time, 4)} s   "
                  f"Classic time : {np.round(self.c_time, 4)} s")
            print(f'Total       ||   NED (st.) : {np.round(self.NED(self.cut_value), 4)}   '
                  f'NED (exp.) : {np.round(self.NED(-1 * self.q_energy), 4)}')

    def plot_convegence(self):

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(list(range(self.iterations)),
                 self.energies,
                 linewidth=0.7,
                 label='Convergence line')

        for coord in self.layers_borders:
            ax.axvline(x=coord,
                       color='red',
                       linestyle='--',
                       alpha=0.7,
                       linewidth=0.7)

        ax.axhline(y= self.c_energy,
                   color='green',
                   linestyle='--',
                   alpha=0.7,
                   linewidth=0.7,
                   label='Exact energy')

        ax.axhline(y=0,
                   color='red',
                   linestyle='--',
                   alpha=0.7,
                   linewidth=0.7,
                   label='Parameter shift')

        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        max_y = 0.9 * max(self.energies)
        min_y = 1.1 * min(self.energies)
        ax.set_ylim(min_y, max_y)

        plt.title(f'Energy convergence graph for parameter optimization for layerwise-QAOA\n'
                  f'Number of nodes: {self.n_nodes}   Number of edges {nx.number_of_edges(self.graph)}   '
                  f'QAOA layers: {self.n_layers}   Ratio: {np.round(-1 * self.q_energy / self.c_energy, 4)}',
                  fontsize=16, pad=20)
        plt.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlabel('Optimization iterations', fontsize=16, labelpad=10)
        ax.set_ylabel('Energy', fontsize=16, labelpad=10)
        ax.legend(loc='upper right', fontsize=12)
        sns.despine()
        plt.show()

    def NED(self, energy):
        return 1 - (energy - self.c_energy) / (self.max_energy - self.c_energy)



if __name__ == '__main__':

    G = ConnectedRandGraph(number_of_nodes=16,
                           number_of_edges=60,
                           weights='pos_neg')


    bits = dict(zip(G.nodes(), np.round(2 * (0.5 - np.random.sample(int(G.number_of_edges())))).astype(int)))
    nx.set_node_attributes(G, bits, "bit")

    H = nx.subgraph(G, [5, 6, 7, 8, 10])

    qaoa = QAOA(graph=H,
                full_graph=G,
                n_layers=1,
                field_amp=0.5,
                logs=True,
                max_iter=1000,
                samples=1000,
                callback=True)

    qaoa.solve()
    qaoa.plot_convegence()

