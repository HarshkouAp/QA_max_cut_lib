import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import time
import warnings
plt.rcParams.update({"font.family": "serif"})
warnings.filterwarnings('ignore')

from Generators.GraphGenerator import DensRandGraph



class D_QAOA:

    def __init__(self, graph, n_layers, frack=(np.pi / 8), max_iter=300, logs=False,
                 callback=False, method='L-BFGS-B', opt_method='layerwise'):

        self.graph = graph
        self.n_layers = n_layers
        self.frack = frack
        self.max_iter = max_iter
        self.logs = logs
        self.callback = callback
        self.method = method
        self.opt_method = opt_method

        self.energies = np.array([])
        self.layers_borders = np.array([])
        self.result, self.iterations = 0, 0

        self.n_nodes = nx.number_of_nodes(self.graph)
        self.state_list = 2 * ((np.arange(2**(self.n_nodes-1))[:, None]
                                >> np.arange(self.n_nodes-1, -1, -1)) & 1) - 1
        self.init_state = (np.ones(1 << (self.n_nodes - 1)) / (np.sqrt(2) ** self.n_nodes))

        self.c_solution = []
        self.q_solution = []
        self.d_solution = []
        self.q_solution_dct = {}
        self.d_solution_dct = {}

        self.c_energy, self.q_energy, self.d_energy = 0., 0., 0.
        self.c_time, self.q_time = 0., 0.
        self.hamiltonian, self.mix_operator = np.array([]), np.array([])
        self.state, self.probs = np.array([]), np.array([])
        self.d_state, self.d_probs = np.array([]), np.array([])
        self.d_params = np.array([])

    def _maxcut_classical_solver(self):

        start_time = time.time()
        total_states = 1 << (self.n_nodes - 1)
        adj = nx.adjacency_matrix(self.graph)
        self.hamiltonian = np.zeros(1 << self.n_nodes - 1)

        batch_size = min(1000, total_states)

        for i in range(0, total_states, batch_size):
            end_idx = min(i + batch_size, total_states)
            batch_states = self.state_list[i:end_idx]
            temp = batch_states @ adj
            batch_energies = -0.5 * np.einsum('ij,ij->i', temp, batch_states)
            self.hamiltonian[i:end_idx] = batch_energies

        self.hamiltonian = -1 * (self.hamiltonian + 0.5 * np.sum(adj)) / 2
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

    def _cost_function(self, depth, final=False):

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
            optimal_angles = np.array(optimal_angles)
            discrete_angles = self.frack * np.round(optimal_angles / self.frack)
            self.d_params = discrete_angles
            self.d_state, self.d_energy = fin_function(discrete_angles)
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

        self.probs = np.abs(self.state) ** 2
        self.d_probs = np.abs(self.d_state) ** 2
        self.q_solution = self.state_list[np.argmax(self.probs)]
        self.d_solution = self.state_list[np.argmax(self.d_probs)]
        self.q_solution_dct = dict(zip(self.graph.nodes(), [bit for bit in self.q_solution]))
        self.d_solution_dct = dict(zip(self.graph.nodes(), [bit for bit in self.d_solution]))
        self.result = 1 if (False not in (self.c_solution == self.q_solution)
                            or False not in (self.c_solution == -self.q_solution)) else 0
        D_QAOA._callback(self)

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
                  f"Energy : {np.round(-1 * self.q_energy, 4)}")
            print(f"Discrete    ||   solution : {''.join([str(int((bit + 1) / 2)) for bit in self.d_solution])}    "
                  f"Energy : {np.round(-1 * self.d_energy, 4)}")
            print(f"Processing  ||   QAOA time : {np.round(self.q_time, 4)} s   "
                  f"Classic time : {np.round(self.c_time, 4)} s")
            print(f'Total       ||   Ratio : {np.round(-1 * self.q_energy / self.c_energy, 4)}   '
                  f'D-Ratio : {np.round(-1 * self.d_energy / self.c_energy, 4)}')
            print(f'Discrete    ||   Frack : {self.frack:.4f}   '
                  f'Params : {np.round(self.d_params, 4)}')

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

        ax.axhline(y=(-1) * self.d_energy,
                   color='blue',
                   linestyle='--',
                   alpha=0.7,
                   linewidth=0.7,
                   label='Disc. energy')

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



if __name__ == '__main__':

    G = DensRandGraph(number_of_nodes=10,
                      density=0.7,
                      weights='random',
                      seed=42)


    q = D_QAOA(G,
               n_layers=5,
               logs=True,
               callback=True,
               frack=(np.pi / 8))
    q.solve()
    q.plot_convegence()









