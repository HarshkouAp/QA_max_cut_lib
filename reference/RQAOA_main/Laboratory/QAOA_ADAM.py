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

    def __init__(self, graph, n_layers, max_iter=300, samples=25, logs=False,
                 callback=False, method='L-BFGS-B', opt_method='layerwise',
                 initialization='no ones'):

        self.graph = nx.Graph(graph)
        self.n_layers = n_layers
        self.max_iter = max_iter
        self.samples = samples
        self.logs = logs
        self.callback = callback
        self.method = method
        self.opt_method = opt_method
        self.initialization = initialization

        self.energies = np.array([])
        self.state_neds = np.array([])
        self.expectation_neds = np.array([])
        self.overlaps = np.array([])
        self.layers_times = np.array([])
        self.layers_energies = np.array([])
        self.layers_borders = np.array([])
        self.result, self.iterations = 0, 0

        self.n_nodes = nx.number_of_nodes(self.graph)
        self.state_list = 2 * ((np.arange(2**(self.n_nodes-1))[:, None]
                                >> np.arange(self.n_nodes-1, -1, -1)) & 1) - 1
        self.init_state = (np.ones(1 << (self.n_nodes - 1)) / (np.sqrt(2) ** self.n_nodes))

        self.c_solution = []
        self.q_solution = []
        self.q_solution_dct = {}

        self.c_energy, self.q_energy = 0., 0.
        self.cut_value = 0
        self.min_energy = 0
        self.max_energy = 0
        self.expectation_ned = 0
        self.state_ned = 0
        self.overlap = 0
        self.ground_state_idx = 0
        self.c_time, self.q_time = 0., 0.
        self.hamiltonian, self.mix_operator = np.array([]), np.array([])
        self.state, self.probs = np.array([]), np.array([])


    def mean_map(self):

        m_map = np.zeros((self.graph.number_of_nodes(), self.graph.number_of_nodes()))
        edges = list(self.graph.edges())
        probs = self.probs.reshape(-1, 1)

        m_map[[e[0] for e in edges], [e[1] for e in edges]] = np.sum(
            self.state_list[:, [e[0] for e in edges]] *
            self.state_list[:, [e[1] for e in edges]] *
            probs[:, [0 for _ in edges]], axis=0)

        return m_map

    def _init_approximation(self, optimal_angles):

        gammas = np.linspace(-2 * np.pi, 2 * np.pi, self.samples)
        betas = np.array([3 * np.pi / 4, 3 * np.pi / 8])
        e_arr = []

        cost_function = self._cost_function(depth = round(len(optimal_angles) / 2),
                                            mode = 'initialization')

        for g in gammas:

            angles_1 = optimal_angles.copy()
            angles_1 = np.insert(angles_1, round(len(optimal_angles) / 2), g)
            angles_1 = np.append(angles_1, betas[0])

            angles_2 = optimal_angles.copy()
            angles_2 = np.insert(angles_2, round(len(optimal_angles) / 2), g)
            angles_2 = np.append(angles_2, betas[1])

            e_arr.append([cost_function(angles_1), cost_function(angles_2)])

        e_arr = np.array(e_arr)

        AB_arr = np.zeros((self.samples, 2))
        AB_arr[:, 0] = -1 * e_arr[:, 0]
        AB_arr[:, 1] = -1 * (e_arr[:, 0] * (np.sqrt(2) / 2) + e_arr[:, 1])

        root = np.zeros(self.samples).reshape(-1, 1)
        coef = np.zeros(self.samples).reshape(-1, 1)
        root[:, 0] = np.sqrt(AB_arr[:, 0] ** 2 + 32 * AB_arr[:, 1] ** 2)
        coef[:, 0] = (1 / (8 * AB_arr[:, 1]))

        t_arr = np.zeros((self.samples, 2))
        t_arr[:, 0] = coef[:, 0] * ((-1) * AB_arr[:, 0] + root[:, 0])
        t_arr[:, 1] = coef[:, 0] * ((-1) * AB_arr[:, 0] - root[:, 0])

        x_arr = np.arccos(t_arr) / 2

        f_arr = np.zeros((self.samples, 2))
        f_arr[:, 0] = AB_arr[:, 0] * np.sin(2 * x_arr[:, 0]) + AB_arr[:, 1] * np.sin(4 * x_arr[:, 0])
        f_arr[:, 1] = AB_arr[:, 0] * np.sin(2 * x_arr[:, 1]) + AB_arr[:, 1] * np.sin(4 * x_arr[:, 1])

        gamma_idx, beta_idx = np.where(f_arr == f_arr.min())
        gamma = gammas[gamma_idx[0]]
        beta = x_arr[gamma_idx[0], beta_idx[0]]

        optimal_angles = np.insert(optimal_angles, round(len(optimal_angles) / 2), gamma)
        optimal_angles = np.append(optimal_angles, beta)

        return optimal_angles

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

    def _adam_optimization(self, init_params, depth):

        alpha, beta_1, beta_2, eps, diff = 0.05, 0.9, 0.999, 1e-8, 1e-6
        angles = init_params.copy()
        m = np.zeros_like(angles)
        v = np.zeros_like(angles)
        cost_function = self._cost_function(depth)
        grad_function = self._cost_function(depth, mode='initialization')

        for t in range(1, self.max_iter + 1):
            grad = np.zeros_like(angles)
            for i in range(len(angles)):
                shift = np.zeros_like(angles)
                shift[i] = diff
                grad[i] = (grad_function(angles + shift) - grad_function(angles - shift)) / (2 * diff)

            m = beta_1 * m + (1 - beta_1) * grad
            v = beta_2 * v + (1 - beta_2) * grad ** 2
            angles -= alpha * (m / (1 - beta_1 ** t)) / (np.sqrt(v / (1 - beta_2 ** t)) + eps)
            cost_function(angles)

        return angles

    def _cmaes_optimization(self, init_params, depth):

        n = len(init_params)
        sigma = 0.5
        lamb = 4 + int(3 * np.log(n))
        mu = lamb // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1 / np.sum(weights ** 2)

        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chi = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        mean = init_params.copy()
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        invsqrtC = np.eye(n)
        cost_function = self._cost_function(depth)
        best_angles = mean.copy()
        best_energy = cost_function(mean)

        for g in range(max(1, self.max_iter // lamb)):
            arz = np.random.randn(lamb, n)
            ary = (arz * D) @ B.T
            arx = mean + sigma * ary
            fitness = np.array([cost_function(x) for x in arx])
            idx = np.argsort(fitness)

            old_mean = mean.copy()
            mean = weights @ arx[idx[:mu]]
            y_w = (mean - old_mean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ y_w)
            hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * g + 2)) / chi < 1.4 + 2 / (n + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_w

            artmp = (arx[idx[:mu]] - old_mean) / sigma
            C = ((1 - c1 - cmu + (1 - hsig) * c1 * cc * (2 - cc)) * C +
                 c1 * np.outer(pc, pc) + cmu * ((weights[:, None] * artmp).T @ artmp))
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chi - 1))

            eig_values, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(eig_values, 1e-30))
            invsqrtC = B @ np.diag(1 / D) @ B.T

            if fitness[idx[0]] < best_energy:
                best_energy = fitness[idx[0]]
                best_angles = arx[idx[0]].copy()

        return best_angles

    def _simultaneous_optimization(self):

        start_time = time.time()
        init_params = np.ones(2 * self.n_layers)

        if self.method == 'ADAM':
            optimal_angles = self._adam_optimization(init_params, self.n_layers)
        elif self.method == 'CMA-ES':
            optimal_angles = self._cmaes_optimization(init_params, self.n_layers)
        else:
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

        self.probs = np.abs(self.state) ** 2
        self.q_solution = self.state_list[np.argmax(self.probs)]
        self._cut_value()

        self.state_neds = np.append(self.state_neds, np.round(self.NED(self.cut_value), 6))
        self.expectation_neds = np.append(self.expectation_neds, np.round(self.NED(-1 * self.q_energy), 6))
        self.overlaps = np.append(self.overlaps, self._ground_overlap(self.state))
        self.layers_times = np.append(self.layers_times, np.round(time.time() - start_time, 6))
        self.layers_energies = np.append(self.layers_energies, self.q_energy)

        self.q_time = time.time() - start_time

    def _layerwise_optimization(self):

        start_opt = time.time()

        if self.initialization == 'ones':
            init_params = np.ones(2)
        else:
            init_params = self._init_approximation(optimal_angles=np.array([]))

        for p in range(1, self.n_layers + 1):

            start_time = time.time()
            if self.method == 'ADAM':
                optimal_angles = self._adam_optimization(init_params, p)
            elif self.method == 'CMA-ES':
                optimal_angles = self._cmaes_optimization(init_params, p)
            else:
                result_min = minimize(self._cost_function(p),
                                      init_params,
                                      method=self.method,
                                      options={'maxiter': self.max_iter,
                                               'maxfun': self.max_iter,
                                               'gtol': 1e-5})
                optimal_angles = result_min.x

            if self.initialization == 'ones':
                init_params = np.insert(optimal_angles, round(len(optimal_angles) / 2), 1)
                init_params = np.append(init_params, 1)
            else:
                init_params = self._init_approximation(optimal_angles=optimal_angles)

            fin_function = self._cost_function(p, final=True)
            self.state, self.q_energy = fin_function(optimal_angles)
            self.layers_borders = np.append(self.layers_borders, self.iterations)

            self.probs = np.abs(self.state) ** 2
            self.q_solution = self.state_list[np.argmax(self.probs)]
            self._cut_value()

            self.state_neds = np.append(self.state_neds, np.round(self.NED(self.cut_value), 6))
            self.expectation_neds = np.append(self.expectation_neds, np.round(self.NED(-1 * self.q_energy), 6))
            self.overlaps = np.append(self.overlaps, self._ground_overlap(self.state))
            self.layers_times = np.append(self.layers_times, np.round(time.time() - start_time, 6))
            self.layers_energies = np.append(self.layers_energies, self.q_energy)

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
        self.q_solution = self.state_list[np.argmax(self.probs)]
        self.q_solution_dct = dict(zip(self.graph.nodes(), [bit for bit in self.q_solution]))
        self.result = 1 if (False not in (self.c_solution == self.q_solution)
                            or False not in (self.c_solution == -self.q_solution)) else 0
        self._cut_value()
        self.state_ned = self.state_neds[-1]
        self.expectation_ned = self.expectation_neds[-1]
        self.overlap = self.overlaps[-1]
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
            print(f'Total       ||   NED (st.) : {self.expectation_ned:.4f}   '
                  f'NED (exp.) : {self.expectation_ned:.4f}')
            print(f'Total       ||   Overlap : {self.overlap:.4f}')

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

    def _ground_overlap(self, state):
        return np.abs(state[self.ground_state_idx]) ** 2



if __name__ == '__main__':

    G = ConnectedRandGraph(number_of_nodes=12,
                           number_of_edges=40,
                           weights='pos_neg',
                           seed=90)

    qaoa = QAOA(graph=G,
                n_layers=10,
                logs=True,
                max_iter=10,
                samples=1000,
                callback=True,
                method="ADAM",
                initialization="no ones")
    qaoa.solve()

    print(qaoa.layers_times)
    print(qaoa.layers_energies)
    print(qaoa.overlaps)
    print(qaoa.state_neds)
    print(qaoa.expectation_neds)






















