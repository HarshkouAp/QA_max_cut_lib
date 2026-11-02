import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import time
import warnings
from itertools import product
from scipy.linalg import logm
import pandas as pd

plt.rcParams.update({"font.family": "serif"})
warnings.filterwarnings('ignore')

from Generators.GraphGenerator import ConnectedRandomGraph



class MutualInfoQAOA:

    def __init__(self, graph, n_layers, max_iter=300, logs=False,
                 callback=False, method='L-BFGS-B', opt_method='layerwise'):

        self.graph = nx.Graph(graph)
        self.n_layers = n_layers
        self.max_iter = max_iter
        self.logs = logs
        self.callback = callback
        self.method = method
        self.opt_method = opt_method

        self.n_nodes = nx.number_of_nodes(self.graph)
        self.state_list = 2 * ((np.arange(2**(self.n_nodes-1))[:, None]
                                >> np.arange(self.n_nodes-1, -1, -1)) & 1) - 1
        self.full_state_list = 2 * ((np.arange(2**self.n_nodes)[:, None]
                                >> np.arange(self.n_nodes-1, -1, -1)) & 1) - 1
        self.init_state = (np.ones(1 << (self.n_nodes - 1)) / (np.sqrt(2) ** self.n_nodes))
        self.dens_matrix = np.array([])

        self.energies = np.array([])
        self.layer_energies = np.array([])
        self.mi_maps = np.zeros((self.n_layers, self.n_nodes, self.n_nodes))
        self.dm_dct = {}
        self.mi_mean = np.array([])
        self.mi_min = np.array([])
        self.mi_max = np.array([])
        self.mi_std = np.array([])
        self.layers_borders = np.array([])
        self.result, self.iterations = 0, 0
        self.nonlocality, self.weighted_nonlocality = 0, 0

        self.c_solution = []
        self.q_solution = []
        self.q_solution_dct = {}

        self.c_energy, self.q_energy = 0., 0.
        self.c_time, self.q_time = 0., 0.
        self.hamiltonian, self.mix_operator = np.array([]), np.array([])
        self.state, self.probs = np.array([]), np.array([])

    def _compute_nonlocality(self):

        n_nodes = nx.number_of_nodes(self.graph)
        wsp_map = nx.floyd_warshall_numpy(self.graph)
        self.weighted_nonlocality = 2 * wsp_map.sum() / (n_nodes * (n_nodes - 1))

        weights = dict(zip(self.graph.edges(), np.ones(self.graph.number_of_edges())))
        nx.set_edge_attributes(self.graph, weights, "weight")
        usp_map = nx.floyd_warshall_numpy(self.graph)
        self.nonlocality = 2 * usp_map.sum() / (n_nodes * (n_nodes - 1))

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

    def _one_q_trace(self, q_ind_1, layer):

        part_trace = [np.sum(self.dens_matrix[
                                 (self.full_state_list[:, q_ind_1] == i),
                                 (self.full_state_list[:, q_ind_1] == k)])
                      for i, k in product([-1, 1], repeat=2)]

        self.dm_dct[(layer - 1, q_ind_1)] = np.array(part_trace).reshape(2, 2)

        return np.array(part_trace).reshape(2, 2)

    def _two_q_trace(self, q_ind_1, q_ind_2):

        part_trace = [np.sum(self.dens_matrix[
                                 (self.full_state_list[:, q_ind_1] == i) &
                                 (self.full_state_list[:, q_ind_2] == k),
                                 (self.full_state_list[:, q_ind_1] == m) &
                                 (self.full_state_list[:, q_ind_2] == l)])
                      for i, k, m, l in product([-1, 1], repeat=4)]

        return np.array(part_trace).reshape(4, 4)

    def _mutual_information_map(self, p):
        mi_map = np.zeros((self.n_nodes, self.n_nodes), dtype='complex128')
        for i , k in product(range(self.n_nodes), repeat=2):
            if i == k :
                continue
            else:
                S_i = - np.trace(self._one_q_trace(i, p) @ logm(self._one_q_trace(i, p)))
                S_k = - np.trace(self._one_q_trace(k, p) @ logm(self._one_q_trace(k, p)))
                S_ik = - np.trace(self._two_q_trace(i, k) @ logm(self._two_q_trace(i, k)))
                mi_map[i, k] = 0.5 * (S_i + S_k - S_ik)

        mi_map = np.real(mi_map)

        mean = np.sum(mi_map) / (self.n_nodes * (self.n_nodes - 1))
        mmax = np.max(mi_map)
        mmin = np.min(mi_map)
        std = np.std(mi_map)

        self.mi_mean = np.append(self.mi_mean, mean)
        self.mi_min = np.append(self.mi_min, mmin)
        self.mi_max = np.append(self.mi_max, mmax)
        self.mi_std = np.append(self.mi_std, std)

        return mi_map

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
            self.layers_borders = np.append(self.layers_borders, self.iterations)
            self.layer_energies = np.append(self.layer_energies, self.q_energy)

            full_state = np.hstack((self.state, self.state[::-1]))
            self.dens_matrix = np.outer(full_state.conj(), full_state)
            self.mi_maps[p - 1] = self._mutual_information_map(p)


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

        self._compute_nonlocality()
        MutualInfoQAOA._callback(self)

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
            print(f"Processing  ||   QAOA time : {np.round(self.q_time, 4)} s   "
                  f"Classic time : {np.round(self.c_time, 4)} s")
            print(f'Total       ||   Ratio : {np.round(-1 * self.q_energy / self.c_energy, 4)}')

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

    def plot_mi_maps(self):

        n_cols = int(np.ceil(np.sqrt(self.n_layers)))
        n_rows = int(np.ceil(self.n_layers / n_cols))

        # Создаем сетку с местом для цветовой шкалы
        fig, axes = plt.subplots(nrows=n_rows,
                                 ncols=n_cols,
                                 figsize=(4 * n_cols, 4 * n_rows),
                                 squeeze=False
                                )

        vmin = 0.
        vmax = np.max(self.mi_maps) * 1.1
        cbar_ax = fig.add_axes((0.92, 0.15, 0.03, 0.7))  # [left, bottom, width, height]

        axes = axes.flatten()
        for i in range(self.n_layers, len(axes)):
            axes[i].set_visible(False)

        for i in range(self.n_layers):
            if i < self.n_layers - 1:
                sns.heatmap(self.mi_maps[i],
                            ax=axes[i],
                            cmap='viridis',
                            square=True,
                            vmin=vmin,
                            vmax=vmax,
                            cbar=False)
            else:
                hm = sns.heatmap(self.mi_maps[i],
                                 ax=axes[i],
                                 cmap='viridis',
                                 square=True,
                                 vmin=vmin,
                                 vmax=vmax,
                                 cbar=True,
                                 cbar_ax=cbar_ax)

            axes[i].set_title(f'Layer {i + 1}', fontsize=30)
            axes[i].set_xticks([])
            axes[i].set_yticks([])

        cbar = hm.collections[0].colorbar
        cbar.ax.tick_params(labelsize=30)

        plt.subplots_adjust(
            left=0.349,  # Отступ слева (5% от ширины фигуры)
            right=0.9,  # Отступ справа (освобождаем место для colorbar)
            bottom=0.01,  # Отступ снизу
            top=0.96,  # Отступ сверху
            wspace=0.035,  # Горизонтальное расстояние между subplot-ами
            hspace=0.320  # Вертикальное расстояние между subplot-ами
        )

        plt.tight_layout(rect=(0, 0, 0.9, 1))  # Оставляем место справа для цветовой шкалы
        plt.savefig('RQAOA_data/test_7/QAOA_mims_line.png',
                    dpi=300,  # Качество
                    bbox_inches='tight',  # Обрезает пустые края
                    pad_inches=0.1,  # Минимальный отступ от края
                    facecolor='white',  # Цвет фона
                    edgecolor='none')  # Без рамки
        plt.show()

    def plot_information(self):

        means = np.sum(self.mi_maps, axis=(1, 2)) / (self.n_nodes * (self.n_nodes - 1))
        maxs = np.max(self.mi_maps, axis=(1, 2))

        mins = [np.min(self.mi_maps[k][self.mi_maps[k]>0]) for k in range(self.n_layers)]
        stds = [np.std(self.mi_maps[k][self.mi_maps[k]>0]) for k in range(self.n_layers)]

        x_points = np.arange(1, self.n_layers + 1)

        plt.figure(figsize=(8, 6))
        sns.lineplot(x=x_points, y=means, label='Mean', marker='o', linewidth=2)
        sns.lineplot(x=x_points, y=mins, label='Min', marker='s', linestyle='--')
        sns.lineplot(x=x_points, y=maxs, label='Max', marker='^', linestyle='--')

        plt.fill_between(x_points,
                         np.array(means) - np.array(stds),
                         np.array(means) + np.array(stds),
                         alpha=0.2, label='STD')

        plt.xlabel('Layers of QAOA', fontsize=14)
        plt.ylabel('Mutual information', fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.title(f'Plot of the evolution of mutual information in QAOA\n'
                  f'Number of nodes: {self.n_nodes}   '
                  f'Number of edges {nx.number_of_edges(self.graph)}   '
                  f'Nonlocality: {self.nonlocality / 2:.3f}',
                  fontsize=16)
        plt.legend(
            loc='lower right',
            fontsize=12,  # Размер шрифта
            fancybox=True)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('RQAOA_data/test_7/QAOA_mi_trend.png',
                    dpi=300,  # Качество
                    bbox_inches='tight',  # Обрезает пустые края
                    pad_inches=0.1,  # Минимальный отступ от края
                    facecolor='white',  # Цвет фона
                    edgecolor='none')  # Без рамки
        plt.show()

    def plot_qubit_information(self):

        x_points = np.arange(1, self.n_layers + 1)
        plt.figure(figsize=(10, 6))

        for k, j in product(range(self.n_nodes), repeat=2):
            sns.lineplot(x=x_points,
                         y=self.mi_maps[: , k, j],
                         linewidth=1)

        plt.xlabel('Layers', fontsize=14)
        plt.ylabel('Information', fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.title(f'Plot of MI tracks for every component of matrix \n'
                  f'Number of nodes: {self.n_nodes}   Number of edges {nx.number_of_edges(self.graph)}',
                  fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(
            loc='lower right',
            fontsize=12,  # Размер шрифта
            fancybox=True,  # Скругленные углы рамки
        )
        plt.tight_layout()
        plt.show()

    def plot_shortest_path(self):

        weights = dict(zip(self.graph.edges(), np.ones(self.graph.number_of_edges())))
        nx.set_edge_attributes(self.graph, weights, "weight")
        disc_dist_matrix = nx.floyd_warshall_numpy(self.graph)

        data = pd.DataFrame(columns=['layer', 'value', 'Distance'])
        for layer, mi_map in enumerate(self.mi_maps):
            df = pd.DataFrame({'layer': np.ones(self.n_nodes ** 2) * layer,
                               'value': mi_map.flatten(),
                               'Distance': disc_dist_matrix.flatten()})
            data = pd.concat([data, df], ignore_index=True)

        plt.figure(figsize=(8, 6))
        sns.lineplot(data=data[data['Distance'] != 0], x="layer", y="value", hue="Distance")

        plt.xlabel('Layers of QAOA', fontsize=12)
        plt.ylabel('Mean mutual information', fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.title(f'A graph of the evolution of the average mutual information \n'
                  f'for vertices separated by different distances.\n'
                  f'Number of nodes: {self.n_nodes}   '
                  f'Number of edges: {nx.number_of_edges(self.graph)}   '
                  f'Nonlocality: {self.nonlocality / 2:.3f}',
                  fontsize=14)
        plt.legend(
            loc='lower right',
            title='Distance\nbetween\n'
                  'vertices',
            fontsize=10,  # Размер шрифта
            title_fontsize=10,
            fancybox=True,  # Скругленные углы рамки
        )
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('RQAOA_data/test_7/QAOA_dist_info.png',
                    dpi=300,  # Качество
                    bbox_inches='tight',  # Обрезает пустые края
                    pad_inches=0.1,  # Минимальный отступ от края
                    facecolor='white',  # Цвет фона
                    edgecolor='none')  # Без рамки
        plt.show()



if __name__ == '__main__':

    G = ConnectedRandomGraph(number_of_nodes=12,
                             number_of_edges=11,
                             seed=42)

    q = MutualInfoQAOA(G,
                       n_layers=16,
                       max_iter=2000,
                       logs=True)
    q.solve()
    # q.plot_convegence()
    # q.plot_mi_maps()
    # q.plot_information()
    # q.plot_qubit_information()
    q.plot_shortest_path()









