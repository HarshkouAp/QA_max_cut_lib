from Tools.GraphDecomposer import *
from MI_QAOA import MutualInfoQAOA
from Generators.GraphGenerator import ConnectedRandomGraph
import networkx as nx
from scipy.sparse import csr_array
from itertools import product
from scipy.linalg import logm
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pandas as pd

ONE_QUBIT_ENTROPY = 0.6931471805599453
TWO_QUBIT_ENTROPY = 0.34657359027997264
ONE_QUBIT_DM = np.eye(2, dtype=np.float64) * 0.5


class MI_RQAOA:

    def __init__(self, graph, treshold, n_layers, max_iter=300,
                 logs=False, callback=True, method='L-BFGS-B'):

        self.graph, self.init_graph = nx.Graph(graph), nx.Graph(graph).copy()
        self.lsr_max_size = self.graph.number_of_nodes()
        self.treshold = treshold
        self.n_layers = n_layers
        self.max_iter = max_iter
        self.method = method
        self.lsr_logs = logs
        self.lsr_callback = callback

        self.subgraph_list = []
        self.lsr_time, self.lsr_step, self.lsr_step_time = 0, 0, 0
        self.n_nodes = self.graph.number_of_nodes()

        self.m_map = csr_array((self.n_nodes, self.n_nodes))
        self.mi_maps = np.zeros(((self.n_nodes - self.treshold) * self.n_layers + 1,
                                 self.n_nodes, self.n_nodes))
        self.reducted_nodes = []
        self.reducted_edges = []


        self.sol_list, self.cor_list = np.arange(self.n_nodes), np.ones(self.n_nodes)
        self.lsr_sol, self.lsr_sol_dct = '', {}
        self.lsr_cut = 0
        self.step_info = []


    @staticmethod
    def _mutual_information(dm_1, dm_2, dm_12):

        s_1 = - np.trace(dm_1 @ logm(dm_1))
        s_2 = - np.trace(dm_1 @ logm(dm_1))
        s_12 = - np.trace(dm_12 @ logm(dm_12))
        mi = 0.5 * (s_1 + s_2 - s_12)

        return np.real(mi)

    def _graph_reduction(self, edge, corr_coef):

        k, j = edge

        for k_n, k_n_w in self.graph[k].items():

            if k_n == j:
                continue
            if k_n in self.graph.neighbors(j):
                self.graph[j][k_n]['weight'] += corr_coef * k_n_w['weight']
            else:
                self.graph.add_edge(j, k_n, weight=corr_coef * k_n_w['weight'])

        for ind, bit in enumerate(self.sol_list):
            if bit == k:
                self.sol_list[ind] = j
                self.cor_list[ind] *= corr_coef

        self.graph.remove_node(k)

        self.reducted_nodes.append(k)
        self.reducted_edges.append((k, j))
        self.reducted_edges.append((j, k))

    def _mi_map(self, mi_maps, dm_dct):

        mapping = dict(zip(self.graph.nodes(), range(self.graph.number_of_nodes())))


        for idx, layer_mi_map in enumerate(mi_maps):

            for i, k in product(self.graph.nodes(), repeat=2):
                self.mi_maps[self.n_layers * self.lsr_step + idx][i, k] =\
                    layer_mi_map[mapping[i], mapping[k]]

                self.mi_maps[self.n_layers * self.lsr_step + idx][k, i] = \
                    layer_mi_map[mapping[i], mapping[k]]

            for i, k in product(self.reducted_nodes, self.init_graph.nodes()):

                if i == k:
                    self.mi_maps[self.n_layers * self.lsr_step + idx][i, k] = 0

                elif (k != i) & (k in self.reducted_nodes):
                    self.mi_maps[self.n_layers * self.lsr_step + idx][i, k] = \
                        TWO_QUBIT_ENTROPY
                    self.mi_maps[self.n_layers * self.lsr_step + idx][k, i] = \
                        TWO_QUBIT_ENTROPY

                else:
                    self.mi_maps[self.n_layers * self.lsr_step + idx][i, k] = \
                        self._mutual_information(dm_1=ONE_QUBIT_DM,
                                                 dm_2=dm_dct[idx, mapping[k]],
                                                 dm_12=np.kron(ONE_QUBIT_DM, dm_dct[idx, mapping[k]]))

                    self.mi_maps[self.n_layers * self.lsr_step + idx][k, i] = \
                        self._mutual_information(dm_1=ONE_QUBIT_DM,
                                                 dm_2=dm_dct[idx, mapping[k]],
                                                 dm_12=np.kron(ONE_QUBIT_DM, dm_dct[idx, mapping[k]]))

    def _mean_map(self, state_list, probs):

        mapping = dict(zip(self.graph.nodes(), range(self.graph.number_of_nodes())))
        edges = list(self.graph.edges())
        probs = probs.reshape(-1, 1)

        self.m_map[[e[0] for e in edges], [e[1] for e in edges]] = np.sum(
            state_list[:, [mapping[e[0]] for e in edges]] *
            state_list[:, [mapping[e[1]] for e in edges]] *
            probs[:, [0 for _ in edges]], axis=0)

    def _solution_encoder(self, solution_dct):

        self.lsr_sol_dct = {ind:int(solution_dct[bit] * self.cor_list[ind])
                             for ind, bit in enumerate(self.sol_list)}

        self.lsr_sol= ''.join(str(int((1 + solution_dct[bit] * self.cor_list[ind]) / 2))
                                for ind, bit in enumerate(self.sol_list))

    def _cut_value(self):

        edges_weights = nx.get_edge_attributes(self.init_graph, "weight")
        self.lsr_cut = -1 * np.sum([edges_weights[(k, j)] if (self.lsr_sol[k] != self.lsr_sol[j]) else 0
                           for k, j in self.init_graph.edges()])

    def _recursive_step(self):

        self.lsr_step_time = time.time()

        qaoa = MutualInfoQAOA(graph=self.graph,
                              n_layers=self.n_layers,
                              callback=False,
                              logs=False)
        qaoa.solve()
        self._mean_map(qaoa.state_list, qaoa.probs)
        self._collect_step_info(qaoa)
        self._logs()

        ind = csr_array(np.abs(self.m_map)).argmax(explicit=True)
        k_node, j_node = self.m_map.nonzero()[0][ind], self.m_map.nonzero()[1][ind]
        corr_sign = np.sign(self.m_map[k_node, j_node])
        self._graph_reduction((k_node, j_node), corr_coef=corr_sign)
        self._mi_map(qaoa.mi_maps, qaoa.dm_dct)
        self.n_nodes -= 1
        self.lsr_step += 1
        self.step_info = []
        self.m_map = csr_array(self.m_map.shape)

    def solve(self):

        self.lsr_time = time.time()

        while self.treshold < self.graph.number_of_nodes():
            self._recursive_step()

        qaoa = MutualInfoQAOA(graph=self.graph,
                              n_layers=self.n_layers,
                              logs=False,
                              callback=False)
        qaoa.solve()



        self._collect_step_info(qaoa)
        self._logs()

        self.reducted_nodes += list(self.graph.nodes())
        self._mi_map(qaoa.mi_maps, qaoa.dm_dct)
        self._solution_encoder(qaoa.q_solution_dct)
        self._cut_value()
        self.lsr_time = time.time() - self.lsr_time
        self._callback()

    def _callback(self):

        if self.lsr_callback:

            print("\033[33m {}".format(f"==" * 30))
            print(f"Graph       ||   {nx.number_of_nodes(self.init_graph)} nodes   "
                  f"{nx.number_of_edges(self.init_graph)} edges ")
            print(f"QAOA        ||   {self.n_layers} layers   optimization method : {self.method}")
            print(f"LS-RQAOA    ||   {self.lsr_step} recursive steps")
            print(f"Classical   ||   solution : {self.lsr_sol}    "
                  f"Energy : {self.lsr_cut:.4f}")
            print(f"Quantum     ||   solution : {self.lsr_sol}    "
                  f"Energy : {self.lsr_cut:.4f}")
            print(f"Time        ||   QAOA : {np.round(self.lsr_time, 4)} s   "
                  f"Classic : ---- s")
            print("\033[33m {}".format(f"==" * 30))

    def _logs(self):

        if self.lsr_logs:

            print("\033[33m {}".format(f"==" * 30))
            print(f"STEP || {self.lsr_step} || Processing time {(time.time() - self.lsr_step_time):.4f} s")

            for local_task in self.step_info:
                color = "\033[1;2;32m {}" if local_task[5] == 1 else "\033[1;2;31m {}"
                print(color.format(f"--" * 30))
                print(f"Iter {self.lsr_step} || "
                      f"Nodes {local_task[0]} || "
                      f"Time {local_task[4]:.4f} s")
                print(f"Sol {''.join([str(int((bit + 1) / 2)) for bit in local_task[1]])} || "
                      f"QE {-1 * local_task[2]:.4f} || "
                      f"CE {local_task[3]:.4f} || "
                      f"Ratio {-1 * local_task[2] / local_task[3]:.4f}")

    def _collect_step_info(self, solver):

        self.step_info.append([
            solver.n_nodes,
            solver.q_solution,
            solver.q_energy,
            solver.c_energy,
            solver.q_time,
            solver.result])

    def plot_mi_maps(self):

        n_cols = int(np.ceil(np.sqrt(self.mi_maps.shape[0])))
        n_rows = int(np.ceil(self.mi_maps.shape[0] / n_cols))
        fig, axes = plt.subplots(nrows=n_rows,
                                 ncols=n_cols,
                                 figsize=(4 * n_cols, 4 * n_rows),
                                 squeeze=False)

        vmin = 0.
        vmax = np.max(self.mi_maps) * 1.1
        cbar_ax = fig.add_axes((0.92, 0.15, 0.03, 0.7))  # [left, bottom, width, height]

        axes = axes.flatten()
        for i in range(self.mi_maps.shape[0], len(axes)):
            axes[i].set_visible(False)

        for i in range(self.mi_maps.shape[0]):
            if i < self.mi_maps.shape[0] - 1:
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

            axes[i].set_title(f'Step {i + 1}', fontsize=30)
            axes[i].set_xticks([])
            axes[i].set_yticks([])

        cbar = hm.collections[0].colorbar
        cbar.ax.tick_params(labelsize=30)

        plt.subplots_adjust(
            left=0.222,  # Отступ слева (5% от ширины фигуры)
            right=0.892,  # Отступ справа (освобождаем место для colorbar)
            bottom=0.013,  # Отступ снизу
            top=0.936,  # Отступ сверху
            wspace=0.044,  # Горизонтальное расстояние между subplot-ами
            hspace=0.231  # Вертикальное расстояние между subplot-ами
        )

        plt.tight_layout(rect=(0, 0, 0.9, 1))  # Оставляем место справа для цветовой шкалы
        plt.savefig('RQAOA_data/test_7/RQAOA_mims_line.png',
                    dpi=300,  # Качество
                    bbox_inches='tight',  # Обрезает пустые края
                    pad_inches=0.1,  # Минимальный отступ от края
                    facecolor='white',  # Цвет фона
                    edgecolor='none')  # Без рамки
        plt.show()

    def plot_information(self):

        n_nodes = self.init_graph.number_of_nodes()
        means = np.sum(self.mi_maps, axis=(1, 2)) / (n_nodes * (n_nodes - 1))
        maxs = np.max(self.mi_maps, axis=(1, 2))

        mins = [np.min(self.mi_maps[k][self.mi_maps[k]>0]) for k in range(self.mi_maps.shape[0])]
        stds = [np.std(self.mi_maps[k][self.mi_maps[k]>0]) for k in range(self.mi_maps.shape[0])]

        x_points = np.arange(1, self.mi_maps.shape[0] + 1)

        plt.figure(figsize=(8, 6))
        sns.lineplot(x=x_points, y=means, label='Mean', marker='o', linewidth=2)
        sns.lineplot(x=x_points, y=mins, label='Min', marker='s', linestyle='--')
        sns.lineplot(x=x_points, y=maxs, label='Max', marker='^', linestyle='--')

        plt.fill_between(x_points,
                         np.array(means) - np.array(stds),
                         np.array(means) + np.array(stds),
                         alpha=0.2, label='STD')

        plt.xlabel('Steps of RQAOA', fontsize=14)
        plt.ylabel('Mutual information', fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.title(f'Plot of the evolution of mutual information in RQAOA\n'
                  f'Number of nodes: {self.init_graph.number_of_nodes()}   '
                  f'Number of edges: {self.init_graph.number_of_edges()}   '
                  f'Nonlocality: 4.333',
                  fontsize=16)
        plt.legend(
            loc='lower right',
            fontsize=12,  # Размер шрифта
            fancybox=True)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('RQAOA_data/test_7/RQAOA_mi_trend.png',
                    dpi=300,  # Качество
                    bbox_inches='tight',  # Обрезает пустые края
                    pad_inches=0.1,  # Минимальный отступ от края
                    facecolor='white',  # Цвет фона
                    edgecolor='none')  # Без рамки
        plt.show()

    def plot_qubit_information(self):

        x_points = np.arange(1, self.mi_maps.shape[0] + 1)
        plt.figure(figsize=(10, 6))

        for k, j in product(range(self.init_graph.number_of_nodes()), repeat=2):
            sns.lineplot(x=x_points, y=self.mi_maps[: , k, j], linewidth=1)

        plt.xlabel('Layers', fontsize=14)
        plt.ylabel('Information', fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.title(f'Plot of MI tracks for every component of matrix \n'
                  f'Number of nodes: {self.init_graph.number_of_nodes()}   '
                  f'Number of edges: {self.init_graph.number_of_edges()}',
                  fontsize=16)

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_shortest_path(self):

        weights = dict(zip(self.init_graph.edges(), np.ones(self.init_graph.number_of_edges())))
        nx.set_edge_attributes(self.init_graph, weights, "weight")
        disc_dist_matrix = nx.floyd_warshall_numpy(self.init_graph)

        data = pd.DataFrame(columns=['layer', 'value', 'Distance'])
        for layer, mi_map in enumerate(self.mi_maps):
            df = pd.DataFrame({'layer': np.ones(self.init_graph.number_of_nodes() ** 2) * layer,
                               'value': mi_map.flatten(),
                               'Distance': disc_dist_matrix.flatten()})
            data = pd.concat([data, df], ignore_index=True)
        sns.lineplot(data=data[data['Distance'] != 0], x="layer", y="value", hue="Distance")

        plt.xlabel('Layers of QAOA', fontsize=12)
        plt.ylabel('Mean mutual information', fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.title(f'A graph of the evolution of the average mutual information \n'
                  f'for vertices separated by different distances.\n'
                  f'Number of nodes: {self.init_graph.number_of_nodes()}   '
                  f'Number of edges: {self.init_graph.number_of_edges()}   '
                  f'Nonlocality: 4.333',
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
        plt.savefig('RQAOA_data/test_7/RQAOA_dist_info.png',
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
    G = nx.Graph(G)
    weights = dict(zip(G.edges(), np.ones(G.number_of_edges())))
    nx.set_edge_attributes(G, weights, "weight")

    lsq = MI_RQAOA(graph=G,
                   treshold=2,
                   n_layers=1,
                   logs=True,
                   max_iter=2000)
    lsq.solve()
    # lsq.plot_mi_maps()
    # lsq.plot_information()
    # lsq.plot_qubit_information()
    lsq.plot_shortest_path()





