import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import warnings
from collections import deque
import time
plt.rcParams.update({"font.family": "serif"})
warnings.filterwarnings('ignore')

from QAOA import QAOA
from Generators.GraphGenerator import ConnectedRandGraph, DensRandGraph, GridGraph


class RecursiveQAOA(QAOA):

    def __init__(self, graph, treshold, n_layers, max_iter=300, logs=False,
                 callback=False, draw_process=False, method='L-BFGS-B'):

        self.treshold = treshold
        self.draw_process = draw_process
        self.rq_callback = callback
        self.rq_logs = logs

        QAOA.__init__(self,
                      graph=graph,
                      n_layers=n_layers,
                      max_iter=max_iter,
                      logs=logs,
                      callback=False,
                      method=method)

        self.step, self.rq_time = 0, 0
        self.steps_borders = np.array([])
        self.steps_energies = np.array([])
        self.steps_neds = np.array([])
        self.steps_angs = []
        self.fixed_corrs = np.empty((0, 3), dtype=int)

    def _expand_correlations(self, node_k, node_j, corr):

        edges = [tuple(row) for row in self.fixed_corrs]
        edges.append((node_k, node_j, corr))

        graph = {}

        for u, v, c in edges:
            graph.setdefault(u, []).append((v, c))
            graph.setdefault(v, []).append((u, c))

        result = []

        for start in graph:

            queue = deque([(start, 1)])
            visited = {start: 1}

            while queue:
                node, sign = queue.popleft()
                for nxt, edge_sign in graph[node]:
                    new_sign = sign * edge_sign

                    if nxt in visited:
                        continue

                    visited[nxt] = new_sign
                    queue.append((nxt, new_sign))

            for end, s in visited.items():

                if start < end:
                    result.append((start, end, s))

        self.fixed_corrs = np.array(result).astype(int)

    def _recursive_step(self):

        start_step = time.time()
        QAOA.solve(self)

        self.m_map  = np.einsum('i,ij,ik->jk',
                                self.probs,
                                self.state_list,
                                self.state_list,
                                optimize=True)

        self.m_map[(self.fixed_corrs[:, 0], self.fixed_corrs[:, 1])] = 0
        self.m_map = np.triu(self.m_map)
        np.fill_diagonal(self.m_map, 0)

        corr = np.abs(self.m_map).max()
        k_node_list, j_node_list = np.where(np.abs(self.m_map) == corr)
        k_node, j_node = k_node_list[0], j_node_list[0]
        self._expand_correlations(k_node, j_node, np.sign(self.m_map[k_node, j_node]))

        idx = np.where((
                self.state_list[:, k_node] *
                self.state_list[:, j_node] *
                np.sign(self.m_map[k_node, j_node]) == 1
        ))

        self.step += 1
        self.steps_borders = np.append(self.steps_borders, self.iterations)
        self.steps_energies = np.append(self.steps_energies, self.energies[-1])
        self.steps_neds = np.append(self.steps_neds, self.ned)
        self.steps_angs.append(self.optimal_angles)
        self.state_list = self.state_list[idx]
        self.probs = np.abs(self.state) ** 2
        self.n_nodes -= 1
        self.init_state = (np.ones(1 << (self.n_nodes - 1)) / (np.sqrt(2) ** self.n_nodes))

        if self.rq_logs:
            print(f'\n------------(Recursive step : {self.step})------------')
            print(f'Energy : {np.round(self.q_energy, 6)}  '
                  f' Ratio : {-1 * np.round(self.q_energy / self.c_energy, 6)}')
            print(f'Solution : {''.join([str(int((bit + 1) / 2)) for bit in self.q_solution])}   '
                  f'Processing time : {np.round(time.time() - start_step, 6)}')
            print(f'--------------------------------------------\n')

    def solve(self):

        self.rq_time = time.time()

        while self.treshold < self.n_nodes:
            self._recursive_step()

        self.rq_time = time.time() - self.rq_time
        self.steps_angs = np.array(self.steps_angs)
        self._callback()

    def _callback(self):
        if self.rq_callback:

            print("\033[37m {}".format(f"-------------------------------------"))

            if self.result == 1:
                print("\033[32m {}".format("***SUCCESS***"))
            else:
                print("\033[31m {}".format("***FAIL***"))

            print(f"Graph       ||   number of nodes : {self.n_nodes}  "
                  f"number of edges : {nx.number_of_edges(self.graph)} ")
            print(f"QAOA        ||   number of layers : {self.n_layers}   "
                  f"optimization method : {self.method}")
            print(f"RQAOA       ||   number of recursive steps : {self.step}")
            print(f"Classical   ||   solution : {''.join([str(int((bit + 1) / 2)) for bit in self.c_solution])}    "
                  f"Energy : {np.round(self.c_energy, 4)}")
            print(f"Quantum     ||   solution : {''.join([str(int((bit + 1) / 2)) for bit in self.q_solution])}    "
                  f"Energy : {np.round(-1 * self.q_energy, 4)}")
            print(f"Processing  ||   QAOA time : {np.round(self.rq_time, 4)} s   "
                  f"Classic time : {np.round(self.c_time, 4)} s")
            print(f'Total       ||   Ratio : {np.round(-1 * self.q_energy / self.c_energy, 4)}')

    def draw_convegence(self):

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(list(range(self.iterations)),
                 self.energies,
                 linewidth=0.7,
                 label='Линия сходимости')

        for coord in self.layers_borders:
            ax.axvline(x=coord,
                       color='red',
                       linestyle='--',
                       alpha=0.3,
                       linewidth=0.7)

        for coord in self.steps_borders:
            ax.axvline(x=coord,
                       color='red',
                       linestyle='-',
                       alpha=0.7,
                       linewidth=0.9)

        ax.axhline(y=self.c_energy,
                   color='green',
                   linestyle='-',
                   alpha=0.7,
                   linewidth=0.9,
                   label='Точное решение')

        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        max_y = 0.9 * max(self.energies)
        min_y = 1.1 * min(self.energies)
        ax.set_ylim(min_y, max_y)

        ax.axhline(y=0,
                   color='red',
                   linestyle='--',
                   alpha=0.3,
                   linewidth=0.7,
                   label='Шаг Layerwise')

        ax.axhline(y=0,
                   color='red',
                   linestyle='-',
                   alpha=0.7,
                   linewidth=0.9,
                   label='Фикс. корреляции')

        plt.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlabel('Оптимизационные итерации', fontsize=16, labelpad=10)
        ax.set_ylabel('Энергия', fontsize=16, labelpad=10)
        ax.legend(loc='upper right', fontsize=12)
        sns.despine()

        plt.savefig('RQAOA_Con_0',
                    dpi=300,  # Качество
                    bbox_inches='tight',  # Обрезает пустые края
                    pad_inches=0.1,  # Минимальный отступ от края
                    facecolor='white',  # Цвет фона
                    edgecolor='none')  # Без рамки
        plt.show()

    def draw_param_evolution(
            self,
            cmap="Blues",
            min_size=40,
            max_size=300,
            arrow_color="gray",
            arrow_alpha=0.6,
            figsize=(7, 6),
            title=None
    ):

        points = np.asarray(self.steps_angs, dtype=float)
        quality = np.asarray(self.steps_neds, dtype=float)

        x = points[:, 0]
        y = points[:, 1]

        point_ids = np.arange(1, len(points) + 1)

        sizes = min_size + quality * (max_size - min_size)

        fig, ax = plt.subplots(figsize=figsize)

        for i in range(len(points) - 1):
            ax.annotate(
                "",
                xy=(x[i + 1], y[i + 1]),
                xytext=(x[i], y[i]),
                arrowprops=dict(
                    arrowstyle="->",
                    color=arrow_color,
                    alpha=arrow_alpha,
                    lw=1.5,
                    shrinkA=5,
                    shrinkB=5,
                ),
            )

        scatter = ax.scatter(
            x,
            y,
            c=point_ids,
            s=sizes,
            cmap=cmap,
            vmin=1,
            vmax=len(points),
            edgecolors="black",
            linewidths=0.7,
            zorder=3,
        )

        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Номер шага", fontsize=20,)

        xy_min = min(x.min(), y.min())
        xy_max = max(x.max(), y.max())

        margin = 0.05 * (xy_max - xy_min)
        if margin == 0:
            margin = 1.0

        ax.set_xlim(xy_min - margin, xy_max + margin)
        ax.set_ylim(xy_min - margin, xy_max + margin)
        ax.set_aspect("equal", adjustable="box")

        if title is not None:
            ax.set_title(title)

        ax.grid(True, alpha=0.3)

        plt.tick_params(axis='both', which='major', labelsize=18)
        ax.set_xlabel(r"$\gamma$", fontsize=20, labelpad=10)
        ax.set_ylabel(r"$\beta$", fontsize=20, labelpad=10)
        sns.despine()
        plt.tight_layout()

        plt.savefig('RQAOA_Evo_0',
                    dpi=300,  # Качество
                    bbox_inches='tight',  # Обрезает пустые края
                    pad_inches=0.1,  # Минимальный отступ от края
                    facecolor='white',  # Цвет фона
                    edgecolor='none')  # Без рамки
        plt.show()



if __name__ == '__main__':

    G = DensRandGraph(
        number_of_nodes=20,
        density=0.7,
        weights="ones",
        seed=1000,
    )

    rq = RecursiveQAOA(graph=G,
                       treshold=3,
                       n_layers=1,
                       logs=True,
                       callback=True,
                       max_iter=300,)
    rq.solve()









