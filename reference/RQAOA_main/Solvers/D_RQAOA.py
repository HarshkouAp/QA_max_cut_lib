import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import warnings
import time
plt.rcParams.update({"font.family": "serif"})
warnings.filterwarnings('ignore')

from D_QAOA import D_QAOA
from Generators.GraphGenerator import DensRandGraph



class RecursiveQAOA(D_QAOA):

    def __init__(self, graph, treshold, n_layers, max_iter=300, logs=False,
                 callback=True, draw_process=False, method='L-BFGS-B'):

        self.treshold = treshold
        self.draw_process = draw_process
        self.rq_callback = callback
        self.rq_logs = logs

        D_QAOA.__init__(self,
                      graph=graph,
                      n_layers=n_layers,
                      max_iter=max_iter,
                      logs=logs,
                      frack=(np.pi / 8),
                      callback=False,
                      method=method)

        self.step, self.rq_time = 0, 0
        self.steps_borders, steps_energies = np.array([]), np.array([])
        self.merged_nodes = ([], [])
        self.m_map = np.zeros((self.n_nodes, self.n_nodes))

    def _draw_mean_map(self):

        plt.figure(figsize=(8, 8))
        sns.heatmap(self.m_map + self.m_map.T,
                    fmt=".3f",
                    cmap="crest",
                    cbar=True,
                    square=True,
                    annot=True)
        plt.tight_layout()
        plt.show()

    def _mean_map(self):

        self.m_map = np.zeros((self.graph.number_of_nodes(), self.graph.number_of_nodes()))
        edges = list(self.graph.edges())
        probs = self.d_probs.reshape(-1, 1)

        self.m_map[[e[0] for e in edges], [e[1] for e in edges]] = np.sum(
            self.state_list[:, [e[0] for e in edges]] *
            self.state_list[:, [e[1] for e in edges]] *
            probs[:, [0 for _ in edges]], axis=0)

    def _recursive_step(self):

        start_step = time.time()
        D_QAOA.solve(self)

        self._mean_map()
        self.m_map[self.merged_nodes] = 0
        k_node_list, j_node_list = np.where(np.abs(self.m_map) == np.abs(self.m_map).max())
        k_node, j_node = k_node_list[0], j_node_list[0]
        self.merged_nodes[0].append(k_node)
        self.merged_nodes[1].append(j_node)

        self.step += 1
        self.steps_borders = np.append(self.steps_borders, self.iterations)

        if self.draw_process:
            self._draw_mean_map()

        idx = np.where((self.state_list[:, k_node] *
                        self.state_list[:, j_node] *
                        np.sign(self.m_map[k_node, j_node]) == 1))

        self.state_list = self.state_list[idx]
        self.n_nodes = int(np.log2(len(self.state_list))) + 1
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
                 label='Convergence line')

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
                   label='Exact energy')

        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        max_y = 0.9 * max(self.energies)
        min_y = 1.1 * min(self.energies)
        ax.set_ylim(min_y, max_y)

        ax.axhline(y=0,
                   color='red',
                   linestyle='--',
                   alpha=0.3,
                   linewidth=0.7,
                   label='Parameter shift')

        ax.axhline(y=0,
                   color='red',
                   linestyle='-',
                   alpha=0.7,
                   linewidth=0.9,
                   label='Recursion step')

        plt.title('Energy convergence graph for parameter optimization for RQAOA '
                  'using layerwise-QAOA \n at each individual recursive step'
                  'for a toric lattice graph with 16 vertices\n'
                  f'The process used QAOA with {self.n_layers} layers and '
                  f'{self.graph.number_of_nodes() - self.treshold} reduction steps.',
                  fontsize=16, pad=20)
        plt.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlabel('Optimization iterations', fontsize=16, labelpad=10)
        ax.set_ylabel('Energy', fontsize=16, labelpad=10)
        ax.legend(loc='upper right', fontsize=12)
        sns.despine()
        plt.show()


if __name__ == '__main__':

    G = DensRandGraph(number_of_nodes=12,
                      density=0.4,
                      weights="random",
                      seed=42)

    rq = RecursiveQAOA(graph=G,
                       treshold=1,
                       n_layers=1,
                       logs=True)
    rq.solve()
    rq.draw_convegence()







