import numpy as np
from Tools.Densest_sub import *
from Solvers.QAOA import QAOA
from Generators.GraphGenerator import DensRandGraph
from Solvers.ClassicSolver import ClassicSolver
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import time


class LargeScaleRQAOA_mk1:

    def __init__(self, graph, max_size, treshold, n_layers, max_iter=300,
                 logs=False, callback=False, method='L-BFGS-B', random_seed=None):

        self.graph, self.init_graph = nx.Graph(graph), nx.Graph(graph).copy()
        self.lsr_max_size = max_size
        self.treshold = treshold
        self.n_layers = n_layers
        self.max_iter = max_iter
        self.method = method
        self.lsr_logs = logs
        self.lsr_callback = callback
        self.random_seed = (np.random.randint(1000) if random_seed is None else random_seed)

        self.subgraph_list = []
        self.lsr_time, self.lsr_step, self.lsr_step_time = 0, 0, 0
        self.n_nodes = self.graph.number_of_nodes()
        self.max_corr_dict = {}
        self.adj_map = self._dict_to_array(nx.get_edge_attributes(self.graph, 'weight'))
        self.sol_list, self.cor_list = np.arange(self.n_nodes), np.ones(self.n_nodes)
        self.lsr_sol, self.lsr_sol_dct = '', {}
        self.lsr_cut = 0
        self.step_info = []

    def _dict_to_array(self, dct):

        n_nodes = self.init_graph.number_of_nodes()
        arr = np.zeros((n_nodes, n_nodes))
        for k, j in dct.keys():
            arr[k, j] = dct[k, j]
        return arr

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

        if self.n_nodes > self.lsr_max_size:
            subgraph, nodes, m = densest_k_subgraph_fast(self.graph, k=self.lsr_max_size)
        else:
            subgraph = self.graph

        old_nodes = list(subgraph.nodes())
        new_nodes = list(range(len(old_nodes)))
        mapping = dict(zip(old_nodes, new_nodes))
        subgraph = nx.relabel_nodes(subgraph, mapping)
        mapping = dict(zip(new_nodes, old_nodes))

        qaoa = QAOA(graph=subgraph,
                    n_layers=self.n_layers,
                    callback=False,
                    logs=False,
                    samples=100,
                    method=self.method,)
        qaoa.solve()

        corr_map = np.einsum(
            'i,ij,ik->jk',
            qaoa.probs,
            qaoa.state_list,
            qaoa.state_list,
            optimize=True
        )

        self._collect_step_info(qaoa, 0, 0)
        self._logs()

        corr_map = np.triu(corr_map)
        np.fill_diagonal(corr_map, 0)
        max_corr = np.max(np.abs(corr_map))
        k_node, j_node = np.where(np.abs(corr_map) == max_corr)
        corr_sign = np.sign(corr_map[k_node[0], j_node[0]])
        k_node, j_node = mapping[k_node[0]], mapping[j_node[0]]

        self.max_corr_dict[max_corr] = (k_node, j_node)
        self._graph_reduction((k_node, j_node), corr_coef=corr_sign)
        self.n_nodes -= 1
        self.lsr_step += 1
        self.step_info = []
        self.adj_map = self._dict_to_array(nx.get_edge_attributes(self.graph, 'weight'))

    def solve(self):

        self.lsr_time = time.time()

        while self.treshold < self.graph.number_of_nodes():
            self._recursive_step()

        CS = ClassicSolver(self.graph)
        CS.solve()

        self._solution_encoder(CS.solution_dct)
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
                color = "\033[1;2;32m {}" if local_task[7] == 1 else "\033[1;2;31m {}"
                print(color.format(f"--" * 30))
                print(f"Iter {local_task[0]} || "
                      f"Idx {local_task[1]} || "
                      f"Nodes {local_task[2]} || "
                      f"Time {local_task[6]:.4f} s")
                print(f"Sol {''.join([str(int((bit + 1) / 2)) for bit in local_task[3]])} || "
                      f"QE {-1 * local_task[4]:.4f} || "
                      f"CE {local_task[5]:.4f} || "
                      f"Ratio {-1 * local_task[4] / local_task[5]:.4f}")

    def _collect_step_info(self, solver, part_iter, task_idx):

        self.step_info.append([
            part_iter,
            task_idx,
            solver.n_nodes,
            solver.q_solution,
            solver.q_energy,
            solver.c_energy,
            solver.q_time,
            solver.result])






if __name__ == '__main__':

    G = DensRandGraph(number_of_nodes=100,
                      density=1,
                      weights='pos_neg',
                      seed=1000)

    lsq = LargeScaleRQAOA_mk1(graph=G,
                          max_size=10,
                          treshold=20,
                          n_layers=1,
                          logs=True,
                          callback=True,
                          method='ADAM',
                          max_iter=30,
                          random_seed=1000)
    lsq.solve()





