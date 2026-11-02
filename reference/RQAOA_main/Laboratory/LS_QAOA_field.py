import numpy as np

from Tools.GraphDecomposer import *
from Laboratory.QAOA_F_M import *
from Laboratory.NodeExtender import *
from Solvers.ClassicSolver import ClassicSolver
from Solvers.GoemansWilliamson import *
import networkx as nx
import time


class LargeScaleRQAOAF:

    def __init__(self, graph, max_size, part_iter, treshold, n_layers, max_iter=300,
                 w_scale=1., alpha=1., beta=1., gamma=1.,
                 logs=False, callback=False, method='L-BFGS-B', random_seed=None):

        self.graph, self.init_graph = nx.Graph(graph), nx.Graph(graph).copy()
        self.lsr_max_size = max_size
        self.treshold = treshold
        self.n_layers = n_layers
        self.max_iter = max_iter
        self.method = method
        self.part_iter = part_iter
        self.w_scale = w_scale
        self.lsr_logs = logs
        self.lsr_callback = callback
        self.random_seed = (np.random.randint(1000) if random_seed is None else random_seed)

        self.subgraph_list = []
        self.lsr_time, self.lsr_step, self.lsr_step_time = 0, 0, 0
        self.lsr_part_time = 0
        self.n_nodes = self.graph.number_of_nodes()
        self.max_exp_arr = []
        self.merged_nodes = []
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
                nx.set_edge_attributes(self.graph, {(j, k_n): 1.}, "corr")

        for ind, bit in enumerate(self.sol_list):
            if bit == k:
                self.sol_list[ind] = j
                self.cor_list[ind] *= corr_coef

        self.graph.remove_node(k)

    def _mean_map(self, graph, state_list, probs):

        m_map = np.zeros((self.lsr_max_size, self.lsr_max_size))
        mapping = dict(zip(graph.nodes(), range(graph.number_of_nodes())))
        mapping_inv = dict(zip(range(graph.number_of_nodes()), graph.nodes()))
        edges = list(graph.edges())
        probs = probs.reshape(-1, 1)

        m_map[[mapping[e[0]] for e in edges], [mapping[e[1]] for e in edges]] = np.sum(
            state_list[:, [mapping[e[0]] for e in edges]] *
            state_list[:, [mapping[e[1]] for e in edges]] *
            probs[:, [0 for _ in edges]], axis=0)

        adj = nx.adjacency_matrix(graph).toarray()
        m_map *= np.abs(adj)
        max_corr = np.max(np.abs(m_map))

        if max_corr < 0.001:
            k_node, j_node = random.choice(edges)
        else:
            k_nodes, j_nodes = np.where(np.abs(m_map) == max_corr)
            k_node, j_node = k_nodes[0], j_nodes[0]

        corr_sign = np.sign(m_map[k_node, j_node])

        return mapping_inv[k_node], mapping_inv[j_node], corr_sign

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

        for part_iter in range(self.part_iter):

            self.lsr_part_time = time.time()
            config = SubgraphConfig(
                max_size=self.lsr_max_size,
                num_subgraphs=int(np.round(self.n_nodes / self.lsr_max_size) + 1),
                allow_overlap=False,
                random_seed= self.random_seed + (part_iter * 10))
            generator = OptimizedDenseSubgraphGenerator(self.graph, config)
            self.subgraph_list = generator.generate_multiple_subgraphs()
            self.lsr_part_time = time.time() - self.lsr_part_time

            for task_idx, subgraph in enumerate(self.subgraph_list):

                qaoa = QAOA(graph=subgraph,
                            full_graph=self.graph,
                            n_layers=self.n_layers,
                            method='ADAM',
                            callback=False,
                            logs=False,
                            samples=1000)
                qaoa.solve()

                nx.set_node_attributes(self.graph, qaoa.q_bits_probs, "bit")
                k_node, j_node, corr_sign = self._mean_map(subgraph, qaoa.state_list, qaoa.probs)
                self.max_exp_arr.append((k_node, j_node, corr_sign))
                self._collect_step_info(qaoa, part_iter, task_idx)

        print('==' * 30)
        for k, j, s in self.max_exp_arr:

            print(k, j, s)
            if (k not in self.graph.nodes()) or (j not in self.graph.nodes()):
                continue
            self._graph_reduction((k, j), corr_coef=s)

        self.n_nodes -= len(self.max_exp_arr)
        self.lsr_step += 1
        self.step_info = []
        self.max_exp_arr = []
        self.adj_map = self._dict_to_array(nx.get_edge_attributes(self.graph, 'weight'))

    def solve(self):

        bits = dict(zip(self.graph.nodes(), np.zeros(int(self.graph.number_of_nodes()))))
        nx.set_node_attributes(self.graph, bits, "bit")

        corrs = dict(zip(self.graph.edges(), np.ones(int(self.graph.number_of_edges()))))
        nx.set_edge_attributes(self.graph, corrs, "corr")

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
            print(f"LS-RQAOA    ||   {self.lsr_step} recursive steps   {self.part_iter} partitions")
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
            print(f"STEP || {self.lsr_step} || Processing time {(time.time() - self.lsr_step_time):.4f} s ||"
                  f" Partition time {self.lsr_part_time:.4f} s ||")

            for local_task in self.step_info:
                color = "\033[1;2;32m {}" if local_task[8] == 1 else "\033[1;2;31m {}"
                print(color.format(f"--" * 30))
                print(f"Iter {local_task[0]} || "
                      f"Idx {local_task[1]} || "
                      f"Nodes {local_task[2]} || "
                      f"Time {local_task[7]:.4f} s")
                print(f"Sol {''.join([str(int((bit + 1) / 2)) for bit in local_task[3]])} || "
                      f"QE {-1 * local_task[4]:.4f} || "
                      f"CE {local_task[5]:.4f} || "
                      f"NED {local_task[6]:.4f}")

    def _collect_step_info(self, solver, part_iter, task_idx):

        self.step_info.append([
            part_iter,
            task_idx,
            solver.n_nodes,
            solver.q_solution,
            solver.q_energy,
            solver.c_energy,
            solver.state_ned,
            solver.q_time,
            solver.result])




if __name__ == '__main__':

    G = ConnectedRandGraph(number_of_nodes=100,
                           number_of_edges=4950,
                           weights='random',
                           seed=1000)

    lsq = LargeScaleRQAOAF(graph=G,
                          max_size=10,
                          part_iter=1,
                          treshold=20,
                          n_layers=1,
                          logs=False,
                          callback=True,
                          max_iter=500,
                          random_seed=1000)
    lsq.solve()







