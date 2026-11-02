from Tools.GraphDecomposer import *
from Diplom.QAOA_DF import QAOA
from Generators.GraphGenerator import DensRandGraph
from Generators.maxcut_benchmark_generator import get_benchmark_graph
from Solvers.ClassicSolver import ClassicSolver
import scipy.sparse as sp
import networkx as nx
import time


class LargeScaleRQAOA_mk6v1:

    def __init__(self, graph, max_size, treshold, n_layers, field_amp=1., repetitions=1, max_iter=300,
                 logs=False, callback=False, method='L-BFGS-B', random_seed=None):

        self.graph, self.init_graph = nx.Graph(graph), nx.Graph(graph).copy()
        self.lsr_max_size = max_size
        self.treshold = treshold
        self.field_amp = field_amp
        self.repetitions = repetitions
        self.n_layers = n_layers
        self.max_iter = max_iter
        self.method = method
        self.lsr_logs = logs
        self.lsr_callback = callback
        self.random_seed = (np.random.randint(1000) if random_seed is None else random_seed)

        self.subgraph_list = []
        self.lsr_time, self.lsr_step, self.lsr_step_time = 0, 0, 0
        self.n_nodes = self.graph.number_of_nodes()
        self.adj_map = self._dict_to_array(nx.get_edge_attributes(self.graph, 'weight'))
        self.sol_list, self.cor_list = np.arange(self.n_nodes), np.ones(self.n_nodes)
        self.lsr_sol, self.lsr_sol_dct = '', {}
        self.lsr_cut = 0
        self.step_info = []

    def _remove_zero_weight_edges(self, eps: float = 1e-6):

        edges_to_remove = [
            (u, v)
            for u, v, data in self.graph.edges(data=True)
            if abs(data.get("weight", 0.0)) <= eps
        ]

        self.graph.remove_edges_from(edges_to_remove)

    def _dict_to_array(self, dct):

        n_nodes = self.init_graph.number_of_nodes()
        arr = np.zeros((n_nodes, n_nodes))
        for k, j in dct.keys():
            arr[k, j] = dct[k, j]
        return arr

    def _graph_reduction(self, edge, corr_coef):

        k, j = edge

        if (j in self.graph.nodes) & (k in self.graph.nodes):

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
        merged_nodes = []

        if self.n_nodes > self.lsr_max_size:

            config = SubgraphConfig(
                max_size=self.lsr_max_size,
                num_subgraphs=5 * int(self.n_nodes / self.lsr_max_size),
                allow_overlap=False,
                random_seed=int(self.random_seed)
            )

            generator = OptimizedDenseSubgraphGenerator(self.graph, config)
            subgraphs = generator.generate_multiple_subgraphs()

        else:
            subgraphs = [self.graph]


        for _ in range(self.repetitions):
            merged_nodes = []

            for task_idx, subgraph in enumerate(subgraphs):

                old_nodes = list(subgraph.nodes())
                new_nodes = list(range(len(old_nodes)))
                mapping = dict(zip(new_nodes, old_nodes))

                qaoa = QAOA(graph=subgraph,
                            full_graph=self.graph,
                            field_amp=self.field_amp,
                            n_layers=self.n_layers,
                            callback=False,
                            logs=False,
                            samples=100,
                            method=self.method,)
                qaoa.solve()

                nx.set_node_attributes(self.graph, qaoa.q_solution_dct, "bit")

                corr_map = np.einsum(
                    'i,ij,ik->jk',
                    qaoa.probs,
                    qaoa.state_list,
                    qaoa.state_list,
                    optimize=True
                )

                corr_map = sp.triu(corr_map, k=1, format="coo")
                adj_map = nx.to_scipy_sparse_array(subgraph, format="coo")
                corr_func = corr_map.multiply(adj_map).tocoo()

                idx = np.argmin(corr_func.data)
                k_node = corr_func.row[idx]
                j_node = corr_func.col[idx]

                self._collect_step_info(qaoa, 0, task_idx)

                corr_sign = np.sign(corr_map.tocsr()[k_node, j_node])
                merged_nodes.append((mapping[k_node], mapping[j_node], corr_sign))

        for k, j, s in merged_nodes:
            self._graph_reduction((k, j), corr_coef=s)

        self._remove_zero_weight_edges()
        self._logs()
        self.n_nodes = self.graph.number_of_nodes()
        self.lsr_step += 1
        self.step_info = []

    def solve(self):

        bits = dict(zip(self.graph.nodes(), np.zeros(self.n_nodes)))
        nx.set_node_attributes(self.graph, bits, "bit")

        self.lsr_time = time.time()

        while self.treshold < self.graph.number_of_nodes():
            self._recursive_step()
            if self.graph.number_of_edges() == 0:
                break

        if self.graph.number_of_nodes() < 26:
            CS = ClassicSolver(self.graph)
            CS.solve()
            self._solution_encoder(CS.solution_dct)
        else:
            solution_dct = {node: -1 for node in self.graph.nodes()}
            self._solution_encoder(solution_dct)

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
            print(f"STEP : {self.lsr_step} || Nodes : {self.n_nodes} || Processing time {(time.time() - self.lsr_step_time):.4f} s")

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

    G = DensRandGraph(
        number_of_nodes=100,
        density=1,
        weights='pos_neg',
        seed=1000
    )

    lsq = LargeScaleRQAOA_mk6v1(
        graph=G,
        max_size=10,
        treshold=20,
        field_amp=1,
        repetitions=5,
        n_layers=1,
        logs=True,
        callback=True,
        method='L-BFGS-B',
        max_iter=100,
        random_seed=1000)
    lsq.solve()





