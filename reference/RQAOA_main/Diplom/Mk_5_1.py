import numpy as np
from Solvers.GoemansWilliamson import GoemansWilliamsonMaxcut
from Tools.GraphDecomposer import *
from Solvers.QAOA import *
from Laboratory.NodeExtender import *
from Solvers.ClassicSolver import ClassicSolver
from Solvers.GoemansWilliamson import *
from Generators.maxcut_benchmark_generator import get_benchmark_graph

import networkx as nx
import time


class LargeScaleRQAOA_mk5:

    def __init__(self, graph, max_size, treshold, n_layers, max_iter=300,
                 logs=False, callback=False, method='ADAM', random_seed=None):

        self.graph, self.init_graph = nx.Graph(graph), nx.Graph(graph).copy()
        self.lsr_max_size = max_size
        self.treshold = treshold
        self.n_layers = n_layers
        self.max_iter = max_iter
        self.method = method
        self.lsr_logs = logs
        self.lsr_callback = callback
        self.random_seed = (np.random.randint(1000) if random_seed is None else random_seed)

        self.lsr_time, self.lsr_step, self.lsr_step_time = 0, 0, 0
        self.lsr_part_time = 0
        self.n_nodes = self.graph.number_of_nodes()
        self.m_map = np.zeros((self.n_nodes, self.n_nodes))
        self.d_map = np.zeros((self.n_nodes, self.n_nodes))
        self.rows, self.cols = np.triu_indices(self.m_map.shape[0], k=1)
        self.merged_nodes = []
        self.max_corr_dict = {}
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
        self.merged_nodes.append(j)

        for k_n, data in list(self.graph[k].items()):
            if k_n == j:
                continue

            new_weight = corr_coef * data["weight"]

            if self.graph.has_edge(j, k_n):
                self.graph[j][k_n]["weight"] += new_weight
            else:
                self.graph.add_edge(j, k_n, weight=new_weight)

            self.adj_map[j, k_n] += new_weight
            self.adj_map[k_n, j] += new_weight

        self.adj_map[k, :] = 0
        self.adj_map[:, k] = 0

        mask = self.sol_list == k
        self.sol_list[mask] = j
        self.cor_list[mask] *= corr_coef

        self.graph.remove_node(k)

        self.m_map[k, :] = 0
        self.m_map[:, k] = 0

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

            if self.lsr_step == 0:
                config = SubgraphConfig(
                    max_size=self.lsr_max_size,
                    num_subgraphs=int(np.round(self.n_nodes / self.lsr_max_size) + 1),
                    allow_overlap=False,
                    random_seed=self.random_seed)
                generator = OptimizedDenseSubgraphGenerator(self.graph, config)
                subgraph_list = generator.generate_multiple_subgraphs()


            else:
                subgraph_list = generate_all_vertex_subgraphs(G=self.graph,
                                                     vertices=self.merged_nodes,
                                                     max_size=self.lsr_max_size,
                                                     weight_attr='w',
                                                     include_center=True)
        else:
            subgraph_list = [self.graph]

        for subgraph in subgraph_list:
            self.m_map[[edge[0] for edge in subgraph.edges()],
            [edge[1] for edge in subgraph.edges()]] = 0

        for task_id, subgraph in enumerate(subgraph_list):
            edges = subgraph.edges()
            self.d_map[[edge[0] for edge in edges], [edge[1] for edge in edges]] += 1

            qaoa = QAOA(graph=subgraph,
                        n_layers=self.n_layers,
                        callback=False,
                        method=self.method,
                        logs=False,
                        samples=1000)
            qaoa.solve()

            corr = np.einsum(
                "s,si,sj->ij",
                qaoa.probs,
                qaoa.state_list,
                qaoa.state_list,
                optimize=True
            )

            node_to_local = {node: i for i, node in enumerate(list(subgraph.nodes()))}

            self.m_map[[e[0] for e in edges], [e[1] for e in edges]] +=\
                corr[[node_to_local[e[0]] for e in edges], [node_to_local[e[1]] for e in edges]]

            self._collect_step_info(qaoa, 0, task_id)


        self._logs()
        self.merged_nodes = []

        self.m_map[self.d_map > 0] *= np.abs(self.adj_map[self.d_map > 0] / self.d_map[self.d_map > 0])

        edges = np.array(list(self.graph.edges()), dtype=int)
        idx = np.argmax(np.abs(self.m_map[edges[:, 0], edges[:, 1]]))
        k_node, j_node = edges[:, 0][idx], edges[:, 1][idx]
        corr_sign = np.sign(self.m_map[k_node, j_node])

        self._graph_reduction((k_node, j_node), corr_coef=corr_sign)
        self._remove_zero_weight_edges()
        self.n_nodes -= 1
        self.lsr_step += 1
        self.step_info = []
        self.d_map = np.zeros_like(self.d_map)

    def solve(self):

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
            print(f"STEP || {self.lsr_step} || Processing time {(time.time() - self.lsr_step_time):.4f} s ||")

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
            solver.ned,
            solver.q_time,
            solver.result])



if __name__ == '__main__':

    #
    # G = DensRandGraph(
    #     number_of_nodes=100,
    #     density=1,
    #     weights='pos_neg',
    #     seed=101
    # )

    G = get_benchmark_graph(25)
    print(G.number_of_edges(), G.number_of_nodes())


    lsq = LargeScaleRQAOA_mk5(
        graph=G,
        max_size=5,
        treshold=20,
        n_layers=1,
        method="L-BFGS-B",
        logs=False,
        callback=True,
        max_iter=100,
        random_seed=1000
    )

    lsq.solve()






