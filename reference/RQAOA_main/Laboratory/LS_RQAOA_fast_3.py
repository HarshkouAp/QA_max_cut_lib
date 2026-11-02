from Tools.GraphDecomposer import *
from Solvers.QAOA import QAOA
from Laboratory.NodeExtender import *
from Generators.GraphGenerator import ConnectedRandGraph
from Solvers.ClassicSolver import ClassicSolver

import networkx as nx
import time


class LargeScaleRQAOA_3:

    def __init__(self, graph, max_size, part_iter, treshold, n_layers, max_iter=300,
                 w_scale=1., n_subgraphs=5,
                 logs=False, callback=False, method='L-BFGS-B', random_seed=None):

        self.graph, self.init_graph = nx.Graph(graph), nx.Graph(graph).copy()
        self.lsr_max_size = max_size
        self.treshold = treshold
        self.n_layers = n_layers
        self.max_iter = max_iter
        self.method = method
        self.part_iter = part_iter
        self.w_scale = w_scale
        self.n_subgraphs = n_subgraphs
        self.lsr_logs = logs
        self.lsr_callback = callback
        self.random_seed = (np.random.randint(1000) if random_seed is None else random_seed)

        self.subgraph_list = []
        self.number_of_solved_problems = 0
        self.lsr_time, self.lsr_step, self.lsr_step_time = 0, 0, 0
        self.lsr_part_time = 0
        self.n_nodes = self.graph.number_of_nodes()
        self.m_map = np.zeros((self.n_nodes, self.n_nodes))
        self.d_map = np.zeros((self.n_nodes, self.n_nodes))
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
        self.merged_nodes.append(j)

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
        self.m_map[k, :] = 0
        self.m_map[:, k] = 0

    def _mean_map(self, graph, state_list, probs):

        mapping = dict(zip(graph.nodes(), range(graph.number_of_nodes())))
        edges = list(graph.edges())
        probs = probs.reshape(-1, 1)

        self.m_map[[e[0] for e in edges], [e[1] for e in edges]] += np.sum(
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

        for part_iter in range(self.part_iter):
            if self.n_nodes > self.lsr_max_size:

                self.lsr_part_time = time.time()

                if self.lsr_step == 0:
                    config = SubgraphConfig(
                        max_size=self.lsr_max_size,
                        num_subgraphs=int(np.round(self.n_nodes / self.lsr_max_size) + 1),
                        allow_overlap=False,
                        random_seed= self.random_seed + (part_iter * 10))
                    generator = OptimizedDenseSubgraphGenerator(self.graph, config)
                    self.subgraph_list = generator.generate_multiple_subgraphs()

                else:
                    self.subgraph_list = generate_all_vertex_subgraphs(G=self.graph,
                                                         vertices=self.merged_nodes,
                                                         max_size=self.lsr_max_size,
                                                         weight_attr='w',
                                                         include_center=True)

                    if len(self.subgraph_list) > self.n_subgraphs:
                        random.shuffle(self.subgraph_list)
                        self.subgraph_list = self.subgraph_list[:self.n_subgraphs]


                self.lsr_part_time = time.time() - self.lsr_part_time

                for subgraph in self.subgraph_list:

                    self.m_map[[edge[0] for edge in subgraph.edges()],
                    [edge[1] for edge in subgraph.edges()]] = 0

                for task_idx, subgraph in enumerate(self.subgraph_list):

                    self.d_map[[edge[0] for edge in subgraph.edges()],
                               [edge[1] for edge in subgraph.edges()]] += 1

                    qaoa = QAOA(graph=subgraph,
                                n_layers=self.n_layers,
                                callback=False,
                                logs=False)
                    qaoa.solve()
                    self.number_of_solved_problems += 1
                    self._mean_map(subgraph, qaoa.state_list, qaoa.probs)
                    self._collect_step_info(qaoa, part_iter, task_idx)

            else:

                self.d_map[[edge[0] for edge in self.graph.edges()],
                           [edge[1] for edge in self.graph.edges()]] += 1

                qaoa = QAOA(graph=self.graph,
                            n_layers=self.n_layers,
                            callback=False,
                            logs=False)
                qaoa.solve()
                self.number_of_solved_problems += 1
                self._mean_map(self.graph, qaoa.state_list, qaoa.probs)
                self._collect_step_info(qaoa, part_iter, task_idx=0)
                break

        self._logs()
        self.merged_nodes = []
        self.m_map[self.d_map != 0] *= np.abs(((self.adj_map[self.d_map != 0] ** self.w_scale)
                                               / self.d_map[self.d_map != 0]))
        max_corr = np.max(np.abs(self.m_map))

        if max_corr < 0.000001:
            edges = list(self.graph.edges())
            k_node, j_node = random.choice(edges)

        else:
            k_nodes, j_nodes = np.where(np.abs(self.m_map) == max_corr)
            k_node, j_node = k_nodes[0], j_nodes[0]

        self.max_corr_dict[max_corr] = (k_node, j_node)
        corr_sign = np.sign(self.m_map[k_node, j_node])
        self._graph_reduction((k_node, j_node), corr_coef=corr_sign)
        self.n_nodes -= 1
        self.lsr_step += 1
        self.step_info = []
        self.d_map = np.zeros_like(self.d_map)
        self.adj_map = self._dict_to_array(nx.get_edge_attributes(self.graph, 'weight'))

    def solve(self):

        self.lsr_time = time.time()

        while self.treshold < self.graph.number_of_nodes():
            self._recursive_step()

        qaoa = QAOA(graph=self.graph,
                    n_layers=self.n_layers,
                    logs=False,
                    callback=False)
        qaoa.solve()
        self.number_of_solved_problems += 1

        self._collect_step_info(qaoa, part_iter=0, task_idx=0)
        self._logs()

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

    def plot_max_corr_hist(self):

        sns.set_palette('deep')
        fig, ax = plt.subplots(figsize=(15, 12))

        max_corr_list = list(self.max_corr_dict.keys())

        sns.histplot(max_corr_list, bins=30, kde=True, ax=ax, stat='count', edgecolor='k', linewidth=0.3)
        ax.set_xlabel('Correlation coefficient')
        ax.set_ylabel('Count')
        ax.set_title(f"Correlation coefficient — distribution")

        mean = np.mean(max_corr_list)
        std = np.std(max_corr_list)
        mn = np.min(max_corr_list)
        mx = np.max(max_corr_list)

        ax.axvline(mean, color='tab:orange', linestyle='-', linewidth=2, label=f"mean = {mean:.3g}")
        if not np.isnan(std) and std > 0:
            ax.axvline(mean - std, color='tab:green', linestyle='--', linewidth=1.5,
                       label=f"mean - std = {mean - std:.3g}")
            ax.axvline(mean + std, color='tab:green', linestyle='--', linewidth=1.5,
                       label=f"mean + std = {mean + std:.3g}")
        ax.axvline(mn, color='tab:red', linestyle=':', linewidth=1.2, label=f"min = {mn:.3g}")
        ax.axvline(mx, color='tab:red', linestyle=':', linewidth=1.2, label=f"max = {mx:.3g}")

        textstr = '\n'.join((
            f"mean: {mean:.4g}",
            f"std:  {std:.4g}",
            f"min:  {mn:.4g}",
            f"max:  {mx:.4g}"
        ))
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        ax.legend(loc='upper left', fontsize=8)
        plt.show()

    def plot_correlations(self):

        max_corr_list = list(self.max_corr_dict.keys())

        plt.plot(range(len(max_corr_list)), max_corr_list, marker='o')
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.ylabel(f"Клики", fontsize=10)
        plt.xlabel('Epsilon', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.title('График кликов наиболее часто встречаемого состояния\n'
                  'Сэмплов: 10000   WS Mod.   Pos/Neg', fontsize=12)
        plt.tight_layout()
        plt.show()




if __name__ == '__main__':

    G = ConnectedRandGraph(number_of_nodes=100,
                           number_of_edges=1500,
                           weights='random',
                           seed=1000)


    lsq = LargeScaleRQAOA_3(graph=G,
                            max_size=10,
                            part_iter=1,
                            treshold=3,
                            n_layers=1,
                            w_scale=1,
                            n_subgraphs=3,
                            logs=True,
                            callback=True,
                            max_iter=500,
                            random_seed=1000)
    lsq.solve()
    # lsq.plot_max_corr_hist()
    # lsq.plot_correlations()





