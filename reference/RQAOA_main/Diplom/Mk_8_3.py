import numpy as np
from Diplom.QAOA2_CE_fixed import QAOAinQAOA
from Solvers.ClassicSolver import ClassicSolver
from Generators.GraphGenerator import DensRandGraph
import networkx as nx
import time




class LargeScaleRQAOA_mk8v1:

    def __init__(self, graph, max_size, treshold, n_layers,
                 repetitions=1, init_amp=1., max_iter=300,
                 logs=False, callback=False, method='ADAM', random_seed=None):

        self.graph, self.init_graph = nx.Graph(graph), nx.Graph(graph).copy()
        self.lsr_max_size = max_size
        self.treshold = treshold
        self.n_layers = n_layers
        self.repetitions = repetitions
        self.init_amp = init_amp
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
        self.max_corr_dict = {}
        self.adj_map = self._dict_to_2d_array(nx.get_edge_attributes(self.graph, 'weight'))
        self.sol_list, self.cor_list = np.arange(self.n_nodes), np.ones(self.n_nodes)
        self.lsr_sol, self.lsr_sol_dct = '', {}
        self.lsr_cut = 0
        self.pre_cut = 0
        self.bit_arr = np.zeros(self.n_nodes)
        self.cut_arr = np.array([])
        self.step_info = []


    def _remove_zero_weight_edges(self, eps: float = 1e-6):

        edges_to_remove = [
            (u, v)
            for u, v, data in self.graph.edges(data=True)
            if abs(data.get("weight", 0.0)) <= eps
        ]

        self.graph.remove_edges_from(edges_to_remove)

    def _dict_to_2d_array(self, dct):

        n_nodes = self.init_graph.number_of_nodes()
        arr = np.zeros((n_nodes, n_nodes))
        for k, j in dct.keys():
            arr[k, j] = dct[k, j]
        return arr

    def _dict_to_1d_array(self, dct):

        n_nodes = self.init_graph.number_of_nodes()
        arr = np.zeros(n_nodes)
        for k in dct.keys():
            arr[k] = dct[k]
        return arr

    def _graph_reduction(self, edge, corr_coef):

        k, j = edge

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

    def _energy_prediction_disc(self, corr_map):

        corr_map += corr_map.T
        prediction_map = np.zeros_like(corr_map)

        for edge in self.graph.edges():

            i, k = edge

            temp = corr_map
            temp[:, i] = corr_map[:, k] * corr_map[i, k]
            temp[i, :] = corr_map[k, :] * corr_map[i, k]
            temp[i, k] = corr_map[i, k]
            temp[k, i] = corr_map[k, i]

            prediction_map[i, k] = np.sum(temp * self.adj_map)

            k, i = edge

            temp = corr_map
            temp[:, i] = corr_map[:, k] * corr_map[i, k]
            temp[i, :] = corr_map[k, :] * corr_map[i, k]
            temp[i, k] = corr_map[i, k]
            temp[k, i] = corr_map[k, i]

            prediction_map[i, k] = np.sum(temp * self.adj_map)

        return prediction_map

    def _recursive_step(self):

        self.lsr_step_time = time.time()

        qaoa2 = QAOAinQAOA(
            max_size=self.lsr_max_size,
            depth=self.n_layers,
            repetition=self.repetitions,
            max_iter=self.max_iter,
            init_amp=self.init_amp,
            callback=False,
        )
        qaoa2.solve(graph=nx.Graph(self.graph))

        self._solution_encoder(qaoa2.solution_dct)
        self._cut_value()

        coef = 0 if len(self.cut_arr) == 0 else self.cut_arr.mean()

        self.m_map = ((self.m_map * coef +
                       self._dict_to_2d_array(qaoa2.corr_dct) * qaoa2.solution_cut) /
                      (coef + qaoa2.solution_cut))

        self.bit_arr = ((self.bit_arr * coef +
                         self._dict_to_1d_array(qaoa2.solution_dct) * qaoa2.solution_cut) /
                        (coef + qaoa2.solution_cut))

        self.pre_cut = qaoa2.solution_cut
        self.cut_arr = np.append(self.cut_arr, qaoa2.solution_cut)

        nx.set_edge_attributes(
            self.graph,
            {(k, j): {"corr": self.m_map[k, j]} for k, j in self.graph.edges()}
        )

        nx.set_node_attributes(
            self.graph,
            {k : {"spin": self.bit_arr[k]} for k in self.graph.nodes()}
        )

        corr_func = self._energy_prediction_disc(self.m_map)
        print(corr_func.min(), corr_func.max())

        edges = np.array(list(self.graph.edges()), dtype=int)
        idx = np.argmin(corr_func[edges[:, 0], edges[:, 1]])
        k_node, j_node = edges[:, 0][idx], edges[:, 1][idx]
        corr_sign = np.sign(self.m_map[k_node, j_node])

        if self.lsr_logs:
            print(30 * "===")
            print(f"Nodes: {self.n_nodes}   Cut: {qaoa2.solution_cut:.3f}   "
                  f"Edge: ({k_node}, {j_node})   Corr: {self.m_map[k_node, j_node]:.3f}   "
                  f"Full Cut: {self.lsr_cut:.3f}")

        self._graph_reduction((k_node, j_node), corr_coef=corr_sign)
        self._remove_zero_weight_edges()
        self.n_nodes -= 1
        self.lsr_step += 1
        self.step_info = []

    def solve(self):

        self.lsr_time = time.time()
        nx.set_edge_attributes(
            self.graph,
            {edge: {"corr": 0} for edge in self.graph.edges()}
        )

        values = self.init_amp * np.random.uniform(
            -1,
            1,
            size=self.graph.number_of_nodes(),
        )
        spins = dict(zip(self.graph.nodes(), np.asarray(values)))
        nx.set_node_attributes(self.graph, spins, "spin")

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


if __name__ == '__main__':


    G = DensRandGraph(
        number_of_nodes=100,
        density=1,
        weights='pos_neg',
        seed=1000
    )

    lsq = LargeScaleRQAOA_mk8v1(
        graph=G,
        max_size=4,
        treshold=20,
        n_layers=1,
        repetitions=1,
        init_amp=0.1,
        method="L-BFGS-B",
        logs=True,
        callback=True,
        max_iter=300,
        random_seed=1000
    )

    lsq.solve()






