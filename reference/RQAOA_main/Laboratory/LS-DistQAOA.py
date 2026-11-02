import random
import numpy as np
from scipy.optimize import minimize
from Laboratory.NodeExtender import *
from Tools.GraphDecomposer import *
from Solvers.GoemansWilliamson import *


class LocalBlockQAOA:
    def __init__(self, graph, n_layers, max_iter=1, method='L-BFGS-B'):
        self.graph = nx.Graph(graph)
        self.nodes = list(self.graph.nodes())
        self.n_nodes = len(self.nodes)
        self.n_layers = n_layers
        self.max_iter = max_iter
        self.method = method
        self.state_list = 2 * ((np.arange(2 ** (self.n_nodes - 1))[:, None]
                                >> np.arange(self.n_nodes - 1, -1, -1)) & 1) - 1
        self.init_state = np.ones(1 << (self.n_nodes - 1), dtype=complex) / (np.sqrt(2) ** self.n_nodes)
        self.mix_operator = self._build_mix_operator()
        self.angles = np.ones(2 * self.n_layers)
        self.hamiltonian = None
        self.state = None
        self.probs = None
        self.energy = None

    def _build_mix_operator(self):
        state_strings = np.array([('0' * self.n_nodes)[:self.n_nodes - len(bin(s)[2:])] + bin(s)[2:]
                                  for s in range(1 << (self.n_nodes - 1))])
        preset = []
        for qubit in range(1, self.n_nodes + 1):
            perm = []
            for x in state_strings:
                if x[qubit - 1] == '0':
                    y = int(x, 2) + 2 ** (self.n_nodes - qubit)
                    perm.append(2 ** self.n_nodes - y - 1 if y >= 2 ** (self.n_nodes - 1) else y)
                else:
                    y = int(x, 2) - 2 ** (self.n_nodes - qubit)
                    perm.append(2 ** self.n_nodes - y - 1 if y >= 2 ** (self.n_nodes - 1) else y)
            preset.append(perm)
        return np.array(preset)

    def build_hamiltonian(self, field=None):
        adj = nx.to_numpy_array(self.graph, nodelist=self.nodes, weight='weight')
        field = np.zeros(self.n_nodes) if field is None else np.asarray(field, dtype=float)
        batch = self.state_list
        const = -0.25 * np.sum(adj)
        internal = 0.25 * np.einsum('bi,ij,bj->b', batch, adj, batch)
        external = batch @ field
        self.hamiltonian = const + internal + external

    def _objective(self, angles, final=False):
        gamma = angles[:self.n_layers]
        beta = angles[self.n_layers:]
        state = self.init_state.copy()
        for layer in range(self.n_layers):
            vec = np.exp(-1j * gamma[layer] * self.hamiltonian) * state
            for q in range(self.n_nodes):
                vec = vec * np.cos(beta[layer]) - 1j * vec[self.mix_operator[q]] * np.sin(beta[layer])
            state = vec
        energy = np.real(2 * np.sum(np.conj(state) * self.hamiltonian * state))
        return (state, energy) if final else energy

    def solve(self, field=None):
        self.build_hamiltonian(field)
        result = minimize(
            self._objective,
            self.angles,
            method=self.method,
            options={'maxiter': self.max_iter, 'maxfun': self.max_iter, 'gtol': 1e-5},
        )
        self.angles = result.x
        self.state, self.energy = self._objective(self.angles, final=True)
        self.probs = np.abs(self.state) ** 2

    def spin_means(self):
        return 2 * np.sum(self.state_list * self.probs[:, None], axis=0)

    def edge_correlations(self):
        edges = list(self.graph.edges())
        if not edges:
            return {}
        idx = {u: i for i, u in enumerate(self.nodes)}
        ii = [idx[u] for u, _ in edges]
        jj = [idx[v] for _, v in edges]
        corr = 2 * np.sum(self.state_list[:, ii] * self.state_list[:, jj] * self.probs[:, None], axis=0)
        return dict(zip(edges, corr))

    def most_probable_spins(self):
        return dict(zip(self.nodes, self.state_list[np.argmax(self.probs)]))


class DistLSRQAOA:
    def __init__(
        self,
        graph,
        max_size,
        threshold,
        n_layers,
        part_iter=1,
        local_max_iter=1,
        dist_max_iter=20,
        tol=1e-4,
        w_scale=1.0,
        method='L-BFGS-B',
        random_seed=0,
    ):
        self.graph = nx.Graph(graph)
        self.init_graph = self.graph.copy()
        self.max_size = max_size
        self.threshold = threshold
        self.n_layers = n_layers
        self.part_iter = part_iter
        self.local_max_iter = local_max_iter
        self.dist_max_iter = dist_max_iter
        self.tol = tol
        self.w_scale = w_scale
        self.method = method
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        self.n0 = self.init_graph.number_of_nodes()
        self.step = 0
        self.merged_nodes = []
        self.sol_list = np.arange(self.n0)
        self.cor_list = np.ones(self.n0, dtype=int)
        self.m_map = np.zeros((self.n0, self.n0), dtype=float)
        self.d_map = np.zeros((self.n0, self.n0), dtype=float)
        self.spin_mean = {u: float(self.rng.choice([-1, 1])) for u in self.graph.nodes()}
        self.solution_spin = None
        self.solution_bits = None
        self.energy = None

    def _adjacency_map(self):
        arr = np.zeros((self.n0, self.n0), dtype=float)
        for u, v, data in self.graph.edges(data=True):
            w = data.get('weight', 1.0)
            arr[u, v] = w
            arr[v, u] = w
        return arr

    def _subgraphs(self, part_iter):
        if self.graph.number_of_nodes() <= self.max_size:
            return [self.graph.copy()]
        if self.step == 0:
            config = SubgraphConfig(
                max_size=self.max_size,
                num_subgraphs=int(np.round(self.graph.number_of_nodes() / self.max_size) + 1),
                allow_overlap=False,
                random_seed=self.random_seed + 10 * part_iter,
            )
            generator = OptimizedDenseSubgraphGenerator(self.graph, config)
            return generator.generate_multiple_subgraphs()
        return generate_all_vertex_subgraphs(
            G=self.graph,
            vertices=self.merged_nodes,
            max_size=self.max_size,
            weight_attr='w',
            include_center=True,
        )

    def _external_field(self, part, spin_mean):
        part_set = set(part)
        field = np.zeros(len(part), dtype=float)
        for i, u in enumerate(part):
            for v in self.graph.neighbors(u):
                if v not in part_set:
                    field[i] += 0.5 * self.graph[u][v].get('weight', 1.0) * spin_mean[v]
        return field

    def _distributed_pass(self, subgraphs):
        blocks = [
            LocalBlockQAOA(g, n_layers=self.n_layers, max_iter=self.local_max_iter, method=self.method)
            for g in subgraphs if g.number_of_nodes() > 1
        ]
        spin_mean = {u: self.spin_mean.get(u, float(self.rng.choice([-1, 1]))) for u in self.graph.nodes()}
        block_corr = {}

        for _ in range(self.dist_max_iter):
            new_spin = dict(spin_mean)
            block_corr = {}
            for block in blocks:
                field = self._external_field(block.nodes, spin_mean)
                block.solve(field=field)
                for u, m in zip(block.nodes, block.spin_means()):
                    new_spin[u] = m
                for (u, v), c in block.edge_correlations().items():
                    block_corr[(u, v)] = c
                    block_corr[(v, u)] = c
            delta = max(abs(new_spin[u] - spin_mean[u]) for u in spin_mean) if spin_mean else 0.0
            spin_mean = new_spin
            if delta < self.tol:
                break

        self.spin_mean = spin_mean
        return block_corr

    def _graph_reduction(self, edge, corr_sign):
        k, j = edge
        self.merged_nodes = [j]

        for n, data in list(self.graph[k].items()):
            if n == j:
                continue
            w = data.get('weight', 1.0)
            if self.graph.has_edge(j, n):
                self.graph[j][n]['weight'] += corr_sign * w
            else:
                self.graph.add_edge(j, n, weight=corr_sign * w)

        for idx, bit in enumerate(self.sol_list):
            if bit == k:
                self.sol_list[idx] = j
                self.cor_list[idx] *= int(corr_sign)

        self.graph.remove_node(k)
        self.spin_mean.pop(k, None)
        self.m_map[k, :] = 0.0
        self.m_map[:, k] = 0.0
        self.d_map[k, :] = 0.0
        self.d_map[:, k] = 0.0

    def _recursive_step(self):
        prev_merged = list(self.merged_nodes)
        self.m_map[:] = 0.0
        self.d_map[:] = 0.0
        adj_map = self._adjacency_map()

        for part_iter in range(self.part_iter):
            if self.step > 0:
                self.merged_nodes = prev_merged
            subgraphs = self._subgraphs(part_iter)
            corr = self._distributed_pass(subgraphs)
            for u, v in corr:
                self.m_map[u, v] += corr[(u, v)]
                self.d_map[u, v] += 1.0

        mask = self.d_map > 0
        self.m_map[mask] *= np.abs((adj_map[mask] ** self.w_scale) / self.d_map[mask])

        if not np.any(mask):
            u, v = random.choice(list(self.graph.edges()))
            self._graph_reduction((u, v), 1)
            self.step += 1
            return

        idx = np.argmax(np.abs(self.m_map))
        k, j = np.unravel_index(idx, self.m_map.shape)
        if abs(self.m_map[k, j]) < 1e-12:
            k, j = random.choice(list(self.graph.edges()))
            corr_sign = 1
        else:
            corr_sign = int(np.sign(self.m_map[k, j]))
        self._graph_reduction((k, j), corr_sign)
        self.step += 1

    def _solve_small_graph_exactly(self):
        nodes = list(self.graph.nodes())
        n = len(nodes)
        best_energy = np.inf
        best_spins = None

        for mask in range(1 << n):
            spins = np.array([1 if ((mask >> i) & 1) == 0 else -1 for i in range(n)], dtype=int)
            spin_dct = dict(zip(nodes, spins))
            energy = -sum(
                data.get('weight', 1.0)
                for u, v, data in self.graph.edges(data=True)
                if spin_dct[u] != spin_dct[v]
            )
            if energy < best_energy:
                best_energy = energy
                best_spins = spin_dct

        return best_spins, best_energy

    def _decode_solution(self, reduced_spin):
        full_spin = {
            i: int(reduced_spin[self.sol_list[i]] * self.cor_list[i])
            for i in range(self.n0)
        }
        full_bits = {i: int(full_spin[i] < 0) for i in range(self.n0)}
        return full_spin, full_bits

    def cut_value(self):
        if self.solution_bits is None:
            return None
        return sum(
            data.get('weight', 1.0)
            for u, v, data in self.init_graph.edges(data=True)
            if self.solution_bits[u] != self.solution_bits[v]
        )

    def solve(self):
        while self.graph.number_of_nodes() > self.threshold:
            self._recursive_step()
        reduced_spin, self.energy = self._solve_small_graph_exactly()
        self.solution_spin, self.solution_bits = self._decode_solution(reduced_spin)
        return self.solution_bits


if __name__ == '__main__':
    from Generators.GraphGenerator import ConnectedRandGraph

    G = ConnectedRandGraph(number_of_nodes=100,
                           number_of_edges=4950,
                           weights='random',
                           seed=1000)

    solver = DistLSRQAOA(
        graph=G,
        max_size=10,
        threshold=20,
        n_layers=1,
        part_iter=1,
        local_max_iter=1,
        dist_max_iter=25,
        tol=1e-4,
        w_scale=1.0,
        random_seed=42,
    )
    solver.solve()
    print('cut =', solver.cut_value())
    print('bits =', solver.solution_bits)

