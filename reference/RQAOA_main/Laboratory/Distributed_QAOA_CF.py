import time
import warnings

import networkx as nx
import numpy as np
from scipy.optimize import minimize
import numpy as np
from ENV.ProblemGenerator import WishartMaxCutGenerator
from Laboratory.QAOA_F_M import QAOA
from Generators.GraphGenerator import DensRandGraph
from Tools.GraphDecomposer import *
from Solvers.GoemansWilliamson import *


warnings.filterwarnings("ignore")


class QAOA_CF:

    def __init__(self, graph, subgraphs, n_layers, field_amp=1.0, spin_amp=0.1,
                 spin_mix=0.5, theta_mix=0.5, angle_step=0.05,
                 max_iter=100, local_iter=1, method="L-BFGS-B",
                 logs=False, callback=False):

        self.full_graph = nx.Graph(graph)
        self.subgraphs = self._prepare_subgraphs(subgraphs)

        self.n_layers = n_layers
        self.field_amp = field_amp
        self.spin_amp = spin_amp
        self.spin_mix = spin_mix
        self.theta_mix = theta_mix
        self.angle_step = angle_step
        self.max_iter = max_iter
        self.local_iter = local_iter
        self.method = method
        self.logs = logs
        self.callback = callback

        self.time = 0.0
        self.iterations = 0

        self.energies = []
        self.history = []

        self.theta_dct = {}
        self.local_data = {}
        self.local_energy_dct = {}
        self.local_spins_dct = {}

        self.solution = {}
        self.cut_value = 0.0
        self.energy = 0.0

        self._init_spins()
        self._init_local_data()

    def _prepare_subgraphs(self, subgraphs):

        result = []

        for subgraph in subgraphs:
            if isinstance(subgraph, nx.Graph):
                nodes = list(subgraph.nodes())
            else:
                nodes = list(subgraph)

            result.append(nx.Graph(self.full_graph.subgraph(nodes)))

        return result

    def _to_spin_bit(self, bit):

        bit = float(bit)

        if np.isclose(bit, 0.0):
            return -1.0

        if np.isclose(bit, 1.0):
            return 1.0

        if np.isclose(bit, -1.0):
            return -1.0

        return float(np.sign(bit))

    def _init_spins(self):

        for node in self.full_graph.nodes():

            if "bit" in self.full_graph.nodes[node]:
                bit = self.full_graph.nodes[node]["bit"]
            elif "spin" in self.full_graph.nodes[node]:
                bit = self.full_graph.nodes[node]["spin"]
            else:
                bit = 1.0

            self.full_graph.nodes[node]["spin"] = self.spin_amp * self._to_spin_bit(bit)

    def _init_local_data(self):

        for subgraph_id, graph in enumerate(self.subgraphs):

            nodes = list(graph.nodes())
            n_nodes = len(nodes)

            states = np.arange(1 << n_nodes)

            state_list = 2 * (
                (states[:, None] >> np.arange(n_nodes - 1, -1, -1)) & 1
            ) - 1

            init_state = np.ones(1 << n_nodes, dtype=np.complex128) / np.sqrt(1 << n_nodes)

            mix_operator = np.array([
                states ^ (1 << (n_nodes - qubit - 1))
                for qubit in range(n_nodes)
            ])

            self.theta_dct[subgraph_id] = np.ones(2 * self.n_layers, dtype=np.float64)

            self.local_data[subgraph_id] = {
                "nodes": nodes,
                "n_nodes": n_nodes,
                "state_list": state_list.astype(np.float64),
                "init_state": init_state,
                "mix_operator": mix_operator,
            }

    def _field(self, subgraph_id):

        nodes = self.local_data[subgraph_id]["nodes"]
        node_set = set(nodes)

        field = np.zeros(len(nodes), dtype=np.float64)
        constant = np.zeros(len(nodes), dtype=np.float64)

        for node_id, node in enumerate(nodes):

            for neighbor, attrs in self.full_graph[node].items():

                if neighbor in node_set:
                    continue

                weight = float(attrs.get("weight", 1.0))
                spin = float(self.full_graph.nodes[neighbor].get("spin", 0.0))

                field[node_id] += weight * spin
                constant[node_id] += weight

        return field, constant

    def _hamiltonian(self, subgraph_id):

        graph = self.subgraphs[subgraph_id]
        data = self.local_data[subgraph_id]

        nodes = data["nodes"]
        state_list = data["state_list"]
        total_states = len(state_list)

        adj = nx.to_numpy_array(
            graph,
            nodelist=nodes,
            weight="weight",
            dtype=np.float64
        )

        hamiltonian = np.zeros(total_states, dtype=np.float64)
        batch_size = min(1000, total_states)

        for i in range(0, total_states, batch_size):

            j = min(i + batch_size, total_states)
            batch_states = state_list[i:j]

            temp = batch_states @ adj
            batch_energies = -0.5 * np.einsum("ij,ij->i", temp, batch_states)

            hamiltonian[i:j] = batch_energies

        hamiltonian = -0.5 * (hamiltonian + 0.5 * np.sum(adj))

        field, constant = self._field(subgraph_id)

        hamiltonian -= self.field_amp * np.sum(
            state_list * field - constant,
            axis=1
        )

        return hamiltonian

    def _qaoa_state(self, subgraph_id, theta, hamiltonian):

        data = self.local_data[subgraph_id]

        state = data["init_state"].copy()
        mix_operator = data["mix_operator"]
        n_nodes = data["n_nodes"]

        gamma = theta[:self.n_layers]
        beta = theta[self.n_layers:2 * self.n_layers]

        for layer in range(self.n_layers):

            vector = np.exp(-1j * gamma[layer] * hamiltonian) * state

            for qubit in range(n_nodes):
                vector = (
                    vector * np.cos(beta[layer])
                    - 1j * vector[mix_operator[qubit]] * np.sin(beta[layer])
                )

            state = vector

        return state

    def _cost_function(self, subgraph_id, hamiltonian):

        def function(theta):

            state = self._qaoa_state(subgraph_id, theta, hamiltonian)
            probs = np.abs(state) ** 2

            energy = np.einsum("i,i->", probs, hamiltonian)
            energy = float(np.real(energy))

            self.energies.append(energy)

            return energy

        return function

    def _local_step(self, subgraph_id):

        hamiltonian = self._hamiltonian(subgraph_id)

        theta_old = self.theta_dct[subgraph_id].copy()

        if self.angle_step is None:
            bounds = None
        else:
            bounds = [
                (theta - self.angle_step, theta + self.angle_step)
                for theta in theta_old
            ]

        result = minimize(
            self._cost_function(subgraph_id, hamiltonian),
            theta_old,
            method=self.method,
            bounds=bounds,
            options={
                "maxiter": self.local_iter,
                "maxfun": max(20, 5 * len(theta_old)),
                "maxls": 5,
                "gtol": 1e-6,
            }
        )

        theta_candidate = result.x

        theta_new = (
            (1.0 - self.theta_mix) * theta_old
            + self.theta_mix * theta_candidate
        )

        state = self._qaoa_state(subgraph_id, theta_new, hamiltonian)
        probs = np.abs(state) ** 2

        local_spins = np.einsum(
            "i,ij->j",
            probs,
            self.local_data[subgraph_id]["state_list"]
        )

        local_energy = np.einsum("i,i->", probs, hamiltonian)
        local_energy = float(np.real(local_energy))

        spin_delta = 0.0

        for node, new_spin in zip(self.local_data[subgraph_id]["nodes"], local_spins):

            old_spin = float(self.full_graph.nodes[node].get("spin", 0.0))

            updated_spin = (
                (1.0 - self.spin_mix) * old_spin
                + self.spin_mix * float(new_spin)
            )

            updated_spin = float(np.clip(updated_spin, -1.0, 1.0))

            self.full_graph.nodes[node]["spin"] = updated_spin
            spin_delta += abs(updated_spin - old_spin)

        self.theta_dct[subgraph_id] = theta_new
        self.local_energy_dct[subgraph_id] = local_energy
        self.local_spins_dct[subgraph_id] = dict(
            zip(self.local_data[subgraph_id]["nodes"], local_spins)
        )

        self.iterations += 1

        global_solution = self._solution_from_spins()
        global_cut = self._cut_value(global_solution)
        global_energy = -global_cut

        self.history.append({
            "iteration": self.iterations,
            "subgraph_id": subgraph_id,
            "local_energy": local_energy,
            "global_cut": float(global_cut),
            "global_energy": float(global_energy),
            "spin_delta": float(spin_delta),
            "theta_delta": float(np.linalg.norm(theta_new - theta_old)),
        })

        if self.logs:
            print(f"Step : {self.iterations:<5d}  Energy : {global_energy:< .8f}")

    def solve(self):

        start_time = time.time()

        for _ in range(self.max_iter):
            for subgraph_id in range(len(self.subgraphs)):
                self._local_step(subgraph_id)

        self.time = time.time() - start_time

        self.solution = self._solution_from_spins()
        self.cut_value = self._cut_value(self.solution)
        self.energy = -self.cut_value

        self._callback()

        return self.solution

    def _solution_from_spins(self):

        return {
            node: 1 if self.full_graph.nodes[node].get("spin", 0.0) >= 0 else -1
            for node in self.full_graph.nodes()
        }

    def _cut_value(self, solution=None):

        if solution is None:
            solution = self._solution_from_spins()

        cut_value = 0.0

        for u, v, attrs in self.full_graph.edges(data=True):
            if solution[u] != solution[v]:
                cut_value += float(attrs.get("weight", 1.0))

        return cut_value

    def _callback(self):

        if self.callback:

            print("-------------------------------------")
            print(
                f"Full graph   || nodes : {self.full_graph.number_of_nodes()}   "
                f"edges : {self.full_graph.number_of_edges()}"
            )
            print(
                f"Subgraphs    || count : {len(self.subgraphs)}   "
                f"QAOA layers : {self.n_layers}"
            )
            print(
                f"Optimization || method : {self.method}   "
                f"local_iter : {self.local_iter}   "
                f"angle_step : {self.angle_step}   "
                f"theta_mix : {self.theta_mix}"
            )
            print(
                f"Result       || cut : {np.round(self.cut_value, 6)}   "
                f"energy : {np.round(self.energy, 6)}"
            )
            print(
                f"Processing   || time : {np.round(self.time, 4)} s"
            )


if __name__ == "__main__":
    rng = np.random.default_rng(7)

    G = DensRandGraph(number_of_nodes=100,
                      weights='pos_neg',
                      density=1,
                      seed=1000)

    # gw = GoemansWilliamsonMaxcut(graph=G,
    #                              no_hyperplane_samples=10,
    #                              callback=True)
    # gw.solve()

    config = SubgraphConfig(
        max_size=10,
        num_subgraphs=20,
        allow_overlap=True,
        random_seed=42)

    generator = OptimizedDenseSubgraphGenerator(G, config)
    partition = generator.generate_multiple_subgraphs()

    bits = {node: np.random.choice([-1, 1]) for node in G.nodes()}
    nx.set_node_attributes(G, bits, "bit")

    qaoa = QAOA_CF(graph=G,
                   subgraphs=partition,
                   n_layers=1,
                   field_amp=1.0,
                   spin_amp=0.01,
                   spin_mix=0.5,
                   theta_mix=0.5,
                   angle_step=0.0005,
                   max_iter=100,
                   local_iter=10,
                   logs=True,
                   callback=True)

    qaoa.solve()
