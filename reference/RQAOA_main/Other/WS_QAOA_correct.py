import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import time
import warnings
import cvxpy as cp
plt.rcParams.update({"font.family": "serif"})
warnings.filterwarnings('ignore')


def apply_single_qubit_gate(n, psi, U, k):
    mask = 1 << (n - 1 - k)
    flipped_indices = np.arange(2 ** n) ^ mask
    a_values = (np.arange(2 ** n) >> (n - 1 - k)) & 1
    result = np.where(a_values == 0,
                      U[0, 0] * psi + U[0, 1] * psi[flipped_indices],
                      U[1, 0] * psi[flipped_indices] + U[1, 1] * psi)
    return result


def apply_exp_pauli_x(n, state, betta):
    U = np.array([[np.cos(betta), -1j * np.sin(betta)],
                  [-1j * np.sin(betta), np.cos(betta)]], dtype=np.complex128)
    for target_qubit in range(n):
        state = apply_single_qubit_gate(n, state, U, target_qubit)
    return state


def create_Hamiltonian(n, weights):
    diagonal = np.zeros(2 ** n, dtype=np.float32)
    basis_states = np.arange(2 ** n, dtype=np.int32)
    for (i, j), w in weights.items():
        bit_i = (basis_states >> (n - 1 - i)) & 1
        bit_j = (basis_states >> (n - 1 - j)) & 1
        different_groups = (bit_i != bit_j)
        diagonal += w * different_groups.astype(np.float32)

    return -diagonal


class WSQAOA:

    def __init__(self, graph, n_layers, max_iter=300, logs=False,
                 callback=True, method='L-BFGS-B', opt_method='layerwise',
                 warm_start=True, gw_random_roundings=1, epsilon=0.25, mixer_type='standard',
                 initialization='ones'):

        self.graph = graph
        self.n_layers = n_layers
        self.max_iter = max_iter
        self.logs = logs
        self.callback = callback
        self.method = method
        self.opt_method = opt_method
        self.warm_start = warm_start
        self.epsilon = epsilon
        self.mixer_type = mixer_type
        self.initialization = initialization

        self.energies = np.array([])
        self.layers_borders = np.array([])
        self.result, self.iterations = 0, 0

        self.n_nodes = nx.number_of_nodes(self.graph)
        self.state_list = np.array([bin(s)[2:].zfill(self.n_nodes) for s in range(1 << self.n_nodes)])

        # Инициализация GW-решения и углов
        self.gw_random_roundings = gw_random_roundings
        self.gw_solution = None
        self.theta = None
        if self.warm_start:
            self.gw_solution = self._get_gw_solution(num_random_roundings=self.gw_random_roundings)
            self._prepare_warm_start_state()
        else:
            self.init_state = (np.ones(1 << self.n_nodes) / (np.sqrt(2) ** self.n_nodes))

        self.c_solution = []
        self.q_solution = []
        self.q_solution_dct = {}

        self.c_energy, self.q_energy = 0., 0.
        self.c_time, self.q_time = 0., 0.
        self.hamiltonian, self.mix_operator = np.array([]), np.array([])
        self.state, self.probs = np.array([]), np.array([])


    @staticmethod
    def _init_approximation(cost_function):

        n_samples = 25
        gammas = np.linspace(-2 * np.pi, 2 * np.pi, n_samples)
        betas = np.array([3 * np.pi / 4, 3 * np.pi / 8])

        e_arr = np.array([[cost_function([g, betas[0]]), cost_function([g, betas[1]])] for g in gammas])

        AB_arr = np.zeros((n_samples, 2))
        AB_arr[:, 0] = -1 * e_arr[:, 0]
        AB_arr[:, 1] = -1 * (e_arr[:, 0] * (np.sqrt(2) / 2) + e_arr[:, 1])

        root = np.zeros(n_samples).reshape(-1, 1)
        coef = np.zeros(n_samples).reshape(-1, 1)
        root[:, 0] = np.sqrt(AB_arr[:, 0] ** 2 + 32 * AB_arr[:, 1] ** 2)
        coef[:, 0] = (1 / (8 * AB_arr[:, 1]))

        t_arr = np.zeros((n_samples, 2))
        t_arr[:, 0] = coef[:, 0] * ((-1) * AB_arr[:, 0] + root[:, 0])
        t_arr[:, 1] = coef[:, 0] * ((-1) * AB_arr[:, 0] - root[:, 0])

        x_arr = np.arccos(t_arr) / 2

        f_arr = np.zeros((n_samples, 2))
        f_arr[:, 0] = AB_arr[:, 0] * np.sin(2 * x_arr[:, 0]) + AB_arr[:, 1] * np.sin(4 * x_arr[:, 0])
        f_arr[:, 1] = AB_arr[:, 0] * np.sin(2 * x_arr[:, 1]) + AB_arr[:, 1] * np.sin(4 * x_arr[:, 1])

        f_arr = np.nan_to_num(f_arr, nan=0)
        gamma_idx, beta_idx = np.where(f_arr == f_arr.min())
        gamma = gammas[gamma_idx[0]]
        beta = x_arr[gamma_idx[0], beta_idx[0]]

        return np.array([gamma, beta])

    def _get_gw_solution(self, solver="SCS", num_random_roundings=1):
        n = self.n_nodes
        nodes = list(self.graph.nodes())

        # Преобразуем граф в матрицу весов
        weight_matrix = nx.to_numpy_array(self.graph, nodelist=nodes, weight="weight")

        # Создаем переменную для полуопределенной матрицы X
        X = cp.Variable((n, n), symmetric=True)

        # Ограничения: X положительно полуопределена, диагональные элементы равны 1
        constraints = [X >> 0, cp.diag(X) == 1]

        # Целевая функция: максимизация суммы (w_ij * (1 - X[i, j])) / 4
        objective = cp.sum(cp.multiply(weight_matrix, 1 - X)) / 4

        # Формулируем и решаем задачу SDP
        problem = cp.Problem(cp.Maximize(objective), constraints)
        problem.solve(solver=solver, verbose=False)

        # Проверяем, что решение найдено
        if X.value is None:
            raise RuntimeError("Решение SDP не найдено.")

        # Извлекаем решение X
        X_value = X.value

        # Множественные случайные округления для лучшего результата
        best_cut = None
        best_cut_value = -np.inf

        for _ in range(num_random_roundings):
            # Генерируем случайный вектор для округления
            w = np.random.randn(n)

            # Округляем решение: cut[i] = sign(<X_i, w>), где X_i — i-я строка матрицы X
            cut = np.sign(X_value @ w)

            # Вычисляем величину разреза
            cut_value = np.sum(weight_matrix * (1 - np.outer(cut, cut))) / 4

            if cut_value > best_cut_value:
                best_cut_value = cut_value
                best_cut = cut

        # Преобразуем в бинарную строку: +1 -> 0, -1 -> 1
        binary_solution = ''.join(['0' if s > 0 else '1' for s in best_cut])

        return binary_solution

    def _prepare_warm_start_state(self):
        n = self.n_nodes
        self.theta = np.zeros(n)
        c_i = []
        for i in range(n):
            if self.gw_solution[i] == '1':
                c = 1 - self.epsilon
            else:
                c = self.epsilon
            c_i.append(c)
        self.theta = 2 * np.arcsin(np.sqrt(c_i))
        self.init_state = np.zeros(1 << n, dtype=np.complex128)
        self.init_state[0] = 1
        for i in range(n):
            Ry = np.array([
                [np.cos(self.theta[i] / 2), -np.sin(self.theta[i] / 2)],
                [np.sin(self.theta[i] / 2), np.cos(self.theta[i] / 2)]
            ], dtype=np.complex128)
            self.init_state = apply_single_qubit_gate(n, self.init_state, Ry, i)

    def _apply_warm_start_mixer(self, n, state, beta, mixer_type='ws'):
        new_state = state.copy()

        for i in range(n):
            # Создаем матрицы вращений
            # Стандартный теплый стартовый миксер: RY(theta_i) RZ(-2β) RY(-theta_i)
            Ry_theta = np.array([
                [np.cos(self.theta[i] / 2), -np.sin(self.theta[i] / 2)],
                [np.sin(self.theta[i] / 2), np.cos(self.theta[i] / 2)]
            ], dtype=np.complex128)
            Ry_minus_theta = np.array([
                [np.cos(-self.theta[i] / 2), -np.sin(-self.theta[i] / 2)],
                [np.sin(-self.theta[i] / 2), np.cos(-self.theta[i] / 2)]
            ], dtype=np.complex128)
            if mixer_type == 'ws_modified':
                # Модифицированный миксер для округленного старта: RY(-theta_i) RZ(-2β) RY(theta_i)
                Ry_theta, Ry_minus_theta = Ry_minus_theta, Ry_theta

            # RZ(-2β) = diag(exp(iβ), exp(-iβ))
            Rz = np.array([
                [np.exp(1j * beta), 0],
                [0, np.exp(-1j * beta)]
            ], dtype=np.complex128)

            # Комбинированное вращение U = Ry_theta @ Rz @ Ry_minus_theta
            new_state = apply_single_qubit_gate(n, new_state, Ry_minus_theta, i)
            new_state = apply_single_qubit_gate(n, new_state, Rz, i)
            new_state = apply_single_qubit_gate(n, new_state, Ry_theta, i)

        return new_state

    def _maxcut_classical_solver(self):

        start_time = time.time()
        mapping = dict(zip(self.graph.nodes(), range(self.n_nodes)))
        edges_weights = {}
        for u, v, weight in self.graph.edges(data='weight'):
            i, j = mapping[u], mapping[v]
            edges_weights[(i, j)] = weight
        self.hamiltonian = create_Hamiltonian(self.n_nodes, edges_weights)

        self.c_energy = np.min(self.hamiltonian)
        min_indices = np.where(self.hamiltonian == self.c_energy)[0]
        self.c_solutions = [self.state_list[i] for i in min_indices]
        self.c_solution = self.state_list[min_indices[0]]

        self.c_time = time.time() - start_time

    def _cost_function(self, depth, final=False):
        def function(angles):
            gamma = angles[0:depth]
            beta = angles[depth:2 * depth]
            state = np.copy(self.init_state)
            for k in range(depth):
                state = np.exp(-1j * gamma[k] * self.hamiltonian) * state
                if self.warm_start:
                    state = self._apply_warm_start_mixer(self.n_nodes, state, beta[k], self.mixer_type)
                else:
                    state = apply_exp_pauli_x(self.n_nodes, state, beta[k])
            energy = np.vdot(state, self.hamiltonian * state)
            self.energies = np.append(self.energies, np.real(energy))
            self.iterations += 1

            if final:
                return state, np.real(energy)
            else:
                return np.real(energy)

        return function

    def _simultaneous_optimization(self):

        start_opt = time.time()
        init_params = np.ones(2 * self.n_layers)

        result_min = minimize(self._cost_function(self.n_layers),
                              init_params,
                              method=self.method,
                              options={'maxiter': self.max_iter,
                                       'maxfun': self.max_iter,
                                       'gtol': 1e-5})

        optimal_angles = result_min.x
        fin_function = self._cost_function(self.n_layers, final=True)
        self.state, self.q_energy = fin_function(optimal_angles)
        self.layers_borders = np.append(self.layers_borders, self.iterations)

        self.q_time = time.time() - start_opt

    def _layerwise_optimization(self):

        start_opt = time.time()
        if self.initialization != 'ones':
            init_params = self._init_approximation(self._cost_function(1))
        else:
            init_params = np.ones(2)

        for p in range(1, self.n_layers + 1):

            start_time = time.time()
            result_min = minimize(self._cost_function(p),
                                  init_params,
                                  method=self.method,
                                  options={'maxiter': self.max_iter,
                                           'maxfun': self.max_iter,
                                           'gtol': 1e-8})
            optimal_angles = result_min.x
            init_params = np.insert(optimal_angles, len(optimal_angles) // 2, 1)
            init_params = np.append(init_params, 1)
            fin_function = self._cost_function(p, final=True)
            self.state, self.q_energy = fin_function(optimal_angles)
            self.layers_borders = np.append(self.layers_borders, self.iterations)

            if self.logs:
                print("\033[37m {}".format(
                      f"Layer : {p:<2d}  "
                      f"Energy : {self.q_energy:<4f}   "
                      f"Processing time : {(time.time() - start_time):<4f}"))

        self.q_time = time.time() - start_opt

    def solve(self):

        self._maxcut_classical_solver()

        if self.opt_method == 'layerwise':
            self._layerwise_optimization()
        elif self.opt_method == 'simultaneous':
            self._simultaneous_optimization()
        else:
            raise ValueError(f"Unknown optimization method: {self.opt_method}")

        self.probs = np.abs(self.state) ** 2
        self.q_solution = self.state_list[np.argmax(self.probs)]
        self.q_solution_dct = dict(zip(self.graph.nodes(), [2 * int(bit) - 1 for bit in self.q_solution]))

        self.result = 1 if self.c_solution == self.q_solution else 0
        WSQAOA._callback(self)

    def _callback(self):

        print("\033[37m {}".format(f"-------------------------------------"))

        if self.result == 1:
            print("\033[32m {}".format("***SUCCESS***"))
        else:
            print("\033[31m {}".format("***FAIL***"))

        print(f"Graph       ||   number of nodes : {self.n_nodes}  number of edges : {nx.number_of_edges(self.graph)} ")
        print(f"QAOA        ||   number of layers : {self.n_layers}   optimization method : {self.method}")
        print(f"Classical   ||   solution : {self.c_solutions[0]}    "
              f"Energy : {np.round(self.c_energy, 4)}")
        print(f"Quantum     ||   solution : {self.q_solution}    "
              f"Energy : {np.round(self.q_energy, 4)}")
        print(f"Processing  ||   QAOA time : {np.round(self.q_time, 4)} s   "
              f"Classic time : {np.round(self.c_time, 4)} s")
        print(f'Total       ||   Ratio : {np.round(self.q_energy / self.c_energy, 4)}')

    def plot_convergence(self):

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(list(range(self.iterations)),
                 self.energies,
                 linewidth=0.7,
                 label='Convergence line')

        for coord in self.layers_borders:
            ax.axvline(x=coord,
                       color='red',
                       linestyle='--',
                       alpha=0.7,
                       linewidth=0.7)

        ax.axhline(y=self.c_energy,
                   color='green',
                   linestyle='--',
                   alpha=0.7,
                   linewidth=0.7,
                   label='Exact energy')

        ax.axhline(y=0,
                   color='red',
                   linestyle='--',
                   alpha=0.7,
                   linewidth=0.7,
                   label='Parameter shift')

        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        max_y = 0.9 * max(self.energies)
        min_y = 1.1 * self.c_energy
        ax.set_ylim(min_y, max_y)

        # plt.title(f'Energy convergence graph for parameter optimization for layerwise-QAOA\n'
        #           f'Number of nodes: {self.n_nodes}   Number of edges {nx.number_of_edges(self.graph)}\n'
        #           f'QAOA layers: {self.n_layers}   Ratio: {np.round(self.q_energy / self.c_energy, 4)}\n'
        #           f'Warm-start: {self.warm_start}   Epsilon: {self.epsilon}   Mixer type: {self.mixer_type}',
        #           fontsize=16, pad=20)
        plt.title(f'Number of nodes: {self.n_nodes}   QAOA layers: {self.n_layers}   Ratio: {np.round(self.q_energy / self.c_energy, 4)}\n'
                  f'Warm-start: {self.warm_start}   Epsilon: {self.epsilon}   Mixer type: {self.mixer_type}',
                  fontsize=16, pad=20)
        plt.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlabel('Optimization iterations', fontsize=16, labelpad=10)
        ax.set_ylabel('Energy', fontsize=16, labelpad=10)
        ax.legend(loc='upper right', fontsize=12)
        sns.despine()
        plt.show()



if __name__ == '__main__':

    n = 12
    while True:
        G = nx.erdos_renyi_graph(n, p=0.3)
        if nx.is_connected(G):
            break
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(0, 2)

    q = WSQAOA(graph=G,
             n_layers=10,
             max_iter=1000,
             logs=True,
             callback=True,
             method='L-BFGS-B',
             warm_start=False,
             gw_random_roundings=1,
             epsilon=0.5,
             mixer_type='standard',
             initialization='approx')


    q.solve()
    q.plot_convergence()

