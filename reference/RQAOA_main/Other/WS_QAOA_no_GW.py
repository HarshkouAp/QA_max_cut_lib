# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import networkx as nx
# import warnings
# import cvxpy as cp

import numpy as np
import networkx as nx
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import time
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


class QAOA:

    def __init__(self, graph, n_layers, max_iter=300, logs=False,
                 callback=True, method='L-BFGS-B', opt_method='layerwise',
                 warm_start=True, epsilon=0.25, mixer_type='standard',
                 initialization='ones', initial_state=None):

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
        self.initial_state = initial_state

        self.energies = np.array([])
        self.layers_borders = np.array([])
        self.result, self.iterations = 0, 0

        self.n_nodes = nx.number_of_nodes(self.graph)
        self.state_list = np.array([bin(s)[2:].zfill(self.n_nodes) for s in range(1 << self.n_nodes)])

        # Инициализация теплого решения и углов
        self.theta = None
        if self.warm_start:
            self._prepare_warm_start_state()
        else:
            self.init_state = (np.ones(1 << self.n_nodes) / (np.sqrt(2) ** self.n_nodes))

        self.c_solutions = []
        self.q_solution = ''
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

        f_arr = np.nan_to_num(f_arr, nan=1000)
        gamma_idx, beta_idx = np.where(f_arr == f_arr.min())
        gamma = gammas[gamma_idx[0]]
        beta = x_arr[gamma_idx[0], beta_idx[0]]

        return np.array([gamma, beta])


    def _prepare_warm_start_state(self):
        n = self.n_nodes
        self.theta = np.zeros(n)
        c_i = []
        for i in range(n):
            if self.initial_state[i] == '1':
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

    def _apply_warm_start_mixer(self, n, state, beta, mixer_type='standard'):
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

    # def _simultaneous_optimization(self):
    #     start_opt = time.time()
    #     init_params = np.ones(2 * self.n_layers)
    #
    #     result_min = minimize(self._cost_function(self.n_layers),
    #                           init_params,
    #                           method=self.method,
    #                           options={'maxiter': self.max_iter,
    #                                    'maxfun': self.max_iter,
    #                                    'gtol': 1e-5})
    #
    #     optimal_angles = result_min.x
    #     fin_function = self._cost_function(self.n_layers, final=True)
    #     self.state, self.q_energy = fin_function(optimal_angles)
    #     self.layers_borders = np.append(self.layers_borders, self.iterations)
    #
    #     self.q_time = time.time() - start_opt

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
                                           'gtol': 1e-6})
            optimal_angles = result_min.x
            init_params = np.insert(optimal_angles, len(optimal_angles) // 2, 1)
            init_params = np.append(init_params, 1)
            fin_function = self._cost_function(p, final=True)
            self.state, self.q_energy = fin_function(optimal_angles)
            self.layers_borders = np.append(self.layers_borders, self.iterations)

            # if self.logs:
            #     print("\033[37m {}".format(
            #           f"Layer : {p:<2d}  "
            #           f"Energy : {self.q_energy:<4f}   "
            #           f"Processing time : {(time.time() - start_time):<4f}"))

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
        # self.q_solution_dct = dict(zip(self.graph.nodes(), [2 * int(bit) - 1 for bit in self.q_solution]))
        self.result = 1 if self.q_solution in self.c_solutions else 0
        if self.callback:
            self._callback()

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


def get_initial_states_for_graph(graph, num_states=5):

    n_nodes = nx.number_of_nodes(graph)

    # Вычисляем гамильтониан
    mapping = dict(zip(graph.nodes(), range(n_nodes)))
    edges_weights = {}
    for u, v, weight in graph.edges(data='weight'):
        i, j = mapping[u], mapping[v]
        edges_weights[(i, j)] = weight
    hamiltonian = create_Hamiltonian(n_nodes, edges_weights)

    # Находим оптимальную энергию
    min_energy = np.min(hamiltonian)

    # Находим все состояния в нужном диапазоне
    mask = (hamiltonian >= 0.85 * min_energy) & (hamiltonian <= 0.8 * min_energy)
    candidate_indices = np.where(mask)[0]

    # if len(candidate_indices) < num_states:
    #     mask = (hamiltonian >= 0.85 * min_energy) & (hamiltonian <= 0.75 * min_energy)
    #     candidate_indices = np.where(mask)[0]

    # Выбираем m случайных состояний (или меньше, если недостаточно)
    m = min(num_states, len(candidate_indices))
    selected_indices = np.random.choice(candidate_indices, size=m, replace=False)

    # Преобразуем индексы в битовые строки
    state_list = np.array([bin(s)[2:].zfill(n_nodes) for s in range(1 << n_nodes)])
    selected_states = [state_list[idx] for idx in selected_indices]

    return selected_states


def collect_statistics(num_graphs=10, num_epsilons=30, num_st=5, n_nodes=10,
                      n_layers=1, mixer_type='standard'):
    """Собирает статистику по зависимости Ratio от epsilon."""

    # Создаем графы
    print("Создание графов...")
    graphs = []
    for i in range(num_graphs):
        while True:
            G = nx.erdos_renyi_graph(n_nodes, p=0.3)
            if nx.is_connected(G):
                for u, v in G.edges():
                    G[u][v]['weight'] = np.random.randint(-10, 11)
                graphs.append(G)
                break

    # Получаем начальные состояния для каждого графа
    print("Поиск начальных состояний для каждого графа...")
    initial_states_per_graph = []
    for graph_id, graph in enumerate(tqdm(graphs, desc="Графы")):
        initial_states = get_initial_states_for_graph(graph, num_states=num_st)
        initial_states_per_graph.append(initial_states)
        if len(initial_states) < num_st:
            print('initial_states) < num_st, STOP!!!!!!!!!!!!')

    # Значения epsilon
    epsilons = np.linspace(0, 0.5, num_epsilons)

    # Массивы для хранения результатов
    all_results = {
        'epsilon': [],
        'ratio': [],
        'graph_id': [],
        'run_id': [],
        'initial_state': []
    }

    # Основной цикл сбора данных
    print("Запуск экспериментов...")
    for graph_id, (graph, initial_states) in enumerate(tqdm(zip(graphs, initial_states_per_graph),
                                                            desc="Эксперименты", total=len(graphs))):
        for epsilon_idx, epsilon in enumerate(tqdm(epsilons, desc=f"Epsilon (граф {graph_id + 1})", leave=False)):
            for run_id in range(num_st):
                # Берем соответствующее начальное состояние для этого запуска
                initial_state = initial_states[run_id]

                # Запускаем QAOA
                q = QAOA(
                    graph=graph,
                    n_layers=n_layers,
                    max_iter=4000,
                    logs=False,
                    callback=False,
                    method='L-BFGS-B',
                    opt_method='layerwise',
                    warm_start=True,
                    epsilon=epsilon,
                    mixer_type=mixer_type,
                    initialization='no ones',
                    initial_state=initial_state
                )
                q.solve()

                # Сохраняем результаты
                ratio = abs(q.q_energy / q.c_energy)
                all_results['epsilon'].append(epsilon)
                all_results['ratio'].append(ratio)
                all_results['graph_id'].append(graph_id)
                all_results['run_id'].append(run_id)
                all_results['initial_state'].append(initial_state)

    return all_results, graphs


def plot_statistics(all_results, num_st, mixer_type='standard'):
    """Строит график зависимости Ratio от epsilon."""
    # Преобразуем в numpy массивы для удобства
    epsilons = np.array(all_results['epsilon'])
    ratios = np.array(all_results['ratio'])

    # Уникальные значения epsilon
    unique_eps = np.unique(epsilons)

    # Вычисляем статистики для каждого epsilon
    mean_ratios = []
    std_ratios = []

    for eps in unique_eps:
        mask = epsilons == eps
        ratios_for_eps = ratios[mask]
        mean_ratios.append(np.mean(ratios_for_eps))
        std_ratios.append(np.std(ratios_for_eps))

    # Создаем график
    fig, ax = plt.subplots(figsize=(12, 8))

    # Все точки (10 графов × 5 запусков × 30 epsilon = 1500 точек)
    ax.scatter(epsilons, ratios, alpha=0.4, s=10, color='gray')

    # Среднее значение
    ax.plot(unique_eps, mean_ratios, 'o-', color='blue', linewidth=3)

    # Область стандартного отклонения
    ax.fill_between(unique_eps,
                    np.array(mean_ratios) - np.array(std_ratios),
                    np.array(mean_ratios) + np.array(std_ratios),
                    alpha=0.3, color='blue')

    # Настройки графика
    ax.set_xlabel('Regularization parameter ε', fontsize=16)
    ax.set_ylabel('Ratio', fontsize=16)
    # ax.set_title(f'Зависимость Ratio от ε\n'
    #              f'Mixer: {mixer_type}\n'
    #              f'Графы: {len(np.unique(all_results["graph_id"]))}, '
    #              f'Значений ε: {len(unique_eps)}, '
    #              f'Запусков на точку: {num_st}',
    #              fontsize=16)
    ax.set_title(f'Mixer: {mixer_type}\n',
                 fontsize=16)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Добавляем статистику
    # textstr = f'Всего точек: {len(ratios)}\n'
    # textstr += f'Среднее Ratio: {np.mean(ratios):.4f}\n'
    # textstr += f'Станд. отклонение: {np.std(ratios):.4f}'
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
    #         verticalalignment='top', bbox=props)

    plt.tight_layout()
    return fig, ax


if __name__ == '__main__':

    NUM_GRAPHS = 15
    NUM_EPSILONS = 30
    NUM_RUNS = 4
    N_NODES = 14
    N_LAYERS = 1
    MIXER_TYPE = 'ws_modified'
    # MIXER_TYPE = 'standard'


    print("=" * 60)
    print("СБОР СТАТИСТИКИ ДЛЯ WS-QAOA")
    print(f"Графы: {NUM_GRAPHS}, Epsilon: {NUM_EPSILONS} значений, Запусков на точку: {NUM_RUNS}")
    print("=" * 60)

    # Сбор данных
    all_results, graphs = collect_statistics(
        num_graphs=NUM_GRAPHS,
        num_epsilons=NUM_EPSILONS,
        num_st=NUM_RUNS,
        n_nodes=N_NODES,
        n_layers=N_LAYERS,
        mixer_type=MIXER_TYPE
    )

    # Вывод статистики
    # print_summary_statistics(all_results)

    # Построение графика
    print("\nПостроение графика...")
    fig, ax = plot_statistics(all_results, num_st=NUM_RUNS, mixer_type=MIXER_TYPE)

    # Сохранение графика
    # output_file = f"POSITIVE{N_NODES}nodes_{NUM_GRAPHS}graphs_{NUM_RUNS}runs_{MIXER_TYPE}.png"
    output_file = f"NEGATIVE{N_NODES}nodes_{NUM_GRAPHS}graphs_{NUM_RUNS}runs_{MIXER_TYPE}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"График сохранен как '{output_file}'")

    # Показ графика
    plt.show()
