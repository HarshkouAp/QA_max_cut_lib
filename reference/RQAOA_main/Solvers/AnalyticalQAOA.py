"""Однослойный QAOA: аналитический Max-Cut и восстановление по двум beta.

Зависимости: numpy, networkx; matplotlib нужен только для графиков.
Конвенция этого класса: C = sum((I - Zi*Zj)/2), U = exp(-i beta B)
exp(-i gamma C), B = sum(Xi). cost_function возвращает ожидаемый разрез.
В приложенном QAOA.py используется -C: для него gamma меняет знак.

Поддерживаются простые неориентированные графы с единичными весами.
Класс ищет углы и средний разрез, но не генерирует битовую строку.
Сетка gamma приближённая; восстановление и оптимизация beta точные.
"""

from collections import Counter
from time import perf_counter
from Generators.GraphGenerator import DensRandGraph

import networkx as nx
import numpy as np



class AnalyticalQAOA:
    """Интерфейс в стиле QAOA(graph, ..., logs=False, callback=False).

    samples: число узлов gamma в [0, pi], включая границы (минимум 2).
    logs: краткий вывод этапов; callback: итоговая сводка после solve().
    Граф копируется при создании. Для изменённого графа создайте новый объект.

    Основные результаты после solve():
      optimal_angles = [gamma, beta] для +C;
      optimal_angles_qaoa = [-gamma, beta] для приложенного QAOA.py;
      q_energy / expected_cut: положительное среднее C;
      energy: среднее -C;
      coefficients = [A0, As, Ac] при лучшем gamma;
      gammas, betas, energies: профиль max_beta C на сетке gamma;
      sampled_costs: две оценки C на каждый gamma, shape (samples, 2);
      coefficient_grid: коэффициенты всех сечений, shape (samples, 3).

    solve() возвращает self. result содержит краткие результаты в словаре.
    iterations считает узлы gamma, cost_evaluations — скалярные оценки
    аналитической функции внутри solve(), включая итоговую проверку.
    Квантовые запуски и shots не выполняются.
    """

    def __init__(self, graph, samples=10000, logs=False, callback=False):
        if (isinstance(samples, (bool, np.bool_))
                or not isinstance(samples, (int, np.integer)) or samples < 2):
            raise ValueError("samples должно быть целым числом >= 2")
        if graph.is_directed() or graph.is_multigraph():
            raise ValueError("Нужен простой неориентированный граф")
        if nx.number_of_selfloops(graph):
            raise ValueError("Петли не поддерживаются")
        for _, _, data in graph.edges(data=True):
            if data.get("weight", 1) != 1:
                raise ValueError("Аналитическая формула поддерживает только вес 1")

        self.graph = nx.freeze(nx.Graph(graph))
        self.samples = int(samples)
        self.logs = bool(logs)
        self.callback = bool(callback)
        self.n_layers = 1
        self.n_nodes = self.graph.number_of_nodes()
        self.n_edges = self.graph.number_of_edges()
        self.kappa = self.n_edges / 2.0
        self.method = "Gamma grid + exact beta reconstruction"
        self.gamma_bounds = (0.0, np.pi)
        self.beta_bounds = (0.0, np.pi / 2)
        self.sample_betas = np.array([3 * np.pi / 4, 3 * np.pi / 8])

        start = perf_counter()
        neighbors = {u: set(self.graph[u]) for u in self.graph}
        patterns = Counter()
        for u, v in self.graph.edges():
            du, dv = sorted((len(neighbors[u]) - 1, len(neighbors[v]) - 1))
            # Перебор меньшего множества без создания списка общих соседей.
            small, large = neighbors[u], neighbors[v]
            if len(small) > len(large):
                small, large = large, small
            triangles = sum(w in large for w in small)
            patterns[(du, dv, triangles)] += 1
        # Рёбра с одинаковыми (du, dv, triangles) имеют одинаковый вклад.
        self._patterns = tuple((*key, count) for key, count in patterns.items())
        self.n_patterns = len(self._patterns)
        self.preprocessing_time = perf_counter() - start
        self._reset_results()

    def _reset_results(self):
        self.solved = False
        self.result = None
        self.gamma = self.beta = None
        self.q_energy = self.expected_cut = self.energy = None
        self.optimal_angles = np.array([])
        self.optimal_angles_qaoa = np.array([])
        self.coefficients = np.array([])
        self.gammas = self.betas = self.energies = np.array([])
        self.sampled_costs = np.empty((0, 2))
        self.coefficient_grid = np.empty((0, 3))
        self.best_idx = None
        self.iterations = self.cost_evaluations = 0
        self.reconstruction_error = None
        self.q_time = 0.0

    def cost_function(self, gamma, beta):
        """Точное <C> при p=1; принимает числа или совместимые массивы.

        Не создаёт квантовое состояние или матрицу гамильтониана.
        Память пропорциональна размеру выходного массива, а не 2**n.
        """
        gamma, beta = np.broadcast_arrays(
            np.asarray(gamma, dtype=float), np.asarray(beta, dtype=float)
        )
        if not (np.all(np.isfinite(gamma)) and np.all(np.isfinite(beta))):
            raise ValueError("Углы должны быть конечными числами")
        cg, sg = np.cos(gamma), np.sin(gamma)
        c2g = np.cos(2 * gamma)
        s4b, s2b_sq = np.sin(4 * beta), np.sin(2 * beta) ** 2
        value = np.full(gamma.shape, self.kappa, dtype=float)
        for du, dv, nt, count in self._patterns:
            value += count * (
                0.25 * s4b * sg * (cg ** du + cg ** dv)
                - 0.25 * s2b_sq * cg ** (du + dv - 2 * nt)
                * (1 - c2g ** nt)
            )
        return float(value) if value.ndim == 0 else value

    def solve(self):
        """Перезапускает поиск; максимум гарантирован только на сетке gamma."""
        self._reset_results()
        start = perf_counter()
        if self.logs:
            print(f"AnalyticalQAOA: {self.n_nodes} nodes, {self.n_edges} edges, "
                  f"{self.n_patterns} edge patterns; {self.samples} gamma points")

        self.gammas = np.linspace(*self.gamma_bounds, self.samples)
        self.sampled_costs = np.column_stack([
            self.cost_function(self.gammas, beta) for beta in self.sample_betas
        ])
        y0, y1 = self.sampled_costs.T
        a0 = (self.kappa + y0) / 2
        a_cos = (self.kappa - y0) / 2
        a_sin = a0 - y1
        self.coefficient_grid = np.column_stack((a0, a_sin, a_cos))

        amplitude = np.hypot(a_sin, a_cos)
        self.betas = (np.arctan2(a_sin, a_cos) / 4) % (np.pi / 2)
        self.betas[amplitude == 0] = 0.0
        self.energies = a0 + amplitude  # Ожидаемый разрез, максимизируем.
        self.best_idx = int(np.argmax(self.energies))
        self.gamma = float(self.gammas[self.best_idx])
        self.beta = float(self.betas[self.best_idx])
        self.optimal_angles = np.array([self.gamma, self.beta])
        self.optimal_angles_qaoa = np.array([-self.gamma, self.beta])
        self.coefficients = self.coefficient_grid[self.best_idx].copy()
        self.expected_cut = self.cost_function(self.gamma, self.beta)
        self.q_energy = self.expected_cut
        self.energy = -self.expected_cut
        self.reconstruction_error = float(abs(
            self.expected_cut - self.energies[self.best_idx]
        ))
        self.iterations = self.samples
        self.cost_evaluations = 2 * self.samples + 1
        self.q_time = perf_counter() - start  # Классическое время поиска.
        self.solved = True
        self.result = {
            "gamma": self.gamma, "beta": self.beta,
            "expected_cut": self.expected_cut, "energy": self.energy,
            "samples": self.samples, "cost_evaluations": self.cost_evaluations,
            "search_time": self.q_time,
            "reconstruction_error": self.reconstruction_error,
        }
        if self.logs:
            print(f"Search finished in {self.q_time:.4f} s; "
                  f"expected cut = {self.expected_cut:.8f}")
        if self.callback:
            self._callback()
        return self

    def _require_solved(self):
        if not self.solved:
            raise RuntimeError("Сначала вызовите solve()")

    def reconstructed_cost(self, beta):
        """Восстановленное сечение при найденном gamma для любого beta."""
        self._require_solved()
        a0, a_sin, a_cos = self.coefficients
        beta = np.asarray(beta, dtype=float)
        return a0 + a_sin * np.sin(4 * beta) + a_cos * np.cos(4 * beta)

    def summary(self):
        """Возвращает сводку; можно вызвать print(solver) в любое время."""
        lines = [
            "AnalyticalQAOA | p = 1 | unweighted Max-Cut",
            f"Graph          | nodes: {self.n_nodes}, edges: {self.n_edges}",
            f"Search         | gamma: [0, pi], points: {self.samples}; beta: exact",
            f"Preprocessing  | patterns: {self.n_patterns}, "
            f"time: {self.preprocessing_time:.4f} s",
        ]
        if not self.solved:
            return "\n".join(lines + ["Status         | call solve() to start"])
        lines.extend([
            "Status         | best on gamma grid; continuous optimum not certified",
            f"Angles (+C)    | gamma: {self.gamma:.10f}, beta: {self.beta:.10f} rad",
            f"Angles (-C)    | gamma: {-self.gamma:.10f}, beta: {self.beta:.10f} rad",
            f"Expected cut   | {self.expected_cut:.10f}; random baseline: {self.kappa:g}",
            f"Energy (-C)    | {self.energy:.10f}",
            f"Coefficients   | A0, As, Ac: {self.coefficients}",
            f"Evaluations    | {self.cost_evaluations} analytical scalar values",
            f"Check          | reconstruction error: {self.reconstruction_error:.3e}",
            f"Search time    | {self.q_time:.4f} s (classical computation)",
            "Output         | expected cut and angles; no sampled bitstring",
        ])
        return "\n".join(lines)

    def _callback(self):
        print(self.summary())

    def __str__(self):
        return self.summary()

    def plot_convergence(self, show=True):
        """Профиль max_beta F(gamma), а не история итеративного оптимизатора.

        Возвращает (fig, ax). При show=False график можно настроить или сохранить.
        """
        self._require_solved()
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.gammas, self.energies, label="Exact beta maximum on gamma grid")
        ax.scatter([self.gamma], [self.expected_cut], color="red", label="Best grid point")
        ax.axhline(self.kappa, linestyle="--", color="gray", label="Random baseline")
        ax.set(xlabel="Gamma (rad)", ylabel="Expected cut",
               title=f"Analytical QAOA, p=1: {self.n_nodes} nodes, {self.n_edges} edges")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax


if __name__ == "__main__":

    G = DensRandGraph(
        number_of_nodes=10,
        density=0.7,
        weights="ones",
        seed=1000,
    )

    qaoa = AnalyticalQAOA(graph=G,
                          samples=10000,
                          logs=True,
                          callback=True)
    qaoa.solve()
