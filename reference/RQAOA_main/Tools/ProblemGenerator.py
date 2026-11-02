import numpy as np
import networkx as nx
from QAOA import QAOA
import random


class ProblemGenerator:

    @staticmethod
    def _cut_value(graph, planted_vector):
        A = nx.to_numpy_array(graph)
        s = np.asarray(planted_vector, dtype=float)
        return float(0.25 * (A.sum() - s @ A @ s))

    @staticmethod
    def _correlation(graph):
        N = graph.number_of_nodes()
        A = nx.to_numpy_array(graph)
        row_corr = np.corrcoef(A, rowvar=True)
        col_corr = np.corrcoef(A, rowvar=False)
        row_corr = np.nan_to_num(row_corr, nan=0.0)
        col_corr = np.nan_to_num(col_corr, nan=0.0)

        return np.sum(np.abs(row_corr + col_corr)) / (2 * N**2)

    @staticmethod
    def recover_binary_vector_soft_fast(A, alpha=1.0, beta=0.5, refine_steps=20):
        S = np.sign(np.asarray(A, dtype=float))
        np.fill_diagonal(S, 1.0)

        Rr = np.nan_to_num(np.corrcoef(S, rowvar=True), nan=0.0)
        Rc = np.nan_to_num(np.corrcoef(S, rowvar=False), nan=0.0)

        C = alpha * S + 0.5 * beta * (Rr + Rc)
        C = 0.5 * (C + C.T)
        np.fill_diagonal(C, 0.0)

        x = np.sign(np.linalg.eigh(C)[1][:, -1])
        x[x == 0] = 1.0

        for _ in range(refine_steps):
            y = np.where(C @ x >= 0, 1.0, -1.0)
            if np.array_equal(y, x):
                break
            x = y

        return x.astype(np.int8)


class HebbianMaxCutGenerator(ProblemGenerator):

    def __init__(self,seed=None):
        self.rng = random.Random(seed)

    def generate(self, M, N):
        mins = np.linspace(start=1, stop=N, num=M) # распределение можно сделать произвольным
        mins[-1] = 2 * N                      # Самый простой способ избежать появления побочного решения
                                              # Влияет на корреляции внутри матрицы смежности
        s_gen = np.array([self.rng.randrange(1, (1 << N)) for _ in range(M)])
        ss = 2 * ((s_gen[:, None] >> np.arange(N - 1, -1, -1)) & 1) - 1
        mask = np.einsum('ni,nj->nij', ss, ss)
        A = np.sum(mask * mins[:, np.newaxis, np.newaxis], axis=0) * (-1) / N
        np.fill_diagonal(A, 0.0)
        graph = nx.from_numpy_array(A)

        return graph, ss[-1], self._cut_value(graph, ss[-1])



class WishartMaxCutGenerator(ProblemGenerator):
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def generate(self, M, N):
        t = self.rng.choice((-1.0, 1.0), size=N)
        Z = self.rng.standard_normal((N, M))
        W = np.sqrt(N / (N - 1.0)) * (Z - np.outer(t, (t @ Z) / N))
        J = -(W @ W.T)
        np.fill_diagonal(J, 0.0)

        A = -J
        solution = t.astype(np.int8)
        cut_energy = float(0.25 * (A.sum() - solution @ A @ solution))
        graph = nx.from_numpy_array(np.triu(A, 1))

        return graph, solution, cut_energy


if __name__ == '__main__':

    gen = WishartMaxCutGenerator()
    G, sol, cut = gen.generate(20, 10)

    print(f'Разрез: {cut}')
    print(f'Корреляции: {gen._correlation(G)}')

    if G.number_of_nodes() <= 16:

        qaoa = QAOA(graph=G,
                    n_layers=5,
                    logs=True,
                    max_iter=1000,
                    samples=1000,
                    callback=True,
                    initialization="no ones")
        qaoa.solve()

    else:
        gw = GoemansWilliamsonMaxcut(graph=G,
                                     no_hyperplane_samples=20,
                                     callback=True)
        gw.solve()

