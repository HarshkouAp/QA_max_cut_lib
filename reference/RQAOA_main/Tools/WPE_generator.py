import numpy as np
import networkx as nx
from Laboratory.LS_RQAOA_fast_5 import *


class WishartMaxCutGenerator:
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

    wpe = WishartMaxCutGenerator(seed=42)
    G, sol, cut = wpe.generate(10, 50)

    print(cut)

    bits = dict(zip(G.nodes(), np.zeros(int(G.number_of_edges()))))
    nx.set_node_attributes(G, bits, "bit")

    lsq = LargeScaleRQAOA_1(graph=G,
                            max_size=10,
                            part_iter=1,
                            treshold=3,
                            n_layers=3,
                            w_scale=1,
                            logs=True,
                            callback=True,
                            max_iter=500,
                            random_seed=1000)
    lsq.solve()
