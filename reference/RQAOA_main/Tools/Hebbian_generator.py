import numpy as np
import networkx as nx
from Solvers.QAOA import QAOA


class MaxCut_definite_generator():

    def __init__(self,
                 number_of_nodes=5,
                 density=1,
                 number_of_mins=4,
                 distribution="Normal"):

        self.n_nodes = number_of_nodes
        self.density = density
        self.n_mins = number_of_mins
        self.distribution = distribution


    def generate_problem(self):

        mins = np.arange(start=1, stop=self.n_mins + 1, step=0.1)
        print(mins.shape)
        state_gen = ((2 ** self.n_nodes) * np.random.sample(self.n_mins)).astype(int)
        states = 2 * ((state_gen[:, None]
                           >> np.arange(self.n_nodes - 1, -1, -1)) & 1) - 1

        mask = np.einsum('ni,nj->nij', states, states)
        print(mask.shape)
        weights = np.sum(mask * mins[:, np.newaxis, np.newaxis], axis=0) * (-1)
        weights[range(self.n_nodes), range(self.n_nodes)] = 0

        print(states[-1])
        graph = nx.from_numpy_array(weights)
        print(np.max(mins))
        print(np.round(mins[-1], 4), np.round(self.cut_value_for_planted(graph, states[-1]), 4))


        return graph

    @staticmethod
    def cut_value_for_planted(G, planted_vector):

        total = 0
        for u, v, w in G.edges(data='weight'):
            if planted_vector[u] != planted_vector[v]:
                total += w
        return total



if __name__ == '__main__':

    gen = MaxCut_definite_generator(
        number_of_nodes=10,
        density=1,
        number_of_mins=10,
        distribution="Normal")

    G = gen.generate_problem()

    qaoa = QAOA(graph=G,
                n_layers=5,
                logs=False,
                max_iter=1000,
                samples=1000,
                callback=True,
                initialization="no ones")
    qaoa.solve()
