import cvxpy as cp
import scipy
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

from Generators.GraphGenerator import ConnectedRandGraph, DensRandGraph


class GoemansWilliamsonMaxcut:

    def __init__(self, graph, no_hyperplane_samples = 10, callback=False):
        self.graph = graph
        self.no_hyperplane_samples = no_hyperplane_samples
        self.callback = callback

        self.nodes = list(graph.nodes())
        self.edges = list(graph.edges(data=True))  # Include edge weights
        self.number_of_nodes = graph.number_of_nodes()
        self.number_of_edges = graph.number_of_edges()

        self.all_cut_values = []
        self.best_solution = 0.
        self.best_cut_value = 0.
        self.energy = 0.
        self.time =  time.time()


    def calculate_cut_value(self, partition_1, partition_2):
        cut_value = 0
        for u, v, data in self.graph.edges(data=True):
            if (u in partition_1 and v in partition_2) or (u in partition_2 and v in partition_1):
                cut_value += data.get('weight', 1)
        return cut_value


    def solve(self):

        X = cp.Variable((self.number_of_nodes, self.number_of_nodes), symmetric=True)

        # Define the constraints
        constraints = [X >> 0]
        constraints += [X[i, i] == 1 for i in range(self.number_of_nodes)]

        # Define the objective function
        objective = cp.Maximize(sum(0.5 * (1 - X[i, j]) * data.get('weight', 1) for i, j, data in self.edges))

        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve()

        matrix_of_vectors = scipy.linalg.sqrtm(X.value)  # Symmetric positive semidefinite matrix

        for i in range(self.no_hyperplane_samples):
            # Recover a solution using randomized rounding
            u = np.random.randn(self.number_of_nodes)  # Random normal vector
            # output_vec = np.sign(matrix_of_vectors @ u)
            # output_vec = np.real(output_vec).astype(int)
            output_vec = matrix_of_vectors @ u
            output_vec = np.real(output_vec)
            output_vec = np.sign(output_vec)
            output_vec = output_vec.astype('int')


            # Partition the nodes
            partition_1 = np.where(output_vec == 1)[0]
            partition_2 = np.where(output_vec == -1)[0]

            cut_value = self.calculate_cut_value(partition_1, partition_2)
            self.all_cut_values.append(cut_value)

            if cut_value > self.best_cut_value:
                self.best_cut_value = cut_value
                self.best_solution = (partition_1, partition_2)

        self.energy = self.best_cut_value
        self.time = time.time() - self.time
        self._callback()


    def _callback(self):

        if self.callback:

            print("\033[33m {}".format(f"==" * 30))
            print("\033[33m {}".format("***SUCCESS***"))
            print(f"Graph       ||   number of nodes : {self.graph.number_of_nodes()}  "
                  f"number of edges : {self.graph.number_of_edges()} ")
            print(f"GW          ||   energy : {np.round(self.energy, 4)}   "
                  f"hyperplanes : {self.no_hyperplane_samples} ")
            print(f"Processing  ||   time : {np.round(self.time, 4)} s")
            print("\033[33m {}".format(f"==" * 30))



if __name__ == '__main__':

    G = DensRandGraph(
        number_of_nodes=100,
        density=0.7,
        weights="ones",
        seed=1000,
    )

    gw = GoemansWilliamsonMaxcut(graph=G,
                                 no_hyperplane_samples=5,
                                 callback=True)
    gw.solve()



