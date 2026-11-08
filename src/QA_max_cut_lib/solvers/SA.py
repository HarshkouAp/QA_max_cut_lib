import numpy as np
import dimod
import time
import warnings
warnings.filterwarnings("ignore")

from dwave.samplers import SimulatedAnnealingSampler
from Generators.GraphGenerator import ConnectedRandGraph



class DwaveAnneal:

    def __init__(self, graph, num_reads=200, callback=False):

        self.graph = graph
        self.num_reads = num_reads
        self.callback = callback

        self.time = time.time()
        self.energy = 0.


    def _build_maxcut_bqm(self):

        h = {}
        J = {}

        for u, v, data in self.graph.edges(data=True):
            w = float(data.get("weight", 1.0))
            J[(u, v)] = -0.5 * w

        return dimod.BinaryQuadraticModel.from_ising(h, J)


    @staticmethod
    def _best_sample(sampleset: dimod.SampleSet):

        rec = sampleset.record
        idx = rec["energy"].argmin()
        energy = float(rec["energy"][idx])
        sample_array = rec["sample"][idx]
        variables = list(sampleset.variables)
        sample = {variables[i]: int(sample_array[i]) for i in range(len(variables))}

        return energy, sample


    @staticmethod
    def _normalize_spins(sample):

        vals = set(sample.values())
        if vals <= {0, 1}:
            return {k: (1 if v == 1 else -1) for k, v in sample.items()}

        return sample


    def _compute_cut_value(self, spins):

        cut = 0.0
        for u, v, data in self.graph.edges(data=True):
            w = float(data.get("weight", 1.0))
            if spins[u] != spins[v]:
                cut += w
        return cut


    def _solve_with_dwave_sa(self, bqm):
        sampler = SimulatedAnnealingSampler()
        return sampler.sample(bqm, num_reads=self.num_reads)


    def solve(self):

        bqm = self._build_maxcut_bqm()

        ss_dwave = self._solve_with_dwave_sa(bqm)
        self.energy, sample_dwave = self._best_sample(ss_dwave)

        self.time = time.time() - self.time
        self._callback()

    def _callback(self):

        if self.callback:

            print("\033[33m {}".format(f"==" * 30))
            print("\033[33m {}".format("***SUCCESS***"))
            print(f"Graph       ||   number of nodes : {self.graph.number_of_nodes()}  "
                  f"number of edges : {self.graph.number_of_edges()} ")
            print(f"Dwave       ||   energy : {np.round(self.energy, 4)}   "
                  f"number of samples : {self.num_reads} ")
            print(f"Processing  ||   time : {np.round(self.time, 4)} s")
            print("\033[33m {}".format(f"==" * 30))



if __name__ == "__main__":

    G = ConnectedRandGraph(
        number_of_nodes=100,
        number_of_edges=3500,
        weights="random",
        seed=1234
    )

    da = DwaveAnneal(graph=G,
                     num_reads=1000,
                     callback=True)
    da.solve()