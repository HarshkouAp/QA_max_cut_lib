from Solvers.LS_RQAOA import *
from Laboratory.EdgeExtender import *
from Solvers.FineTuning import FineTuning


class Test:

    def __init__(self, graph):
        self.graph = graph
        self.energy = 0



    def solve(self):

        lsq = LargeScaleRQAOA(graph=self.graph,
                              max_size=10,
                              part_iter=1,
                              treshold=3,
                              n_layers=1,
                              w_scale=1,
                              logs=False,
                              callback=False,
                              max_iter=500,
                              random_seed=1000)
        lsq.solve()


        ft = FineTuning(graph=self.graph,
                        solution=lsq.lsr_sol,
                        corr_dict=lsq.max_corr_dict,
                        max_size=10,
                        number_of_edges=-1)
        ft.solve()
        self.energy = ft.c_energy



if __name__ == '__main__':

    G = ConnectedRandGraph(number_of_nodes=50,
                           number_of_edges=600,
                           weights='random',
                           seed=100)

    ft = Test(G)
    ft.solve()
    print(ft.energy)