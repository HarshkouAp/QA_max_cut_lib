from Laboratory.NodeExtender import *
from Laboratory.EdgeExtender import *
from Generators.GraphGenerator import ConnectedRandGraph
from Tools.PartitionEfficiency import PartitionEfficiency
import matplotlib.pyplot as plt
import numpy as np
import time


class GraphScanner:

    def __init__(self, graph, max_subgraphs_size=10,
                 coverage=0.7, max_number_of_subgraphs=50,
                 callback=False):


        self.graph = graph
        self.max_subgraphs_size = max_subgraphs_size
        self.coverage = coverage
        self.max_number_of_subgraphs = max_number_of_subgraphs
        self.callback = callback

        self.n_nodes = self.graph.number_of_nodes()
        self.n_edges = self.graph.number_of_edges()
        self.subgraph_list = []
        self.uncovered_nodes = list(self.graph.nodes())
        self.uncovered_edges = list(self.graph.edges())
        self.covered_nodes = set({})
        self.covered_edges = set({})

        self.processing_time = 0



    def solve(self):

        start_time = time.time()

        while (((len(self.covered_edges) / self.n_edges) < self.coverage) &
               (len(self.subgraph_list) < self.max_number_of_subgraphs)):

                start_edge = random.choice(self.uncovered_edges)
                subgraphs = generate_all_edge_subgraphs(G=self.graph,
                                                        edges=[start_edge],
                                                        max_size=self.max_subgraphs_size,
                                                        weight_attr='w')

                for subgraph in subgraphs:
                    self.subgraph_list.append(subgraph)
                    edges = set(subgraph.edges())
                    nodes = set(subgraph.nodes())
                    self.covered_edges |= edges
                    self.uncovered_edges = [edge  for edge in self.uncovered_edges if (edge not in edges)]
                    self.covered_nodes |= nodes
                    self.uncovered_nodes = [node for node in self.uncovered_nodes if (node not in nodes)]


        self.processing_time = time.time() - start_time
        self._callback()


    def _callback(self):

        if self.callback:

            print("\033[33m {}".format(f"==" * 30))
            print(f"Graph       ||   {self.n_nodes} nodes   {self.n_edges} edges")
            print(f"Coverage    ||   {100 * (len(self.covered_nodes) / self.n_nodes):.2f}% nodes   "
                  f"{100 * (len(self.covered_edges) / self.n_edges):.2f}% edges")
            print(f"Generated {len(self.subgraph_list)} subgraphs")
            print(f"Proccessing time: {self.processing_time:.3f} seconds")
            print("\033[33m {}".format(f"==" * 30))









if __name__ == '__main__':

    G = ConnectedRandGraph(number_of_nodes=50,
                           number_of_edges=400,
                           weights='ones',
                           seed=1000)

    gs = GraphScanner(graph=G,
                      max_subgraphs_size=16,
                      coverage=0.9,
                      max_number_of_subgraphs=200,
                      callback=True)

    gs.solve()
    subgraphs = gs.subgraph_list

    pe = PartitionEfficiency(G, subgraphs)
    pe.efficiency(sample_threshold=200, sample_pairs=800, random_seed=1)
    pe.plot_statistic(bins=50, figsize=(14, 10), title="Demo partition statistics")
