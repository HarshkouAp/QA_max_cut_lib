import networkx as nx
import numpy as np
import time

class mod:

    def __init__(self, graph):
        self.graph = graph
        self.mod = 0

    def solve(self):

        self.mod = np.sum(np.abs(nx.to_numpy_array(self.graph))) / 2
        time.sleep(0.1)