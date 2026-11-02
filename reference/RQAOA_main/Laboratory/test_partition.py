import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from Tools.Graph_part_interception import *
from Tools.GraphDecomposer import *
from Tools.PartitionEfficiency import *

sns.set_style("whitegrid")
plt.rcParams.update({'figure.autolayout': True})





if __name__ == "__main__":


    G = ConnectedRandGraph(number_of_nodes=1000,
                           number_of_edges=300000,
                           weights='ones')



    config = SubgraphConfig(
        max_size=10,
        num_subgraphs=100,
        selection_strategy="adaptive",
        expansion_policy="hybrid",
        allow_overlap=False,
        random_seed=42
    )

    generator = OptimizedDenseSubgraphGenerator(G, config)
    subgraphs = generator.generate_multiple_subgraphs()
    pe = PartitionEfficiency(G, subgraphs)

    pe.efficiency(sample_threshold=200, sample_pairs=800, random_seed=1)
    pe.plot_statistic(bins=25, figsize=(14, 10), title="Demo partition statistics")