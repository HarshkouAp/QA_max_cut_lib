import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import defaultdict
from math import comb

from Tools.Graph_part_interception import *
from Tools.GraphDecomposer import *

sns.set_style("whitegrid")
plt.rcParams.update({'figure.autolayout': True})


class PartitionEfficiency:
    def __init__(self, graph, partition):
        """
        graph: networkx.Graph (полный граф)
        partition: iterable of networkx.Graph (подграфы, каждое содержит только свои рёбра)
        """
        self.graph = graph
        self.partition = list(partition)

        self.graph_nodes = self.graph.number_of_nodes()
        self.graph_edges = self.graph.number_of_edges()

        # глобальные свойства графа
        # clustering: corrected for networkx API differences (nx.clustering returns dict)
        clustering_vals = list(nx.clustering(self.graph).values()) if self.graph_nodes > 0 else [0.0]
        self.graph_clustering = float(np.mean(clustering_vals))
        self.graph_density = self._density(self.graph)
        self.regularity = (self.graph_edges / self.graph_nodes) if self.graph_nodes > 0 else 0.0

        # per-partition statistics (filled by efficiency())
        self.no_nodes_list = []
        self.no_edges_list = []
        self.clustering_list = []
        self.density_list = []
        self.nonlocality_list = []
        self.regularity_list = []

        # summary stats (scalars)
        self.summary = {}


    @staticmethod
    def _density(graph):
        n_n = graph.number_of_nodes()
        if n_n < 2:
            return 0.0
        n_e = graph.number_of_edges()
        return 2.0 * n_e / (n_n * (n_n - 1))


    @staticmethod
    def _nonlocality(graph):

        n_nodes = graph.number_of_nodes()
        sp_map = nx.floyd_warshall_numpy(graph)
        nonlocality = 2 * sp_map.sum() / (n_nodes * (n_nodes - 1))
        return nonlocality


    def efficiency(self, sample_threshold=300, sample_pairs=1000, random_seed=0):
        """
        Compute per-subgraph metrics and summary statistics.
        sample_threshold/sample_pairs govern the approximation of nonlocality for large subgraphs.
        """
        rnd = random.Random(random_seed)

        no_nodes_list = []
        no_edges_list = []
        density_list = []
        nonlocality_list = []
        clustering_list = []
        regularity_list = []

        for subgraph in self.partition:
            n_nodes = subgraph.number_of_nodes()
            n_edges = subgraph.number_of_edges()
            no_nodes_list.append(n_nodes)
            no_edges_list.append(n_edges)

            # density
            density_list.append(self._density(subgraph))

            # nonlocality (approx for big subgraphs)
            nonlocality_list.append(self._nonlocality(subgraph))

            # clustering (average local clustering)
            if n_nodes > 0:
                cvals = list(nx.clustering(subgraph).values())
                clustering_list.append(float(np.mean(cvals)) if len(cvals) > 0 else 0.0)
            else:
                clustering_list.append(0.0)

            # regularity: nodes / edges (if edges=0 -> set to NaN)
            regularity_list.append((n_nodes / n_edges) if n_edges > 0 else np.nan)

        # store lists
        self.no_nodes_list = np.array(no_nodes_list, dtype=float)
        self.no_edges_list = np.array(no_edges_list, dtype=float)
        self.density_list = np.array(density_list, dtype=float)
        self.nonlocality_list = np.array(nonlocality_list, dtype=float)
        self.clustering_list = np.array(clustering_list, dtype=float)
        self.regularity_list = np.array(regularity_list, dtype=float)

        # compute summary stats (mean, std, min, max) for each metric and store in dict
        def summarize(arr):
            arr_clean = arr[~np.isnan(arr)]
            if arr_clean.size == 0:
                return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
            return {
                'mean': float(np.mean(arr_clean)),
                'std': float(np.std(arr_clean, ddof=0)),
                'min': float(np.min(arr_clean)),
                'max': float(np.max(arr_clean))
            }

        self.summary = {
            'nodes': summarize(self.no_nodes_list),
            'edges': summarize(self.no_edges_list),
            'clustering': summarize(self.clustering_list),
            'density': summarize(self.density_list),
            'nonlocality': summarize(self.nonlocality_list),
            'regularity': summarize(self.regularity_list)
        }

        return self.summary

    def plot_statistic(self, bins=30, figsize=(15, 12), palette='deep', title=None, show=True):
        """
        Build 6 histograms (nodes, edges, clustering, density, nonlocality, regularity).
        Each plot marks: mean (solid), ±std (dashed), min and max (dotted).
        """
        # Ensure metrics computed
        if not self.summary:
            raise RuntimeError("Call efficiency() before plot_statistic().")

        metrics = [
            ('nodes', self.no_nodes_list, 'Number of nodes'),
            ('edges', self.no_edges_list, 'Number of edges'),
            ('clustering', self.clustering_list, 'Average clustering'),
            ('density', self.density_list, 'Density'),
            ('nonlocality', self.nonlocality_list, 'Nonlocality'),
            ('regularity', self.regularity_list, 'Nodes / Edges')
        ]

        sns.set_palette(palette)
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        axes = axes.flatten()
        for ax, (key, arr, xlabel) in zip(axes, metrics):
            arr_clean = arr[~np.isnan(arr)]
            if arr_clean.size == 0:
                ax.text(0.5, 0.5, "No data", ha='center', va='center')
                ax.set_title(xlabel)
                continue

            # histogram + KDE
            sns.histplot(arr_clean, bins=bins, kde=True, ax=ax, stat='count', edgecolor='k', linewidth=0.3)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Count')
            ax.set_title(f"{xlabel} — distribution")

            stats = self.summary[key]
            mean = stats['mean']
            std = stats['std']
            mn = stats['min']
            mx = stats['max']

            # vertical lines: mean (solid), mean ± std (dashed), min/max (dotted)
            ax.axvline(mean, color='tab:orange', linestyle='-', linewidth=2, label=f"mean = {mean:.3g}")
            if not np.isnan(std) and std > 0:
                ax.axvline(mean - std, color='tab:green', linestyle='--', linewidth=1.5, label=f"mean - std = {mean - std:.3g}")
                ax.axvline(mean + std, color='tab:green', linestyle='--', linewidth=1.5, label=f"mean + std = {mean + std:.3g}")
            ax.axvline(mn, color='tab:red', linestyle=':', linewidth=1.2, label=f"min = {mn:.3g}")
            ax.axvline(mx, color='tab:red', linestyle=':', linewidth=1.2, label=f"max = {mx:.3g}")

            # small stats box in top-right
            textstr = '\n'.join((
                f"mean: {mean:.4g}",
                f"std:  {std:.4g}",
                f"min:  {mn:.4g}",
                f"max:  {mx:.4g}",
                f"count: {arr_clean.size}"
            ))
            props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
            ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right', bbox=props)

            # legend (one entry) — avoid duplicate legend entries by creating proxy artists
            # but use ax.legend to show labels of vertical lines
            ax.legend(loc='upper left', fontsize=8)

        if title is None:
            title = "Partition statistics"
        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        if show:
            plt.show()
        return fig


# ---------------------------
# Additional metrics (suggestions)
# ---------------------------
#
# Можно дополнительно вычислять и визуализировать:
# 1) edge_weight_sum: суммарный вес внутренних рёбер в подгруппе — отражает «весовую плотность».
# 2) coverage: доля рёбер исходного графа, покрытая суммой всех подграфов (проверка корректности).
# 3) overlap_distribution: распределение числа кластеров на вершину (how many clusters a vertex participates).
# 4) conductance или normalized cut (если рассматривать подгруппы относительно внешних рёбер).
# 5) internal/external edge ratio: отношение внутренних рёбер к сумме рёбер, инцидентных вершинам подгруппы.
# 6) modularity-like score per subgraph w.r.t. original graph (если нужно сравнивать качество).
#
# Метрики 1,3,5 особенно полезны для оценки компромисса плотности <-> перекрытия.
# Метрики 2 и 4 — для глобальной валидации покрытия и «очистки» краёв.


# ---------------------------
# Example usage (synthetic)
# ---------------------------
if __name__ == "__main__":


    G = ConnectedRandGraph(number_of_nodes=1000,
                           number_of_edges=300000,
                           weights='ones')

    subgraphs = partition_graph_max_overlap_fast(G, M=10, min_size=8,
                                                 seed_strategy='max_overlap', random_state=42,
                                                 w_internal=1.0, w_overlap=2.0)
    pe = PartitionEfficiency(G, subgraphs)
    pe.efficiency(sample_threshold=200, sample_pairs=800, random_seed=1)
    pe.plot_statistic(bins=25, figsize=(14, 10), title="Demo partition statistics")

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


