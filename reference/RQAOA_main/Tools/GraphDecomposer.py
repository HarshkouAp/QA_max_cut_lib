import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Set, Dict, Tuple, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import heapq
import math
import itertools
from scipy import sparse
import warnings
from Tools.PartitionEfficiency import *
warnings.filterwarnings('ignore')

from Generators.GraphGenerator import ConnectedRandGraph
from Tools.PartitionEfficiency import *


from dataclasses import dataclass
from typing import Optional, Set, Dict, List
import random
import numpy as np
import networkx as nx
import time
from collections import defaultdict

@dataclass
class SubgraphConfig:
    """Основные параметры генератора."""
    max_size: int = 15
    min_size: int = 3
    num_subgraphs: int = 8
    alpha: float = 1.0     # вес связей с уже выбранными вершинами
    beta: float = 0.7      # вес связей внутри кандидатов
    gamma: float = 0.3     # вес локальной кластеризации
    delta: float = 0.2     # вес глобальных метрик (степень/треугольники)
    temperature: float = 0.5
    allow_overlap: bool = False
    max_overlap_ratio: float = 0.3
    ensure_connectivity: bool = True
    random_seed: Optional[int] = 42

class GraphUtils:
    """Минимальный набор вспомогательных функций."""
    @staticmethod
    def embeddedness(graph: nx.Graph, nodes: Set[int]) -> Dict[int, float]:
        """Доля соседей вершины, находящихся внутри множества nodes."""
        out: Dict[int, float] = {}
        for v in nodes:
            neigh = set(graph.neighbors(v))
            total = len(neigh)
            inside = len(neigh & nodes)
            out[v] = (inside / total) if total > 0 else 0.0
        return out

    @staticmethod
    def jaccard(a: Set[int], b: Set[int]) -> float:
        if not a and not b:
            return 1.0
        inter = len(a & b)
        uni = len(a | b)
        return inter / uni if uni > 0 else 0.0

class OptimizedDenseSubgraphGenerator:
    """
    Чистая реализация: генерация плотных подграфов.
    Основная идея: стартовая вершина -> итеративное добавление по рейтингу ->
    локальная пост-оптимизация (замены слабых вершин).
    """
    def __init__(self, graph: nx.Graph, config: Optional[SubgraphConfig] = None):
        self.G = graph.copy()
        self.config = config or SubgraphConfig()
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

        self._precompute()

    def _precompute(self) -> None:
        """Предварительные ускорители: соседство, степени, кластеризация, треугольники."""
        self.neighbors: Dict[int, Set[int]] = {n: set(self.G.neighbors(n)) for n in self.G.nodes()}
        self.degrees: Dict[int, int] = dict(self.G.degree())
        self.clustering: Dict[int, float] = nx.clustering(self.G)
        # быстрое приближение числа треугольников на вершину
        self.triangles: Dict[int, int] = nx.triangles(self.G)

        # предвычисленные максимумы для нормализации (чтобы не делать каждый раз)
        self._max_degree = max(self.degrees.values()) if self.degrees else 1
        self._max_triangles = max(self.triangles.values()) if self.triangles else 1

    def _avg_distance_to_selected(self, node: int, selected: Set[int]) -> float:
        """
        Среднее кратчайшее расстояние до множества selected.
        Для небольших selected вызываем nx.shortest_path_length многократно.
        При необходимости замените на многовыходный BFS — это ускорит большие selected.
        """
        if not selected:
            return 1.0
        dists = []
        for s in selected:
            if s == node:
                continue
            try:
                d = nx.shortest_path_length(self.G, node, s)
            except nx.NetworkXNoPath:
                d = len(self.G)  # большое значение при отсутствии пути
            dists.append(d)
        return float(np.mean(dists)) if dists else 1.0

    def _node_rating(self, node: int, selected: Set[int], candidates: Set[int]) -> float:
        """
        Составной рейтинг вершины:
         - r1: связи с уже выбранными (alpha)
         - r2: связи с кандидатами (beta)
         - r3: локальная кластеризация (gamma)
         - r4: глобальные метрики (delta): нормализованная степень + треугольники
         - r5: штраф/бонус за близость к выбранным (compactness)
        """
        neigh = self.neighbors.get(node, set())

        r1 = self.config.alpha * len(neigh & selected)
        r2 = self.config.beta * len(neigh & candidates)
        r3 = self.config.gamma * self.clustering.get(node, 0.0)

        deg_norm = (self.degrees.get(node, 0) / self._max_degree) if self._max_degree > 0 else 0.0
        tri_ratio = (self.triangles.get(node, 0) / self._max_triangles) if self._max_triangles > 0 else 0.0
        r4 = self.config.delta * (deg_norm + tri_ratio)

        r5 = 0.0
        if selected:
            avgd = self._avg_distance_to_selected(node, selected)
            r5 = 0.1 * (1.0 / (avgd + 1.0))

        return r1 + r2 + r3 + r4 + r5

    def _select_nodes_adaptive(self, ratings: Dict[int, float], k: int) -> Set[int]:
        """Выбор k вершин: топ-1 deterministically + остальные по softmax (temperature)."""
        if not ratings or k <= 0:
            return set()
        nodes = list(ratings.keys())
        vals = np.array([ratings[n] for n in nodes], dtype=float)

        if len(nodes) <= k:
            return set(nodes)

        # нормализация
        minv, maxv = float(vals.min()), float(vals.max())
        if maxv > minv:
            vals = (vals - minv) / (maxv - minv)
        else:
            vals = np.ones_like(vals)

        top_idx = int(np.argmax(vals))
        chosen = {nodes[top_idx]}
        remaining = [nodes[i] for i in range(len(nodes)) if i != top_idx]
        remaining_vals = np.array([vals[i] for i in range(len(nodes)) if i != top_idx], dtype=float)

        if self.config.temperature > 0:
            exps = np.exp(remaining_vals / self.config.temperature)
            probs = exps / exps.sum()
        else:
            probs = remaining_vals / remaining_vals.sum()

        num_needed = min(k - 1, len(remaining))
        if num_needed > 0:
            selected_indices = np.random.choice(len(remaining), size=num_needed, replace=False, p=probs)
            for idx in selected_indices:
                chosen.add(remaining[idx])

        return chosen

    def _select_optimal_start(self) -> int:
        """Стартовая вершина: по сумме нормализованных локальных и глобальных метрик."""
        best_node = None
        best_score = -float('inf')

        for v in self.G.nodes():
            deg_norm = self.degrees.get(v, 0) / self._max_degree if self._max_degree > 0 else 0.0
            tri_norm = self.triangles.get(v, 0) / self._max_triangles if self._max_triangles > 0 else 0.0
            cluster = self.clustering.get(v, 0.0)
            score = 0.4 * deg_norm + 0.3 * cluster + 0.2 * tri_norm + 0.1 * (deg_norm * tri_norm)
            if score > best_score:
                best_score = score
                best_node = v

        if best_node is None:
            best_node = random.choice(list(self.G.nodes()))
        return best_node

    def _optimize_local(self, nodes: Set[int]) -> Set[int]:
        """
        Локальная пост-оптимизация: пытаемся заменить несколько наименее встроенных вершин
        на их внешних соседей, если это повышает связность с подграфом.
        """
        if len(nodes) <= self.config.min_size:
            return nodes

        emb = GraphUtils.embeddedness(self.G, nodes)
        # возьмём несколько наименее встроенных
        sorted_by_emb = sorted(nodes, key=lambda x: emb.get(x, 0.0))
        for node_to_remove in sorted_by_emb[:3]:
            neighbors = self.neighbors.get(node_to_remove, set())
            candidates = neighbors - nodes
            if not candidates:
                continue
            best_candidate = None
            best_connections = -1
            for cand in candidates:
                connections = len(self.neighbors.get(cand, set()) & (nodes - {node_to_remove}))
                if connections > best_connections:
                    best_connections = connections
                    best_candidate = cand
            # простая замена, если кандидат явно лучше
            if best_candidate and best_connections > emb.get(node_to_remove, 0.0) * len(nodes):
                nodes.remove(node_to_remove)
                nodes.add(best_candidate)
        return nodes

    def _ensure_connectivity(self, sub: nx.Graph, start_node: int) -> nx.Graph:
        """Возвращает компоненту, содержащую start_node (или крупнейшую), чтобы гарантировать связь."""
        if nx.is_connected(sub):
            return sub
        comps = list(nx.connected_components(sub))
        for comp in comps:
            if start_node in comp:
                return self.G.subgraph(comp).copy()
        largest = max(comps, key=len)
        return self.G.subgraph(largest).copy()

    def generate_dense_subgraph(self, start_node: Optional[int] = None) -> nx.Graph:
        """Сгенерировать один плотный подграф."""
        if start_node is None:
            start_node = self._select_optimal_start()

        selected: Set[int] = {start_node}
        boundary: Set[int] = set(self.neighbors.get(start_node, set())) - selected
        iterations = 0
        max_iter = self.config.max_size * 3

        while len(selected) < self.config.max_size and boundary and iterations < max_iter:
            iterations += 1
            ratings = {v: self._node_rating(v, selected, boundary) for v in boundary}
            num_to_add = min(self.config.max_size - len(selected), max(1, len(boundary) // 2))
            new_nodes = self._select_nodes_adaptive(ratings, num_to_add)
            if not new_nodes:
                break
            selected.update(new_nodes)
            # обновляем границу: соседи новых - уже выбранные
            for v in new_nodes:
                boundary.update(self.neighbors.get(v, set()) - selected)
            # удаляем добавленные из границы
            boundary.difference_update(new_nodes)

        # пост-оптимизация
        if len(selected) >= self.config.min_size:
            selected = self._optimize_local(selected)

        sub = self.G.subgraph(selected).copy()

        if self.config.ensure_connectivity and not nx.is_connected(sub):
            sub = self._ensure_connectivity(sub, start_node)

        return sub

    def generate_multiple_subgraphs(self) -> List[nx.Graph]:
        """Генерация серии подграфов с контролем перекрытия."""
        subgraphs: List[nx.Graph] = []
        used_nodes: Set[int] = set()
        attempts = 0
        max_attempts = max(100, self.config.num_subgraphs * 10)

        while len(subgraphs) < self.config.num_subgraphs and attempts < max_attempts:
            attempts += 1
            if not self.config.allow_overlap and used_nodes:
                available = list(set(self.G.nodes()) - used_nodes)
                if not available:
                    break
                start_node = random.choice(available)
            else:
                start_node = None

            sg = self.generate_dense_subgraph(start_node)
            if sg.number_of_nodes() < self.config.min_size:
                continue

            if not self.config.allow_overlap:
                sg_nodes = set(sg.nodes())
                overlap_ratio = (len(sg_nodes & used_nodes) / len(sg_nodes)) if sg_nodes else 0.0
                if overlap_ratio > self.config.max_overlap_ratio:
                    continue
                used_nodes.update(sg_nodes)

            subgraphs.append(sg)

        return subgraphs



if __name__ == "__main__":


    G = ConnectedRandGraph(number_of_nodes=100,
                           number_of_edges=3000,
                           weights='ones',
                           seed=42)

    start = time.time()
    config = SubgraphConfig(
        max_size=10,
        num_subgraphs=50,
        random_seed=42
    )
    generator = OptimizedDenseSubgraphGenerator(G, config)
    subgraphs = generator.generate_multiple_subgraphs()
    part_time = time.time() - start
    print(f'Part time: {part_time}')

    covered = set().union(*[set(sg.nodes()) for sg in subgraphs]) if subgraphs else set()
    uncovered = set(G.nodes()) - covered
    print(f"Generated {len(subgraphs)} subgraphs")
    print(f"Covered vertices: {len(covered)} / {G.number_of_nodes()}")
    print(f"Uncovered vertices: {len(uncovered)}")

    total_edges = 0
    for i, sg in enumerate(subgraphs, 1):
        print(f"SG{i}: nodes={sg.number_of_nodes()}, edges={sg.number_of_edges()}, density={nx.density(sg):.4f}")
        total_edges += sg.number_of_edges()

    print(total_edges)

    pe = PartitionEfficiency(G, subgraphs)
    pe.efficiency(sample_threshold=200, sample_pairs=800, random_seed=1)
    pe.plot_statistic(bins=25, figsize=(14, 10), title="Demo partition statistics")