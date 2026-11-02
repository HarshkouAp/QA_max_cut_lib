import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def generate_connected_graph(num_nodes, num_edges, seed):

    graph = nx.path_graph(num_nodes)
    edges = num_nodes - 1
    iteration = 0

    np.random.seed(seed)
    j_nodes = np.random.randint(0, num_nodes, 10000)
    np.random.seed(seed + 100)
    k_nodes = np.random.randint(0, num_nodes, 10000)

    while edges < num_edges:

        if ((j_nodes[iteration] not in nx.neighbors(graph, k_nodes[iteration])) &
                (j_nodes[iteration] != k_nodes[iteration])):
            graph.add_edge(j_nodes[iteration], k_nodes[iteration])
            edges += 1

        iteration += 1

    return graph


def calculate_nonlocality(G):

    n_nodes = G.number_of_nodes()
    sp_map = nx.floyd_warshall_numpy(G)
    nonlocality = 2 * sp_map.sum() / (n_nodes * (n_nodes - 1))
    return nonlocality


def generate_graphs_and_measure_nonlocality(N, V, output_file="graph_data.csv"):

    min_edges = N - 1
    max_edges = N * (N - 1) // 2

    print(f"Генерация графов с {N} вершинами")
    print(f"Минимальное количество рёбер: {min_edges}")
    print(f"Максимальное количество рёбер: {max_edges}")
    print(f"Количество вариаций для каждого числа рёбер: {V}")

    data = []
    total_graphs = (max_edges - min_edges + 1) * V
    pbar = tqdm(total=total_graphs, desc="Генерация графов")

    base_seed = 42

    for num_edges in range(min_edges, max_edges + 1):
        for v in range(V):
            current_seed = base_seed + num_edges * 1000 + v * 100

            G = generate_connected_graph(N, num_edges, seed=current_seed)

            if nx.is_connected(G):
                nonlocality = calculate_nonlocality(G)
                data.append({
                    'N_nodes': N,
                    'N_edges': num_edges,
                    'Seed': current_seed,
                    'Nonlocality': nonlocality,
                    'Variation': v + 1})

            else:
                print(f"Не удалось сгенерировать связный граф с {num_edges} рёбрами (попытка {v + 1})")

            pbar.update(1)
    pbar.close()

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

    print(f"\nДанные сохранены в файл: {output_file}")
    print(f"Всего сгенерировано графов: {len(df)}")
    print("\nСтатистика по нелокальности:")
    print(f"Минимальная нелокальность: {df['Nonlocality'].min():.4f}")
    print(f"Максимальная нелокальность: {df['Nonlocality'].max():.4f}")
    print(f"Средняя нелокальность: {df['Nonlocality'].mean():.4f}")

    return df


def visualize_results(df):

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='N_edges', y='Nonlocality', alpha=0.6)

    trend_data = df.groupby('N_edges')['Nonlocality'].mean().reset_index()
    sns.lineplot(data=trend_data, x='N_edges', y='Nonlocality',
                 color='red', linewidth=2, label='Средняя нелокальность')

    plt.title(f'Нелокальность графов с {df["N_nodes"].iloc[0]} вершинами', fontsize=16)
    plt.xlabel('Количество рёбер', fontsize=12)
    plt.ylabel('Мера нелокальности', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":


    df = generate_graphs_and_measure_nonlocality(12, 50, "../Data/test_6/connected_12.csv")
    visualize_results(df)
    correlation = df['N_edges'].corr(df['Nonlocality'])
    print(f"\nКорреляция между количеством рёбер и нелокальностью: {correlation:.4f}")