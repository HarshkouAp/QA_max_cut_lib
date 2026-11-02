import random
import networkx as nx

FAMILIES = [
    "er_sparse",
    "er_dense",
    "random_regular",
    "barabasi_albert",
    "watts_strogatz",
    "sbm",
    "planted_cut",
    "random_geometric",
    "toroidal_grid",
    "adversarial_mixed",
    "complete",
]

SIZES = [100, 100, 100, 200, 200, 200, 300, 300, 300]
WEIGHT_TYPES = ["unweighted", "positive_continuous", "signed_discrete"]
WEIGHT_SEED_OFFSETS = {
    "unweighted": 101,
    "positive_continuous": 202,
    "signed_discrete": 303,
}


def get_seed_info(seed: int, base_seed: int = 42) -> dict:
    """
    Расшифровывает benchmark seed из диапазона 0..296.

    seed здесь является ID графа, а не напрямую random seed.
    Каждые три последовательных seed соответствуют одной топологии
    с тремя типами весов:
        0 -> unweighted
        1 -> positive_continuous
        2 -> signed_discrete
    """

    if not isinstance(seed, int):
        raise TypeError("seed должен быть целым числом.")

    max_seed = len(FAMILIES) * len(SIZES) * len(WEIGHT_TYPES) - 1

    if seed < 0 or seed > max_seed:
        raise ValueError(f"seed должен лежать в диапазоне 0..{max_seed}.")

    topology_id = seed // len(WEIGHT_TYPES)
    weight_idx = seed % len(WEIGHT_TYPES)

    family_idx = topology_id // len(SIZES)
    local_id = topology_id % len(SIZES)

    family = FAMILIES[family_idx]
    N = SIZES[local_id]
    size_rep = local_id % 3
    weight_type = WEIGHT_TYPES[weight_idx]

    topology_seed = base_seed + 10_000 * topology_id
    weight_seed = topology_seed + WEIGHT_SEED_OFFSETS[weight_type]

    name = f"{seed:03d}_{family}_N{N}_rep{size_rep}_{weight_type}"

    return {
        "seed": seed,
        "name": name,
        "family": family,
        "N": N,
        "size_rep": size_rep,
        "local_id_within_family": local_id,
        "topology_id": topology_id,
        "weight_type": weight_type,
        "topology_seed": topology_seed,
        "weight_seed": weight_seed,
    }


def assign_weights(G: nx.Graph, weight_type: str, weight_seed: int) -> nx.Graph:
    rng = random.Random(weight_seed)
    H = G.copy()

    if weight_type == "unweighted":
        for u, v in H.edges():
            H[u][v]["weight"] = 1.0

    elif weight_type == "positive_continuous":
        for u, v in H.edges():
            H[u][v]["weight"] = rng.random()

    elif weight_type == "signed_discrete":
        values = list(range(-10, 0)) + list(range(1, 11))

        for u, v in H.edges():
            H[u][v]["weight"] = float(rng.choice(values))

    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")

    return H


def generate_connected(generator, max_tries: int = 100) -> nx.Graph:
    for _ in range(max_tries):
        G = generator()

        if nx.is_connected(G):
            return G

    raise RuntimeError("Не удалось сгенерировать связный граф.")


def generate_er_sparse(N: int, seed: int) -> nx.Graph:
    rng = random.Random(seed)
    p = rng.uniform(6 / N, 12 / N)

    return generate_connected(
        lambda: nx.erdos_renyi_graph(N, p, seed=rng.randint(0, 10**9))
    )


def generate_er_dense(N: int, seed: int) -> nx.Graph:
    rng = random.Random(seed)
    p = rng.uniform(0.15, 0.45)

    return generate_connected(
        lambda: nx.erdos_renyi_graph(N, p, seed=rng.randint(0, 10**9))
    )


def generate_random_regular(N: int, seed: int) -> nx.Graph:
    rng = random.Random(seed)
    possible_degrees = [3, 4, 5, 8, 10, 12]
    degrees = [d for d in possible_degrees if N * d % 2 == 0 and d < N]
    d = rng.choice(degrees)

    return generate_connected(
        lambda: nx.random_regular_graph(d, N, seed=rng.randint(0, 10**9))
    )


def generate_barabasi_albert(N: int, seed: int) -> nx.Graph:
    rng = random.Random(seed)
    m = rng.choice([2, 3, 4, 5, 6])

    return nx.barabasi_albert_graph(N, m, seed=seed)


def generate_watts_strogatz(N: int, seed: int) -> nx.Graph:
    rng = random.Random(seed)
    k = rng.choice([4, 6, 8, 10])
    beta = rng.uniform(0.05, 0.35)

    return nx.connected_watts_strogatz_graph(
        n=N,
        k=k,
        p=beta,
        tries=100,
        seed=seed,
    )


def generate_sbm(N: int, seed: int) -> nx.Graph:
    rng = random.Random(seed)
    n_blocks = rng.choice([3, 4, 5])

    base = N // n_blocks
    sizes = [base] * n_blocks
    sizes[-1] += N - sum(sizes)

    p_in = rng.uniform(0.08, 0.20)
    p_out = rng.uniform(0.01, 0.05)

    probs = [
        [p_in if i == j else p_out for j in range(n_blocks)]
        for i in range(n_blocks)
    ]

    return generate_connected(
        lambda: nx.stochastic_block_model(
            sizes,
            probs,
            seed=rng.randint(0, 10**9),
        )
    )


def generate_planted_cut(N: int, seed: int) -> nx.Graph:
    rng = random.Random(seed)

    n1 = N // 2
    n2 = N - n1

    p_between = rng.uniform(0.10, 0.30)
    p_inside = rng.uniform(0.01, 0.06)

    G = nx.Graph()
    G.add_nodes_from(range(N))

    left = list(range(n1))
    right = list(range(n1, N))

    for i, u in enumerate(left):
        for v in left[i + 1:]:
            if rng.random() < p_inside:
                G.add_edge(u, v)

    for i, u in enumerate(right):
        for v in right[i + 1:]:
            if rng.random() < p_inside:
                G.add_edge(u, v)

    for u in left:
        for v in right:
            if rng.random() < p_between:
                G.add_edge(u, v)

    if nx.is_connected(G):
        return G

    components = [list(c) for c in nx.connected_components(G)]

    for c1, c2 in zip(components[:-1], components[1:]):
        u = rng.choice(c1)
        v = rng.choice(c2)
        G.add_edge(u, v)

    return G


def generate_random_geometric(N: int, seed: int) -> nx.Graph:
    rng = random.Random(seed)

    if N == 100:
        radius_range = (0.16, 0.23)
    elif N == 200:
        radius_range = (0.12, 0.18)
    elif N == 300:
        radius_range = (0.10, 0.15)
    else:
        raise ValueError("random_geometric configured only for N in {100, 200, 300}.")

    radius = rng.uniform(*radius_range)

    return generate_connected(
        lambda: nx.random_geometric_graph(
            N,
            radius,
            seed=rng.randint(0, 10**9),
        )
    )


def generate_toroidal_grid_like(N: int, seed: int) -> nx.Graph:
    if N == 100:
        rows, cols = 10, 10
    elif N == 200:
        rows, cols = 10, 20
    elif N == 300:
        rows, cols = 15, 20
    else:
        raise ValueError("Toroidal grid configured only for N in {100, 200, 300}.")

    G = nx.grid_2d_graph(rows, cols, periodic=True)
    mapping = {node: idx for idx, node in enumerate(G.nodes())}

    return nx.relabel_nodes(G, mapping)


def generate_adversarial_mixed(N: int, seed: int) -> nx.Graph:
    rng = random.Random(seed)
    G = generate_sbm(N, seed)
    nodes = list(G.nodes())

    n_hubs = max(2, N // 50)
    hubs = rng.sample(nodes, n_hubs)

    for hub in hubs:
        targets = rng.sample(nodes, k=max(5, N // 5))

        for v in targets:
            if v != hub:
                G.add_edge(hub, v)

    for _ in range(N):
        u, v = rng.sample(nodes, 2)
        G.add_edge(u, v)

    return G


def generate_complete(N: int, seed: int) -> nx.Graph:
    return nx.complete_graph(N)


def generate_base_graph(family: str, N: int, topology_seed: int) -> nx.Graph:
    if family == "er_sparse":
        return generate_er_sparse(N, topology_seed)

    if family == "er_dense":
        return generate_er_dense(N, topology_seed)

    if family == "random_regular":
        return generate_random_regular(N, topology_seed)

    if family == "barabasi_albert":
        return generate_barabasi_albert(N, topology_seed)

    if family == "watts_strogatz":
        return generate_watts_strogatz(N, topology_seed)

    if family == "sbm":
        return generate_sbm(N, topology_seed)

    if family == "planted_cut":
        return generate_planted_cut(N, topology_seed)

    if family == "random_geometric":
        return generate_random_geometric(N, topology_seed)

    if family == "toroidal_grid":
        return generate_toroidal_grid_like(N, topology_seed)

    if family == "adversarial_mixed":
        return generate_adversarial_mixed(N, topology_seed)

    if family == "complete":
        return generate_complete(N, topology_seed)

    raise ValueError(f"Unknown graph family: {family}")


def get_benchmark_graph(seed: int, base_seed: int = 42, return_info: bool = False):
    """
    Возвращает benchmark-граф по ID seed из диапазона 0..296.

    Пример:
        G = get_benchmark_graph(17)
        G, info = get_benchmark_graph(17, return_info=True)

    Вес ребра хранится в атрибуте "weight".
    """

    info = get_seed_info(seed, base_seed=base_seed)

    G_base = generate_base_graph(
        family=info["family"],
        N=info["N"],
        topology_seed=info["topology_seed"],
    )

    G = assign_weights(
        G_base,
        weight_type=info["weight_type"],
        weight_seed=info["weight_seed"],
    )

    G.graph.update(info)

    if return_info:
        return G, info

    return G


if __name__ == "__main__":

    G, info = get_benchmark_graph(17, return_info=True)

    print(info)
