"""Use independent small-system calculations as scientific references."""

from itertools import product
import json
import numpy as np
import pytest
from scipy.linalg import expm
from QA_max_cut_lib import WeightedGraph, ExactCPU, QAOACPU, GraphFactory
from QA_max_cut_lib.solvers.qaoa_cpu import cut_diagonal, state_at_angles
from QA_max_cut_lib.solvers.objectives import objective_value, validate_objective
from QA_max_cut_lib.tools import correlation_matrix, quality_score


@pytest.mark.parametrize("seed", range(5))
def test_exact_against_independent_assignments(seed):
    rng = np.random.default_rng(seed)
    nodes = [-9, 2, 7, 14, 28]
    edges = [
        (u, v, float(rng.integers(-4, 5)))
        for i, u in enumerate(nodes)
        for v in nodes[i + 1 :]
    ]
    graph = WeightedGraph(edges, nodes=nodes + [100])
    values = []
    for spins in product([-1, 1], repeat=len(graph)):
        mapping = dict(zip(graph.nodes, spins))
        values.append(sum(w for u, v, w in edges if mapping[u] != mapping[v]))
    solver = ExactCPU({"parameters": {"batch_size": 7}})
    solution = solver.solve(graph)
    assert graph.cut_value(solution) == max(values)
    assert solver.worst_cut_value == min(values)
    assert solver.optimal_count == values.count(max(values))
    assert set(solution) == set(graph.nodes)
    assert all(type(v) is int and v in (-1, 1) for v in solution.values())


def test_exact_empty_and_all_zero():
    solver = ExactCPU()
    assert solver.solve(WeightedGraph()) == {}
    assert solver.cut_value == 0 and solver.optimal_count == 1
    solver.solve(WeightedGraph(nodes=[2, 4, 100]))
    assert solver.optimal_count == 8
    assert quality_score(0, 0, 0) is None


@pytest.mark.parametrize("layers", [1, 2])
def test_qaoa_against_dense_matrix_exponential(layers):
    graph = WeightedGraph([(2, 9, -1.25), (2, 50, 2.0), (9, 50, 0.7)], nodes=[9, 50, 2])
    n = len(graph)
    size = 2**n
    costs = []
    for state in range(size):
        solution = {
            node: 1 - 2 * ((state >> i) & 1) for i, node in enumerate(graph.nodes)
        }
        costs.append(-sum(w for u, v, w in graph.edges if solution[u] != solution[v]))
    mixer = np.zeros((size, size))
    for state in range(size):
        for qubit in range(n):
            mixer[state ^ (1 << qubit), state] += 1
    angles = np.linspace(-0.7, 0.9, 2 * layers)
    reference = np.ones(size, dtype=complex) / np.sqrt(size)
    for layer in range(layers):
        reference = expm(-1j * angles[layer] * np.diag(costs)) @ reference
        reference = expm(-1j * angles[layers + layer] * mixer) @ reference
    actual, probabilities = state_at_angles(cut_diagonal(graph), angles, layers)
    np.testing.assert_allclose(actual, reference, atol=2e-14)
    np.testing.assert_allclose(probabilities, abs(reference) ** 2, atol=2e-14)
    assert np.isclose(probabilities.sum(), 1)


def test_one_edge_and_physical_correlations():
    graph = WeightedGraph([(7, 41, 1.0)])
    _, p = state_at_angles(cut_diagonal(graph), [-np.pi / 2, np.pi / 8], 1)
    assert np.isclose(p @ cut_diagonal(graph), 1)
    assert np.isclose(correlation_matrix(p)[0, 1], -1)
    np.testing.assert_allclose(correlation_matrix(np.ones(8) / 8), np.eye(3))


@pytest.mark.parametrize(
    "kind,parameters,expected",
    [
        ("mean_cut", {}, 1.1),
        ("cvar", {"alpha": 1.0}, 1.1),
        ("cvar", {"alpha": 0.25}, 3.0),
        ("cvar", {"alpha": 0.75}, 2.0),
        ("threshold_probability", {"threshold": 0.0}, 0.8),
        ("threshold_probability", {"threshold": 3.0}, 0.5),
        ("threshold_probability", {"threshold": 4.0}, 0.0),
        ("best_of_k", {"k": 1}, 1.1),
    ],
)
def test_known_objective_distribution(kind, parameters, expected):
    config = validate_objective({"type": kind, "parameters": parameters})
    assert np.isclose(objective_value([-2, 0, 3], [0.2, 0.3, 0.5], config), expected)


def test_best_of_k_independent_draws_and_gibbs_stability():
    cuts = [-2, 0, 3]
    p = [0.2, 0.3, 0.5]
    expected = sum(
        max(cuts[i], cuts[j]) * p[i] * p[j] for i in range(3) for j in range(3)
    )
    cfg = validate_objective({"type": "best_of_k", "parameters": {"k": 2}})
    assert np.isclose(objective_value(cuts, p, cfg), expected)
    cfg = validate_objective({"type": "gibbs", "parameters": {"eta": 10000.0}})
    assert np.isclose(objective_value(cuts, p, cfg), 3 + np.log(0.5) / 10000)


def qaoa_config():
    return {
        "seed": 17,
        "parameters": {"layers": 1},
        "optimizer": {"parameters": {"max_steps": 3}},
        "estimation": {"mode": "shots", "shots_per_evaluation": 16},
        "readout": {"mode": "best_of_shots", "shots": 32},
        "limits": {"shots_total": 120},
        "collect": {
            "probabilities": True,
            "statevector": True,
            "optimizer_history": True,
            "objective_history": True,
            "correlations": True,
        },
    }


def test_shot_budget_seed_reuse_and_result_independence():
    graph = WeightedGraph([(2, 7, 2), (7, 19, -1), (2, 19, 1)], nodes=[2, 7, 19, 80])
    original = graph.snapshot()
    solver = QAOACPU(qaoa_config())
    solution = solver.solve(graph)
    first = solver.last_run
    data = solver.result
    assert solver.resources["shots_used"] == 112 <= 120
    assert solver.resources["optimizer_steps"] == 1
    assert solver.termination_reason == "shot_budget"
    assert solver.solve(graph) == solution
    np.testing.assert_array_equal(solver.optimal_angles, first.optimal_angles)
    assert solver.optimizer_history == first.optimizer_history
    assert graph.snapshot() == original
    assert solver.config == solver.result["config"]
    data["solution"]["2"] = 99
    assert solver.solution[2] in (-1, 1)
    graph.set_weight(2, 7, -3)
    solver.solve(graph)
    assert first.graph == original
    assert first.solution == solution
    json.dumps(first.result, allow_nan=False)


def test_budget_can_only_cover_readout_and_zero_training():
    graph = WeightedGraph([(0, 2, 1)])
    config = qaoa_config()
    config["limits"]["shots_total"] = 32
    solver = QAOACPU(config)
    solver.solve(graph)
    assert solver.resources["objective_evaluations"] == 0
    assert solver.resources["shots_used"] == 32
    assert solver.objective_value is None
    config["limits"]["shots_total"] = 31
    with pytest.raises(ValueError):
        QAOACPU(config).solve(graph)


def test_training_improves_the_one_edge_expectation():
    graph = WeightedGraph([(4, 9, 1)])
    config = {"seed": 2, "optimizer": {"parameters": {"max_steps": 60}}}
    solver = QAOACPU(config)
    solver.solve(graph)
    assert solver.expected_cut > 0.98
    assert solver.cut_value == 1


@pytest.mark.parametrize(
    "dtype,tolerance", [("complex64", 2e-5), ("complex128", 1e-12)]
)
def test_probability_normalization_by_precision(dtype, tolerance):
    graph = WeightedGraph([(2, 7, -1), (7, 99, 2)])
    state, p = state_at_angles(cut_diagonal(graph), [0.2, 0.7, -0.3, 1.2], 2, dtype)
    assert state.dtype == np.dtype(dtype)
    assert abs(p.sum() - 1) < tolerance


def test_generator_seed_labels_partition_and_edits():
    from QA_max_cut_lib.generators import partition

    recipe = {
        "type": "gnm",
        "parameters": {"n_vertices": 4, "n_edges": 4, "nodes": [-8, 2, 10, 90]},
        "seed": 7,
        "weights": {
            "type": "signed_discrete",
            "parameters": {"values": [-3, 1]},
            "seed": 8,
        },
    }
    a = GraphFactory.from_recipe(recipe)
    b = GraphFactory.from_recipe(recipe)
    assert a.snapshot() == b.snapshot()
    parts = partition(a, 2, seed=2)
    assert set().union(*(set(p.nodes) for p in parts)) == set(a.nodes)
    for part in parts:
        for u, v, w in part.edges:
            assert a.weight(u, v) == w
    before = a.snapshot()["fingerprint"]
    a.set_weight(-8, 2, 5)
    assert a.snapshot()["fingerprint"] != before
