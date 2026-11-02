"""Independent checks for the exact Max-Cut and Ising reduction core."""

from itertools import product
import json
import math

import pytest

import QA_max_cut_lib
from QA_max_cut_lib import (
    ExactCPU,
    GraphReducer,
    IsingProblem,
    SpinConstraint,
    WeightedGraph,
)


def assignments(nodes):
    """Yield every spin assignment for a node tuple."""
    for spins in product((-1, 1), repeat=len(nodes)):
        yield dict(zip(nodes, spins))


def direct_cut(graph, spins):
    """Sum the cut directly from graph edges."""
    return math.fsum(
        weight for u, v, weight in graph.edges if spins[u] != spins[v]
    )


def direct_ising(problem, spins):
    """Sum the Ising polynomial directly from its coefficients."""
    terms = [problem.constant]
    terms.extend(problem.field(node) * spins[node] for node in problem.nodes)
    terms.extend(
        coupling * spins[u] * spins[v]
        for (u, v), coupling in problem.couplings.items()
    )
    return math.fsum(terms)


def close(left, right, tolerance=1e-12):
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def control_graph():
    return WeightedGraph(
        edges=[(10, 30, 2.0), (10, 80, 3.0), (30, 80, 5.0)],
        nodes=[10, 30, 80, 101],
    )


def test_graph_to_ising_matches_cut_for_all_assignments():
    graph = WeightedGraph(
        edges=[(-2, 5, 1.5), (-2, 9, -2.0), (5, 9, 0.0), (5, 20, 3.25)],
        nodes=[-2, 5, 9, 20, 77],
    )
    problem = IsingProblem.from_maxcut(graph)
    assert close(problem.constant, -0.5 * math.fsum(w for _, _, w in graph.edges))
    for spins in assignments(graph.nodes):
        assert close(problem.energy(spins), -direct_cut(graph, spins))


def test_control_example_metrics_graph_and_restore():
    graph = control_graph()
    reducer = GraphReducer()
    state = reducer.start(graph)
    assert close(state.ising.constant, -5.0)
    assert state.cut_offset == 0.0
    assert state.ising.edge_count == 3
    before_graph = graph.snapshot()

    event = reducer.reduce(state, 10, 30, -1, correlation=-0.6)

    assert event["step"] == 1
    assert event["removed_node"] == 10
    assert event["retained_node"] == 30
    assert event["sign"] == -1
    assert close(event["selected_weight"], 2.0)
    assert close(event["fixed_energy"], -2.0)
    assert close(event["expected_fixed_energy"], -1.6)
    assert close(event["constraint_probability"], 0.8)
    assert close(event["cut_offset_delta"], 5.0)
    assert close(event["cut_offset"], 5.0)
    assert close(event["ising_constant_before"], -5.0)
    assert close(event["ising_constant"], -6.0)
    assert close(event["ising_constant_delta"], -1.0)
    assert event["vertices_after"] == 3
    assert event["edges_after"] == 1
    assert close(event["sum_abs_weight"], 2.0)
    assert close(event["mean_abs_weight"], 2.0)
    assert event["maxcut_metrics_applicable"] is True

    working = state.current_graph()
    assert working.nodes == (30, 80, 101)
    assert working.edges == ((30, 80, 2.0),)
    assert state.ising.constant == -state.cut_offset - 0.5 * 2.0

    original_problem = IsingProblem.from_maxcut(graph)
    for residual in assignments(state.current_nodes):
        before = {10: -residual[30], **residual}
        assert close(
            direct_ising(original_problem, before),
            direct_ising(state.ising, residual),
        )
        assert close(
            direct_cut(graph, before),
            state.cut_offset + direct_cut(working, residual),
        )

    residual = {30: 1, 80: -1, 101: 1}
    restored = reducer.restore(state, residual)
    assert restored == {10: -1, 30: 1, 80: -1, 101: 1}
    assert set(restored) == set(graph.nodes)
    assert close(state.original_cut_value(restored), 7.0)
    assert close(state.original_energy(restored), -7.0)
    assert graph.snapshot() == before_graph


@pytest.mark.parametrize("sign", [-1, 1])
def test_general_ising_merge_with_fields(sign):
    problem = IsingProblem(
        constant=0.7,
        fields={-1: 0.3, 2: -0.4, 7: 0.9},
        couplings={(-1, 2): -0.5, (-1, 7): 1.25, (2, 7): 0.75},
        nodes=[-1, 2, 7, 100],
    )
    reducer = GraphReducer()
    state = reducer.start_from_ising(problem)
    event = reducer.reduce(state, -1, 2, sign, correlation=0.25)
    assert event["maxcut_metrics_applicable"] is False
    assert event["maxcut_metrics_reason"] == "original_problem_is_general_ising"
    assert event["selected_weight"] is None
    assert event["fixed_energy"] is None
    assert event["cut_offset"] is None
    assert close(state.ising.field(2), -0.4 + sign * 0.3)

    for residual in assignments(state.current_nodes):
        before = {-1: sign * residual[2], **residual}
        assert close(
            direct_ising(problem, before),
            direct_ising(state.ising, residual),
        )
    assert event["ising_constant"] == state.ising.constant
    residual = {node: 1 for node in state.current_nodes}
    restored = reducer.restore(state, residual)
    assert set(restored) == set(problem.nodes)


def test_non_adjacent_pair_has_zero_weight_but_can_change_offset():
    graph = WeightedGraph(
        edges=[(0, 1, 2.0), (0, 2, 3.0), (1, 2, -1.0), (2, 3, 4.0)],
        nodes=[0, 1, 2, 3, 9],
    )
    reducer = GraphReducer()
    state = reducer.start(graph)
    event = reducer.reduce(state, 0, 3, -1)
    assert close(event["selected_weight"], 0.0)
    assert close(event["fixed_energy"], 0.0)
    assert close(event["cut_offset_delta"], 5.0)
    assert close(event["cut_offset"], 5.0)
    working = state.current_graph()
    for residual in assignments(state.current_nodes):
        before = {0: -residual[3], **residual}
        assert close(
            direct_cut(graph, before),
            state.cut_offset + direct_cut(working, residual),
        )


def test_exact_zero_coupling_is_removed_and_small_value_is_kept():
    graph = WeightedGraph(
        edges=[
            (1, 3, 2.0),
            (2, 3, 2.0),
            (1, 4, 1.0),
            (2, 4, 1.0 + 1e-12),
        ],
        nodes=[1, 2, 3, 4],
    )
    reducer = GraphReducer()
    state = reducer.start(graph)
    event = reducer.reduce(state, 1, 2, -1)
    assert state.ising.coupling(2, 3) == 0.0
    assert close(state.ising.coupling(2, 4), 1e-12)
    assert state.ising.edge_count == 1
    assert close(event["sum_abs_weight"], 2e-12)
    for residual in assignments(state.current_nodes):
        before = {1: -residual[2], **residual}
        assert close(
            direct_cut(graph, before),
            state.cut_offset + direct_cut(state.current_graph(), residual),
        )


def test_chain_restores_in_reverse_order_including_removed_retained_node():
    graph = WeightedGraph(
        edges=[
            (0, 1, 1.0),
            (0, 2, 2.0),
            (1, 2, -1.0),
            (1, 3, 3.0),
            (2, 4, 0.5),
            (3, 4, 2.0),
        ],
        nodes=[0, 1, 2, 3, 4],
    )
    reducer = GraphReducer()
    state = reducer.start(graph)
    reducer.reduce(state, 0, 1, -1)
    reducer.reduce(state, 1, 2, 1)
    reducer.reduce(state, 2, 3, -1)
    assert state.current_nodes == (3, 4)
    assert [item.removed for item in state.constraints] == [0, 1, 2]
    assert [item.retained for item in state.constraints] == [1, 2, 3]
    assert state.trace[1]["fixed_energy"] == 0.0
    assert state.trace[1]["cut_offset_delta"] == 0.0
    assert close(
        state.ising.constant,
        -state.cut_offset - 0.5 * math.fsum(w for _, _, w in state.current_graph().edges),
    )

    for residual in assignments(state.current_nodes):
        expected = {3: residual[3], 4: residual[4]}
        expected[2] = -expected[3]
        expected[1] = expected[2]
        expected[0] = -expected[1]
        assert close(
            direct_cut(graph, expected),
            state.cut_offset + direct_cut(state.current_graph(), residual),
        )
        assert close(
            direct_ising(IsingProblem.from_maxcut(graph), expected),
            direct_ising(state.ising, residual),
        )
        assert reducer.restore(state, residual) == expected


def test_negative_labels_gaps_and_isolated_nodes():
    graph = WeightedGraph(
        edges=[(-7, 3, 2.0), (-7, 1000000, -1.0), (3, 42, 4.0)],
        nodes=[-7, 1000000, 3, 42, 900],
    )
    reducer = GraphReducer()
    state = reducer.start(graph)
    reducer.reduce(state, -7, 3, 1)
    residual = {node: 1 for node in state.current_nodes}
    restored = reducer.restore(state, residual)
    assert set(restored) == set(graph.nodes)
    assert all(type(spin) is int and spin in (-1, 1) for spin in restored.values())
    assert 900 in restored


def test_empty_one_two_nodes_and_zero_weights():
    reducer = GraphReducer()
    empty = reducer.start(WeightedGraph())
    assert empty.current_nodes == ()
    assert empty.restore({}) == {}
    assert empty.original_energy({}) == 0.0

    single = WeightedGraph(nodes=[7])
    state = reducer.start(single)
    assert state.restore({7: -1}) == {7: -1}

    pair = WeightedGraph([(4, 9, 1.5)], nodes=[4, 9])
    state = reducer.start(pair)
    reducer.reduce(state, 4, 9, -1)
    assert state.restore({9: 1}) == {4: -1, 9: 1}
    assert close(state.original_cut_value({4: -1, 9: 1}), 1.5)

    zero = WeightedGraph([(0, 1, 0.0)], nodes=[0, 1])
    state = reducer.start(zero)
    event = reducer.reduce(state, 0, 1, -1)
    assert state.ising.edge_count == 0
    assert event["mean_abs_weight"] is None
    assert event["mean_abs_weight_reason"] == "empty_nonzero_edge_set"
    assert close(event["sum_abs_weight"], 0.0)
    assert event["expected_fixed_energy"] is None
    assert event["expected_fixed_energy_reason"] == "physical_correlation_not_provided"
    assert state.restore({1: 1}) == {0: -1, 1: 1}


def test_snapshot_roundtrip_continues_and_restores():
    graph = WeightedGraph(
        edges=[(0, 1, 1.0), (0, 2, 2.0), (1, 2, -1.0), (2, 3, 4.0)],
        nodes=[0, 1, 2, 3, 8],
    )
    reducer = GraphReducer()
    state = reducer.start(graph)
    reducer.reduce(state, 0, 1, -1, correlation=-0.4)
    snapshot = state.snapshot()
    text = json.dumps(snapshot, ensure_ascii=False, allow_nan=False)
    restored_state = reducer.from_snapshot(json.loads(text))
    assert json.dumps(restored_state.snapshot(), sort_keys=True) == json.dumps(
        snapshot, sort_keys=True
    )
    assert restored_state.constraints == state.constraints
    assert close(restored_state.cut_offset, state.cut_offset)

    event_a = reducer.reduce(state, 1, 2, 1)
    event_b = reducer.reduce(restored_state, 1, 2, 1)
    assert event_a == event_b
    assert json.dumps(state.snapshot(), sort_keys=True) == json.dumps(
        restored_state.snapshot(), sort_keys=True
    )
    residual = {node: 1 for node in state.current_nodes}
    assert reducer.restore(state, residual) == reducer.restore(
        restored_state, residual
    )


def test_invalid_requests_leave_state_unchanged():
    reducer = GraphReducer()
    state = reducer.start(control_graph())
    before = json.dumps(state.snapshot(), sort_keys=True)
    invalid_calls = [
        (10, 10, 1),
        (999, 30, 1),
        (10, 999, 1),
        (10, 30, 0),
        (10, 30, 2),
        (10, 30, True),
        (10, 30, 1.0),
        (10, 30, 1, {"correlation": float("nan")}),
        (10, 30, 1, {"correlation": float("inf")}),
        (10, 30, 1, {"correlation": 1.001}),
        (10, 30, 1, {"correlation": -1.001}),
        (10, 30, 1, {"selection_score": float("nan")}),
    ]
    for removed, retained, sign, *extra in invalid_calls:
        options = extra[0] if extra else {}
        with pytest.raises((TypeError, ValueError, ArithmeticError)):
            reducer.reduce(state, removed, retained, sign, **options)
        assert json.dumps(state.snapshot(), sort_keys=True) == before

    reducer.reduce(state, 10, 30, 1)
    after_valid = json.dumps(state.snapshot(), sort_keys=True)
    with pytest.raises(ValueError):
        reducer.reduce(state, 10, 30, 1)
    assert json.dumps(state.snapshot(), sort_keys=True) == after_valid

    for residual in ({30: 1}, {30: 1, 80: 1}, {30: 0, 80: 1, 101: 1},
                     {30: True, 80: 1, 101: 1}, {30: 1.0, 80: 1, 101: 1}):
        with pytest.raises((TypeError, ValueError)):
            reducer.restore(state, residual)


def test_general_ising_metrics_and_empty_domain():
    problem = IsingProblem(
        constant=-0.25,
        fields={1: 0.5},
        couplings={(1, 2): 0.75},
        nodes=[1, 2, 3],
    )
    reducer = GraphReducer()
    state = reducer.start_from_ising(problem)
    event = reducer.reduce(state, 1, 2, -1, correlation=0.5, selection_score=2.0)
    assert event["maxcut_metrics_applicable"] is False
    assert event["maxcut_metrics_reason"] == "original_problem_is_general_ising"
    assert event["selected_weight"] is None
    assert event["fixed_energy"] is None
    assert event["expected_fixed_energy"] is None
    assert event["cut_offset"] is None
    assert event["constraint_probability"] == 0.25
    assert event["selection_score"] == 2.0
    assert math.isfinite(event["sum_abs_coupling"])
    with pytest.raises(ValueError):
        state.current_graph()


def test_ising_snapshot_and_field_rejection():
    problem = IsingProblem(
        constant=0.5,
        fields={1: 0.25, 2: -0.5},
        couplings={(1, 2): 0.75, (2, 3): -1.0},
        nodes=[1, 2, 3, 4],
    )
    snapshot = problem.snapshot()
    restored = IsingProblem.from_snapshot(json.loads(json.dumps(snapshot)))
    for spins in assignments(problem.nodes):
        assert close(direct_ising(problem, spins), direct_ising(restored, spins))
    with pytest.raises(ValueError):
        problem.to_weighted_graph()
    zero_fields = IsingProblem(couplings={(1, 2): 1.0}, nodes=[1, 2])
    graph = zero_fields.to_weighted_graph()
    assert graph.edges == ((1, 2, 2.0),)


def test_version_is_consistent_in_results_and_package():
    assert QA_max_cut_lib.__version__ == "0.1.0a13"
    solver = ExactCPU()
    solver.solve(WeightedGraph([(0, 1, 1.0)]))
    assert solver.result["package_version"] == "0.1.0a13"
    assert SpinConstraint(1, 2, -1).to_dict() == {
        "removed_node": 1,
        "retained_node": 2,
        "sign": -1,
    }
