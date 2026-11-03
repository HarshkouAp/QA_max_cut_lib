"""Formula tests for final decision/quantisation components."""

import math

import numpy as np
import pytest

from QA_max_cut_lib import WeightedGraph
from QA_max_cut_lib.solvers.strategies import (
    ParityForest,
    hierarchical_proxy,
    legacy_history_mix,
    odds_fusion,
    positive_history_mix,
    quantize_angles,
    restriction_cut_proxy,
    selected_observation,
)


def test_parity_forest_accept_conflict_redundant():
    forest = ParityForest([10, 30, 80, 101])
    assert forest.add(10, 30, -1)["status"] == "accepted"
    assert forest.add(30, 80, -1)["status"] == "accepted"
    assert forest.add(10, 80, -1)["status"] == "conflict"
    assert forest.add(10, 80, 1)["status"] == "redundant"


def test_odds_and_selected_observation():
    evidence = odds_fusion([0.6, 0.6])
    assert evidence.value == pytest.approx(15 / 17)
    assert evidence.observations["p_plus"] == pytest.approx(16 / 17)
    conflict = odds_fusion([1.0, -1.0])
    assert conflict.valid is False
    best = selected_observation(
        [
            {"physical_correlation": -0.8, "weight": 1, "round": 0, "task_id": "a"},
            {"physical_correlation": 0.2, "weight": 1, "round": 1, "task_id": "b"},
            {"physical_correlation": 0.9, "weight": 1, "round": 2, "task_id": "c"},
        ]
    )
    assert best.value == 0.9


def test_angle_quantization_ties_to_even():
    delta = math.pi / 8
    grid = quantize_angles([0.5 * delta, -0.5 * delta, 1.5 * delta, 2.5 * delta], delta=delta)
    assert grid["integer_grid"] == [0, 0, 2, 2]


def test_restriction_proxy_is_read_only():
    graph = WeightedGraph(
        [(10, 30, 2), (10, 80, 3), (30, 80, 5)],
        nodes=[10, 30, 80, 101],
    )
    correlations = {(10, 30): -0.4, (10, 80): 0.2, (30, 80): -0.3}
    before = dict(correlations)
    result = restriction_cut_proxy(graph, correlations, (10, 30), -1)
    assert result["value"] == pytest.approx(6.3)
    assert correlations == before


def test_history_and_hierarchy():
    assert legacy_history_mix(0.8, -0.8, -1, 2)["value"] == pytest.approx(-2.4)
    assert positive_history_mix(0.8, -0.8, -1, 2)["value"] == pytest.approx(-0.8)
    proxy = hierarchical_proxy(0, 0, 1, 0.9, 0)
    assert proxy.raw == pytest.approx(math.sqrt(0.19))
    assert proxy.valid is False
    assert proxy.reason == "frechet_violation"
