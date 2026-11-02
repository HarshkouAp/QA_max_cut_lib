"""Mathematical tests for MI tools and passive observer assemblies."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from QA_max_cut_lib import WeightedGraph
from QA_max_cut_lib.generators.graphs import GraphFactory
from QA_max_cut_lib.laboratory import QAOAWithMIObserverCPU
from QA_max_cut_lib.solvers.mi_observer import QAOAWithMIObserverCPU as ObserverClass
from QA_max_cut_lib.solvers.qaoa_cpu import QAOACPU
from QA_max_cut_lib.tools.mutual_information import (
    MutualInformationError,
    classical_from_counts,
    classical_from_moments,
    classical_from_probabilities,
    classical_mutual_information,
    hop_distances,
    map_summary,
    quantum_mutual_information,
    reduced_density_matrix,
    von_neumann_entropy,
)


def test_standard_qmi_table():
    plus = np.ones(4) / 2
    bell_phi = np.array([1, 0, 0, 1]) / math.sqrt(2)
    bell_psi = np.array([0, 1, 1, 0]) / math.sqrt(2)
    cz_plus = np.array([1, 1, 1, -1]) / 2
    assert quantum_mutual_information(plus, [0, 1], (0, 1)) == pytest.approx(0.0)
    assert classical_from_probabilities(abs(plus) ** 2, [0, 1], (0, 1)) == pytest.approx(0.0)
    assert quantum_mutual_information(bell_phi, [0, 1], (0, 1)) == pytest.approx(2.0)
    assert quantum_mutual_information(bell_psi, [0, 1], (0, 1)) == pytest.approx(2.0)
    assert classical_from_probabilities(abs(bell_phi) ** 2, [0, 1], (0, 1)) == pytest.approx(1.0)
    assert classical_from_probabilities(abs(bell_psi) ** 2, [0, 1], (0, 1)) == pytest.approx(1.0)
    assert quantum_mutual_information(cz_plus, [0, 1], (0, 1)) == pytest.approx(2.0)
    assert classical_from_probabilities(abs(cz_plus) ** 2, [0, 1], (0, 1)) == pytest.approx(0.0)


def test_ghz_pair_is_mixed():
    ghz = np.zeros(8)
    ghz[0] = ghz[7] = 1 / math.sqrt(2)
    labels = [0, 1, 2]
    assert von_neumann_entropy(reduced_density_matrix(ghz, [0], labels)) == pytest.approx(1.0)
    assert quantum_mutual_information(ghz, labels, (0, 1)) == pytest.approx(1.0)
    assert classical_from_probabilities(abs(ghz) ** 2, labels, (0, 1)) == pytest.approx(1.0)


def test_legacy_entropy_bug_is_rejected_by_correct_pair():
    state = np.zeros(8, dtype=complex)
    state[0] = state[5] = 1 / math.sqrt(2)  # Bell(A,E) tensor |0>_B; A=0,B=1,E=2.
    labels = [0, 1, 2]
    assert quantum_mutual_information(state, labels, (0, 1)) == pytest.approx(0.0)
    # Correct entropy of B is zero; repeating S_A would be a false nonzero value.
    assert von_neumann_entropy(reduced_density_matrix(state, [1], labels)) == pytest.approx(0.0)


def test_complex_rdm_labels_and_basis_order():
    state = np.zeros(16, dtype=complex)
    state[0] = state[5] = 0.5
    state[2] = state[7] = 0.5j
    labels = [80, -7, 101, 10]
    rho = reduced_density_matrix(state, [1], labels)
    assert rho == pytest.approx(np.array([[0.5, -0.5j], [0.5j, 0.5]]), abs=1e-12)
    assert quantum_mutual_information(state, labels, (80, 101)) == pytest.approx(2.0)


def test_counts_and_moments():
    assert classical_from_counts([[3, 1], [1, 3]]) == pytest.approx(0.18872187554086717)
    assert classical_from_counts([[8, 0], [0, 0]]) == pytest.approx(0.0)
    assert classical_from_moments(0, 0, 0) == pytest.approx(0.0)
    assert classical_mutual_information([[0.5, 0], [0, 0.5]]) == pytest.approx(1.0)
    with pytest.raises(MutualInformationError):
        classical_mutual_information([[1.1, -0.1], [0, 0]])


def test_masks_statistics_and_hop_profile():
    values = np.array([2.0, 0.0, 0.0])
    mask = np.array([True, True, True])
    summary = map_summary(values, mask)
    assert summary["count"] == 3
    assert summary["min"] == 0.0
    assert summary["mean"] == pytest.approx(2 / 3)
    assert summary["population_std"] == pytest.approx(2 * math.sqrt(2) / 3)
    graph = WeightedGraph([(0, 1, 2), (1, 2, -3)], nodes=[0, 1, 2, 3])
    distances, disconnected = hop_distances(graph)
    assert distances[(0, 2)] == 2
    assert distances[(0, 3)] is None
    assert (0, 3) in disconnected


def test_target_qaoa_observer_is_passive():
    graph = GraphFactory.from_recipe(json.load(open("configs/graphs/block_mean_field_6.json")))
    config = json.loads(Path("configs/solvers/qaoa_mi_cpu.json").read_text())
    baseline = dict(config)
    baseline["type"] = "qaoa_cpu"
    baseline.pop("diagnostics", None)
    baseline["output"]["report"] = 0
    plain = QAOACPU(baseline)
    plain_solution = plain.solve(graph)
    observer = QAOAWithMIObserverCPU(config)
    observed_solution = observer.solve(graph)
    assert observed_solution == plain_solution
    assert observer.cut_value == plain.cut_value
    assert np.allclose(observer.optimal_angles, plain.optimal_angles)
    assert observer.result["mi_summary"]["snapshot_count"] == 3
    assert len(observer.result["mi_frames"][0]["pairs"]) == 15
    assert observer.result["resources"]["shots_used"] == plain.result["resources"]["shots_used"]


def test_prefix_theta_uses_two_halves():
    graph = WeightedGraph([(0, 1, 2), (1, 2, -3), (0, 2, 5)], nodes=[0, 1, 2])
    config = json.loads(Path("configs/solvers/qaoa_mi_cpu.json").read_text())
    config["parameters"]["layers"] = 3
    config["initialization"]["parameters"]["values"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    config["estimation"] = {"mode": "exact"}
    config["limits"] = {"shots_total": 0}
    config["readout"] = {"mode": "most_probable"}
    config["checkpoint"] = {"enabled": False}
    config["diagnostics"]["mutual_information"]["snapshots"] = {
        "type": "final_angle_prefixes",
        "include_initial": False,
    }
    config["output"]["report"] = 0
    solver = QAOAWithMIObserverCPU(config)
    solver.solve(graph)
    frames = solver.result["mi_frames"]
    assert [frame["layer"] for frame in frames] == [1, 2, 3]
    assert frames[1]["pairs"]
