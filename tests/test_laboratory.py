"""Contract tests for the first Laboratory warm-start transfer."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from QA_max_cut_lib import RunControl, WeightedGraph, restore
from QA_max_cut_lib.laboratory import (
    WarmStartQAOACPU,
    catalog_summary,
    get_file,
    get_variant,
    list_variants,
)
from QA_max_cut_lib.solvers.qaoa_cpu import QAOACPU
from QA_max_cut_lib.solvers.warm_start import (
    WarmStartError,
    clipped_probability,
    product_state,
    state_at_warm_angles,
)
from QA_max_cut_lib.solvers.qaoa_state import cut_diagonal
from QA_max_cut_lib.solvers.warm_start import mixer_unitary


def graph():
    return WeightedGraph(
        [(10, 30, 2), (10, 80, 3), (30, 80, -5)], nodes=[80, 10, 30]
    )


def warm_config(*, mixer="rounded_modified", epsilon=0.25, max_steps=1, shots=True):
    return {
        "type": "laboratory.warm_start_qaoa_cpu",
        "seed": 42,
        "num_treads": 1,
        "parameters": {"layers": 1, "max_vertices": 20, "dtype": "complex128"},
        "warm_start": {
            "epsilon": epsilon,
            "source": {
                "type": "provided_spins",
                "parameters": {
                    "entries": [
                        {"node": 80, "spin": 1},
                        {"node": 10, "spin": -1},
                        {"node": 30, "spin": 1},
                    ]
                },
            },
            "mixer": {"type": mixer},
        },
        "objective": {"type": "mean_cut"},
        "estimation": {"mode": "shots" if shots else "exact", "shots_per_evaluation": 32},
        "optimizer": {
            "type": "adam",
            "parameters": {"max_steps": max_steps, "learning_rate": 0.05, "gradient_tolerance": 0.0},
            "gradient": {"type": "central_difference", "parameters": {"step": 0.01}},
        },
        "initialization": {"type": "fixed_angles", "parameters": {"values": [0.4, 0.2]}},
        "readout": {"mode": "best_of_shots", "shots": 64} if shots else {"mode": "most_probable"},
        "limits": {"shots_total": 544 if shots else 0},
        "collect": {"probabilities": True, "optimizer_history": True},
        "output": {"report": 0, "logs": 0},
    }


def base_config(*, layers=1):
    config = copy.deepcopy(warm_config(max_steps=0, shots=False))
    config.pop("warm_start")
    config["type"] = "qaoa_cpu"
    config["parameters"]["layers"] = layers
    config["initialization"]["parameters"]["values"] = [0.4] * layers + [0.2] * layers
    return config


def test_catalog_covers_archive_and_keeps_deferred_visible():
    summary = catalog_summary()
    assert summary["source_members"] == 89
    assert summary["parsed_members"] == 89
    assert summary["variants"] >= 6
    assert "warm_start_product_qaoa_cpu_v1" in summary["implemented_variant_ids"]
    assert "warm_start_rqaoa" in summary["deferred_variant_ids"]
    alias = get_file("Diplom/Mk_5.py")
    assert alias["ast_sha256"] == get_file("Laboratory/LS_RQAOA_fast_5.py")["ast_sha256"]
    assert alias["path"] != get_file("Laboratory/LS_RQAOA_fast_5.py")["path"]
    deferred = get_variant("warm_start_rqaoa")
    assert deferred["status"]["implementation"] == "not_implemented"
    assert any(not card["runnable"] for card in list_variants())


def test_a_product_state_and_initial_statistics():
    state = product_state([0.25, 0.75])
    probabilities = np.abs(state) ** 2
    assert np.allclose(probabilities, [3 / 16, 1 / 16, 9 / 16, 3 / 16])
    assert np.isclose(probabilities.sum(), 1.0)
    cuts = cut_diagonal(WeightedGraph([(10, 30, 2)], nodes=[10, 30]))
    assert np.isclose(cuts @ probabilities, 1.25)


def test_b_epsilon_half_matches_standard_qaoa_for_both_mixers():
    g = WeightedGraph([(0, 1, 2), (1, 2, -1), (0, 2, 3)], nodes=[0, 1, 2])
    source_entries = [{"node": node, "spin": 1 if node != 1 else -1} for node in g.nodes]
    for layers in (1, 2):
        angles = [0.17 * (i + 1) for i in range(2 * layers)]
        for mixer, beta_sign in (("rounded_modified", 1), ("warm_aligned", -1)):
            config = warm_config(mixer=mixer, epsilon=0.5, max_steps=0, shots=False)
            config["parameters"]["layers"] = layers
            config["warm_start"]["source"]["parameters"]["entries"] = source_entries
            config["initialization"]["parameters"]["values"] = angles
            warm = WarmStartQAOACPU(config)
            warm.solve(g)
            standard_angles = angles[:layers] + [beta_sign * value for value in angles[layers:]]
            standard_config = base_config(layers=layers)
            standard_config["initialization"]["parameters"]["values"] = standard_angles
            standard = QAOACPU(standard_config)
            standard.solve(g)
            assert np.allclose(warm.probabilities, standard.probabilities, atol=1e-12)


def test_c_epsilon_zero_keeps_discrete_basis_state():
    g = WeightedGraph([(0, 1, 2), (1, 2, -3)], nodes=[0, 1, 2])
    entries = [{"node": 0, "spin": 1}, {"node": 1, "spin": -1}, {"node": 2, "spin": 1}]
    for mixer in ("warm_aligned", "rounded_modified"):
        config = warm_config(mixer=mixer, epsilon=0.0, max_steps=0, shots=False)
        config["warm_start"]["source"]["parameters"]["entries"] = entries
        config["initialization"]["parameters"]["values"] = [0.33, 0.7]
        solver = WarmStartQAOACPU(config)
        solver.solve(g)
        assert np.argmax(solver.probabilities) == 2  # bits 0,1,0 -> spins +,-,+
        assert np.isclose(solver.probabilities[2], 1.0, atol=1e-14)
        assert solver.source_cut == solver.cut_value


def test_d_aligned_zero_gamma_preserves_product_and_modified_can_move():
    g = WeightedGraph([(0, 1, 2)], nodes=[0, 1])
    entries = [{"node": 0, "spin": 1}, {"node": 1, "spin": -1}]
    cuts = cut_diagonal(g)
    initial = np.abs(product_state([0.25, 0.75])) ** 2
    _, aligned = state_at_warm_angles(cuts, [0.0, 1.1], 1, [0.25, 0.75], "warm_aligned")
    _, modified = state_at_warm_angles(cuts, [0.0, 1.1], 1, [0.25, 0.75], "rounded_modified")
    assert np.allclose(aligned, initial, atol=1e-14)
    assert not np.allclose(modified, initial, atol=1e-8)
    _, complement = state_at_warm_angles(cuts, [0.0, np.pi / 2], 1, [0.25, 0.75], "rounded_modified")
    assert np.isclose(complement[1], 1.0, atol=1e-12)
    assert g.cut_value({0: -1, 1: 1}) == g.cut_value({0: 1, 1: -1})


def test_e_fractional_clipping_and_invalid_sources():
    assert np.allclose(clipped_probability([0.1, 0.6, 0.95], 0.2), [0.2, 0.6, 0.8])
    with pytest.raises(WarmStartError):
        clipped_probability([1.1], 0.2)
    with pytest.raises(WarmStartError):
        clipped_probability([0.5], -0.1)
    with pytest.raises(WarmStartError):
        clipped_probability([0.5], 0.6)
    config = warm_config(max_steps=0)
    config["warm_start"]["source"]["parameters"]["entries"].pop()
    with pytest.raises(WarmStartError):
        WarmStartQAOACPU(config).solve(graph())
    duplicate = warm_config(max_steps=0)
    duplicate["warm_start"]["source"]["parameters"]["entries"].append(
        {"node": 80, "spin": 1}
    )
    with pytest.raises(WarmStartError):
        WarmStartQAOACPU(duplicate).solve(graph())


def test_f_budget_and_pause_resume(tmp_path):
    g = graph()
    continuous = WarmStartQAOACPU(warm_config(max_steps=3))
    continuous_solution = continuous.solve(g)
    assert continuous.resources["shots_used"] == 544
    paused_config = warm_config(max_steps=3)
    paused_config["checkpoint"] = {
        "enabled": True,
        "path": str(tmp_path / "run"),
        "every": {"optimizer_iterations": 1},
    }
    paused = WarmStartQAOACPU(paused_config)
    paused.solve(g, control=RunControl(pause_after={"optimizer_iterations": 1}))
    assert paused.status == "paused"
    assert paused.resources["known_spent"] == 160
    assert paused.resources["readout_reserved"] == 64
    resumed = restore(paused.last_checkpoint.path)
    final = resumed.resume()
    assert resumed.status == "completed"
    assert resumed.resources["shots_used"] == 544
    assert np.allclose(resumed.optimal_angles, continuous.optimal_angles)
    assert final == continuous_solution
    assert resumed.result["variant_id"] == "warm_start_product_qaoa_cpu_v1"
    assert resumed.result["mixer_id"] == "rounded_modified"


def test_classical_exact_source_is_explicit_and_reusable(tmp_path):
    g = graph()
    config = warm_config(max_steps=0, shots=False)
    config["warm_start"]["source"] = {
        "type": "classical_solver",
        "parameters": {
            "solver_type": "exact_cpu",
            "config": {"type": "exact_cpu", "parameters": {"max_vertices": 8}},
        },
    }
    solver = WarmStartQAOACPU(config)
    solver.solve(g)
    assert solver.source_type == "classical_solver"
    assert solver.source_cut is not None
    assert solver.source_run["solver_type"] == "exact_cpu"
    assert solver.source_cost is not None
    # A second graph gets fresh data and no stale labels.
    g2 = WeightedGraph([(1, 2, 1)], nodes=[1, 2])
    solver.solve(g2)
    assert solver.source_run["solution"] == {1: -1, 2: 1} or solver.source_run["solution"] == {1: 1, 2: -1}
    assert solver.warm_start_data["node_order"] == [1, 2]


def test_h_controlled_series_plan_is_42_runs():
    from QA_max_cut_lib import tools

    spec = tools.load_json("experiments/benchmark_warm_start.json")
    plan = tools.plan_experiments(spec)
    assert plan.counts["graph_instances"] == 3
    assert plan.counts["cases"] == 21
    assert plan.counts["trials"] == 42
    assert plan.counts["reference_jobs"] == 3


def test_mixer_unitaries_norm_and_endpoints():
    for q in (0.0, 0.25, 0.5, 0.75, 1.0):
        for beta in (-1.3, -0.2, 0.0, 0.7, 2.2):
            for mixer in ("x", "warm_aligned", "rounded_modified"):
                matrix = mixer_unitary(beta, q, mixer)
                assert np.allclose(matrix.conj().T @ matrix, np.eye(2), atol=1e-12)
                state = product_state([q, 1 - q])
                assert np.isclose(np.vdot(state, state).real, 1.0)
                _, probabilities = state_at_warm_angles(
                    np.array([0.0, 2.0, 2.0, 0.0]), [0.0, beta], 1, [q, 1 - q], mixer
                )
                assert np.isclose(probabilities.sum(), 1.0)
