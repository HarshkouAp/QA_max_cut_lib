"""Checkpoint, resume, resource-journal, and compatibility tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from QA_max_cut_lib import QAOACPU, RunControl, WeightedGraph, inspect_checkpoint, restore
from QA_max_cut_lib.core.checkpoint import (
    CheckpointCorrupt,
    CheckpointLocked,
    CheckpointManager,
    ResourceJournal,
    load_checkpoint,
)


def qaoa_config(root, *, layers=1, mode="shots", max_steps=4, shots=32, readout=64, total=704):
    return {
        "type": "qaoa_cpu",
        "seed": 42,
        "num_treads": 1,
        "parameters": {"layers": layers},
        "objective": {"type": "mean_cut"},
        "estimation": {"mode": mode, "shots_per_evaluation": shots},
        "optimizer": {
            "type": "adam",
            "parameters": {"max_steps": max_steps, "learning_rate": 0.05, "gradient_tolerance": 0.0},
            "gradient": {"type": "central_difference", "parameters": {"step": 0.01}},
        },
        "initialization": {"type": "fixed_angles", "parameters": {"values": [0.5] * layers + [0.3] * layers}},
        "readout": {"mode": "best_of_shots" if readout else "most_probable", **({"shots": readout} if readout else {})},
        "limits": {"shots_total": total},
        "checkpoint": {
            "enabled": True,
            "path": str(root),
            "format": "json_npy_v1",
            "every": {"optimizer_iterations": 1},
            "at_phase_boundaries": True,
            "on_pause": True,
            "durable_resource_journal": True,
            "durability": "local_fsync",
            "on_write_error": "stop",
            "retention": {"type": "keep_all"},
        },
        "collect": {"optimizer_history": True, "objective_history": True},
        "output": {"report": 0, "logs": 0},
    }


def graph():
    return WeightedGraph([(0, 1, 1), (1, 2, -2), (2, 3, 3)], nodes=[0, 1, 2, 3])


def test_pause_resume_matches_continuous_shots(tmp_path):
    continuous = QAOACPU(qaoa_config(tmp_path / "continuous"))
    first_solution = continuous.solve(graph())
    paused = QAOACPU(qaoa_config(tmp_path / "paused"))
    paused_solution = paused.solve(graph(), control=RunControl(pause_after={"optimizer_iterations": 2}))
    assert paused.status == "paused"
    assert paused.readout_completed is False
    assert paused.resources["known_spent"] == 320
    assert paused.resources["readout_reserved"] == 64
    assert paused.resources["shots_available"] == 320
    assert paused_solution is not None
    checkpoint = paused.last_checkpoint.path
    info = inspect_checkpoint(checkpoint)
    assert info["phase"] == "training"
    assert info["next_operation"] == {"optimizer_iteration": 3}
    continued = restore(checkpoint)
    continued_solution = continued.resume()
    assert continued.status == "completed"
    assert continued.resources["objective_evaluations"] == 20
    assert continued.resources["shots_used"] == 704
    assert np.array_equal(continued.optimal_angles, continuous.optimal_angles)
    assert continued_solution == first_solution
    assert continued.optimizer_history == continuous.optimizer_history
    assert continued.run_id == paused.run_id


def test_p2_exact_and_best_probe_boundary(tmp_path):
    continuous = QAOACPU(qaoa_config(tmp_path / "c", layers=2, mode="exact", readout=0, total=0))
    first = continuous.solve(graph())
    paused = QAOACPU(qaoa_config(tmp_path / "p", layers=2, mode="exact", readout=0, total=0))
    paused.solve(graph(), control=RunControl(pause_after={"optimizer_iterations": 1}))
    resumed = restore(paused.last_checkpoint.path)
    second = resumed.resume()
    assert second == first
    assert np.array_equal(resumed.optimal_angles, continuous.optimal_angles)
    assert resumed.resources["objective_evaluations"] == 36


def test_max_steps_zero_and_early_gradient_stop(tmp_path):
    cfg = qaoa_config(tmp_path / "zero", max_steps=0, readout=0, total=0)
    solver = QAOACPU(cfg)
    solver.solve(graph())
    assert solver.resources["optimizer_steps"] == 0
    assert solver.termination_reason == "max_steps"


def test_journal_conservative_charge_once(tmp_path):
    journal = ResourceJournal(tmp_path)
    journal.begin("iteration-3", attempt_id="attempt-1", upper_bound=160, kind="optimizer_iteration")
    ledger = journal.reconcile()
    assert ledger["known_spent"] == 0
    assert ledger["uncertain_upper_charge"] == 160
    journal.resolve("iteration-3", charge=160, reason="recovery")
    again = journal.reconcile()
    assert again["known_spent"] == 160
    assert again["uncertain_upper_charge"] == 0
    assert again["total_charged"] == 160


def test_crash_recovery_charges_uncertain_and_stops(tmp_path):
    paused = QAOACPU(qaoa_config(tmp_path / "run"))
    paused.solve(graph(), control=RunControl(pause_after={"optimizer_iterations": 2}))
    run_dir = paused.last_checkpoint.path.parent
    journal = ResourceJournal(run_dir)
    journal.begin(f"{paused.run_id}:iteration:3", attempt_id="crash", upper_bound=160, kind="optimizer_iteration")
    resumed = restore(paused.last_checkpoint.path)
    resumed.resume()
    assert resumed.resources["known_spent"] == 544
    assert resumed.resources["uncertain_upper_charge"] == 160
    assert resumed.resources["total_charged"] == 704
    assert resumed.resources["optimizer_steps"] == 3
    assert resumed.status == "stopped"


def test_corrupt_new_generation_preserves_old(tmp_path):
    solver = QAOACPU(qaoa_config(tmp_path / "run"))
    solver.solve(graph())
    old_path = solver.last_checkpoint.path
    manager = CheckpointManager(
        solver.config["checkpoint"], run_id=solver.run_id, component_id="qaoa_cpu"
    )
    new_ref = manager.write_generation(
        {"phase": "training", "next_iteration": 9, "config": solver.config, "graph": graph().snapshot()},
        {"theta": np.array([1.0, 2.0])},
        parent=manager.last_ref.generation,
        phase="training",
        next_operation={"optimizer_iteration": 9},
    )
    (new_ref.path / "theta.npy").write_bytes(b"corrupt")
    with pytest.raises(CheckpointCorrupt):
        load_checkpoint(new_ref.path)
    assert load_checkpoint(old_path)["manifest"]["generation"] == solver.last_checkpoint.generation


def test_owner_lock_rejects_second_writer(tmp_path):
    manager = CheckpointManager(
        {"enabled": True, "path": str(tmp_path), "every": {"optimizer_iterations": 1}},
        run_id="run",
        component_id="qaoa_cpu",
    )
    manager.acquire()
    second = CheckpointManager(
        {"enabled": True, "path": str(tmp_path), "every": {"optimizer_iterations": 1}},
        run_id="run",
        component_id="qaoa_cpu",
    )
    with pytest.raises(CheckpointLocked):
        second.acquire()
    manager.release()
    second.acquire()
    second.release()


def test_completed_readout_resume_is_idempotent(tmp_path):
    solver = QAOACPU(qaoa_config(tmp_path / "run", max_steps=1, mode="exact", readout=0, total=0))
    solver.solve(graph())
    before = len(list(solver.last_checkpoint.path.parent.parent.glob("gen_*")))
    final = restore(solver.last_checkpoint.path)
    final.resume()
    after = len(list(solver.last_checkpoint.path.parent.parent.glob("gen_*")))
    assert after == before
    assert final.status == "completed"


def test_new_solve_after_restore_is_not_resume(tmp_path):
    solver = QAOACPU(qaoa_config(tmp_path / "run"))
    solver.solve(graph(), control=RunControl(pause_after={"optimizer_iterations": 1}))
    old_run = solver.run_id
    restored = restore(solver.last_checkpoint.path)
    restored.solve(graph())
    assert restored.status == "completed"
    assert restored.run_id != old_run


def test_edited_scientific_config_is_rejected(tmp_path):
    import json
    from QA_max_cut_lib.core.checkpoint import CheckpointIncompatible

    solver = QAOACPU(qaoa_config(tmp_path / "run"))
    solver.solve(graph(), control=RunControl(pause_after={"optimizer_iterations": 1}))
    state_path = Path(solver.last_checkpoint.path) / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["config"]["seed"] = 999
    state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(CheckpointCorrupt):
        restore(solver.last_checkpoint.path)
