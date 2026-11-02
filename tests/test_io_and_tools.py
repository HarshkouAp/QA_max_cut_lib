"""Check export integrity, plotting contracts, and package entry points."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from QA_max_cut_lib import WeightedGraph, QAOACPU, ExactCPU, SolverFactory
from QA_max_cut_lib import tools


@pytest.fixture
def solved():
    graph = WeightedGraph([(2, 7, 2), (7, 19, -1), (2, 19, 1)])
    config = {
        "seed": 4,
        "optimizer": {"parameters": {"max_steps": 2}},
        "collect": {
            "statevector": True,
            "probabilities": True,
            "optimizer_history": True,
        },
    }
    solver = QAOACPU(config)
    solver.solve(graph)
    return solver


@pytest.mark.parametrize("savedir", [False, True])
def test_selected_json_native_roundtrip_and_no_overwrite(tmp_path, solved, savedir):
    config = {
        "path": str(tmp_path),
        "filename": "result",
        "savedir": savedir,
        "atts": ["solution", "cut_value"],
        "save_original_data": ["statevector", "probabilities"],
    }
    files = tools.save_result(solved, config)
    folder = tmp_path / "result" if savedir else tmp_path
    assert files.primary == folder / "result.json"
    document = json.loads(files.primary.read_text())
    assert "statevector" not in document["attributes"]
    restored = tools.load_result(files.primary, load_original_data=True)
    assert restored.solution == solved.solution
    np.testing.assert_array_equal(restored.statevector, solved.statevector)
    np.testing.assert_array_equal(restored.probabilities, solved.probabilities)
    before = {p.name: p.read_bytes() for p in files.files}
    with pytest.raises(FileExistsError):
        tools.save_result(solved, config)
    assert before == {p.name: p.read_bytes() for p in files.files}


def test_additional_file_conflict_leaves_no_primary(tmp_path, solved):
    old = tmp_path / "run_statevector.npy"
    old.write_bytes(b"previous scientific data")
    config = {
        "path": str(tmp_path),
        "filename": "run",
        "atts": [],
        "save_original_data": ["statevector"],
    }
    with pytest.raises(FileExistsError):
        tools.save_result(solved, config)
    assert old.read_bytes() == b"previous scientific data"
    assert not (tmp_path / "run.json").exists()


def test_simultaneous_exports_have_exactly_one_winner(tmp_path, solved):
    config = {
        "path": str(tmp_path),
        "filename": "race",
        "atts": ["solution"],
        "save_original_data": ["statevector"],
    }

    def export():
        try:
            tools.save_result(solved, config)
            return "saved"
        except FileExistsError:
            return "conflict"

    with ThreadPoolExecutor(2) as pool:
        statuses = list(pool.map(lambda _: export(), range(2)))
    assert statuses.count("saved") == 1
    assert statuses.count("conflict") == 1
    restored = tools.load_result(tmp_path / "race.json", load_original_data=True)
    np.testing.assert_array_equal(restored.statevector, solved.statevector)


def test_save_is_optional_and_export_failure_retains_result(tmp_path):
    graph = WeightedGraph([(0, 2, 1)])
    assert list(tmp_path.iterdir()) == []
    solver = ExactCPU()
    solver.solve(graph)
    assert list(tmp_path.iterdir()) == []
    (tmp_path / "result.json").write_text("old")
    solver = ExactCPU(
        {"save": {"path": str(tmp_path), "filename": "result", "atts": ["solution"]}}
    )
    with pytest.raises(FileExistsError):
        solver.solve(graph)
    assert solver.cut_value == 1
    assert (tmp_path / "result.json").read_text() == "old"


@pytest.mark.parametrize("report", [0, 1, 2])
@pytest.mark.parametrize("logs", [0, 1, 2])
def test_independent_output_levels(report, logs, capsys):
    solver = QAOACPU(
        {
            "seed": 4,
            "optimizer": {"parameters": {"max_steps": 1}},
            "output": {"color": "never"},
        }
    )
    solver.solve(WeightedGraph([(0, 2, 1)]), report=report, logs=logs)
    output = capsys.readouterr().out
    assert ("QA_max_cut_lib |" in output) == (report > 0)
    assert ("QAOA training starts" in output) == (logs > 0)
    assert ("Step 1:" in output) == (logs == 2)
    assert "\033[" not in output
    assert solver.resources["objective_evaluations"] == 5


def test_no_silent_unimplemented_features():
    graph = WeightedGraph([(0, 2, 1)])
    with pytest.raises(ValueError):
        ExactCPU().solve(graph, num_treads=2)
    with pytest.raises(ValueError):
        ExactCPU({"checkpoint": {"enabled": True, "path": "checkpoints/unsupported"}})
    with pytest.raises(ValueError):
        SolverFactory.from_config({"type": "qaoa_quantum"})
    assert SolverFactory.available() == (
        "exact_cpu",
        "qaoa_cpu",
        "rqaoa_cpu",
        "ls_rqaoa_cpu",
        "qaoa_gpu",
        "rqaoa_gpu",
        "ls_rqaoa_gpu",
        "qaoa_in_qaoa_cpu",
        "qaoa_in_qaoa_gpu",
        "gw_cpu",
        "simulated_annealing_cpu",
        "refined",
    )


def test_import_is_inert_and_solver_module_examples_run():
    code = "import sys; import QA_max_cut_lib; assert 'matplotlib.pyplot' not in sys.modules; assert 'cupy' not in sys.modules"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    for module in ["exact_cpu", "qaoa_cpu"]:
        result = subprocess.run(
            [sys.executable, "-m", "QA_max_cut_lib.solvers." + module, "--report", "0"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == "" and "RuntimeWarning" not in result.stderr


def test_plots_use_record_snapshot_and_local_style(tmp_path, solved):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    old = matplotlib.rcParams.copy()
    cfg = {
        "style": {"text_engine": "mathtext", "figsize": [7, 4]},
        "save": {
            "path": str(tmp_path),
            "filename": "probabilities",
            "format": "png",
            "DPI": 100,
            "metadata": True,
        },
    }
    figures = [
        tools.plot_probabilities(solved, cfg),
        tools.plot_graph_solution(solved, config={"style": {"figsize": [7, 4]}}),
        tools.plot_graph_info(solved, {"style": {"figsize": [7, 4]}}),
        tools.plot_optimization(solved, {"style": {"figsize": [7, 4]}}),
    ]
    assert (tmp_path / "probabilities.png").stat().st_size > 1000
    assert figures[0].texts
    for key in ["font.family", "font.serif", "text.usetex", "grid.alpha"]:
        assert matplotlib.rcParams[key] == old[key]
    for fig in figures:
        plt.close(fig)
    with pytest.raises(ValueError):
        tools.plot_probabilities(ExactCPU().last_run)


def test_landscape_axis_order_and_roundtrip(tmp_path):
    graph = WeightedGraph([(7, 20, 1)])
    scan = tools.scan_landscape(
        graph, {"gamma": [-np.pi / 2, 0.0, np.pi / 2], "beta": [0.0, np.pi / 8]}
    )
    assert scan.values.shape == (2, 3)
    assert np.isclose(scan.values[1, 0], -1.0)
    assert np.isclose(scan.values[1, 2], 0.0)
    cfg = {
        "path": str(tmp_path),
        "filename": "map",
        "atts": ["gamma", "beta"],
        "save_original_data": ["values"],
    }
    files = tools.save_result(scan, cfg)
    restored = tools.load_result(files.primary, load_original_data=True)
    np.testing.assert_array_equal(restored.values, scan.values)
    fig = tools.plot_landscape(restored, {"style": {"figsize": [7, 4]}})
    import matplotlib.pyplot as plt

    plt.close(fig)
