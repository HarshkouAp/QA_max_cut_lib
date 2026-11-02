"""Build the executable Stage 06 GPU notebook."""

from pathlib import Path
import nbformat as nbf


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


cells = [
    md(
        """# NVIDIA GPU QAOA

**Версия 0.1.0a6.** Ноутбук проверяет реальный CuPy-путь: диагностику
устройства, full и pair-представления, CPU/GPU-сравнение при фиксированных
углах, все пять критериев, семплер, два readout-режима, GPU-RQAOA и небольшую
GPU-LS задачу.

GPU-среда не обязательна для импорта пакета. Этот ноутбук должен исполняться
с установленным CuPy на NVIDIA-машине."""
    ),
    code(
        """from pathlib import Path
import json
import tempfile
import numpy as np

import QA_max_cut_lib
from QA_max_cut_lib import (
    GraphFactory, QAOAGPU, QAOAGPUCorrelationSource, RQAOAGPU, LSRQAOAGPU, tools,
)
from QA_max_cut_lib.solvers.gpu_state import (
    gpu_available, gpu_device_info, cut_diagonal_gpu, state_at_angles_gpu,
    restore_full_gpu, correlations_gpu, objective_value_gpu, sample_full_indices_gpu,
)
from QA_max_cut_lib.solvers.qaoa_state import cut_diagonal, state_at_angles
from QA_max_cut_lib.solvers.objectives import validate_objective, objective_value

PROJECT = Path.cwd()
if PROJECT.name == "notebooks":
    PROJECT = PROJECT.parent
print(QA_max_cut_lib.__version__)
print("package:", Path(QA_max_cut_lib.__file__).resolve())
print("gpu_available:", gpu_available())
assert QA_max_cut_lib.__version__ == "0.1.0a6"
assert gpu_available()"""
    ),
    md("## Диагностика устройства"),
    code(
        """info = gpu_device_info(0)
print(json.dumps(info, indent=2))
assert info["name"]
assert info["memory_total"] > info["memory_free"]"""
    ),
    md(
        """## Full и pair-состояния при фиксированных углах

Pair-режим хранит `chi`, а не физический `psi`. Восстановленный полный вектор
сравнивается с CPU-эталоном."""
    ),
    code(
        """graph = GraphFactory.from_recipe({
    "type": "complete", "seed": 11,
    "parameters": {"n_vertices": 6},
    "weights": {"type": "signed_discrete", "seed": 12, "parameters": {"values": [-2, -1, 1, 2]}},
})
angles = [0.3, 0.2, 0.4, -0.1]
cpu_state, cpu_masses = state_at_angles(cut_diagonal(graph), angles, 2, "complex128")
for representation in ("full", "spin_flip_pairs"):
    gpu_cuts = cut_diagonal_gpu(graph, representation)
    gpu_state, gpu_masses = state_at_angles_gpu(
        gpu_cuts, angles, 2, n_vertices=len(graph),
        representation=representation, dtype="complex128",
    )
    if representation == "full":
        error = float(np.max(np.abs(gpu_state.get() - cpu_state)))
        print("full state error:", error)
        assert error < 1e-11
    else:
        restored = restore_full_gpu(gpu_state, len(graph)).get()
        pair_error = float(np.max(np.abs(restored - cpu_state)))
        print("pair restored error:", pair_error)
        assert pair_error < 1e-11
        assert abs(float(gpu_masses.sum()) - 1) < 1e-11"""
    ),
    md("## Все пять критериев"),
    code(
        """cuts = np.array([-2.0, 0.0, 1.0, 3.0])
masses = np.array([0.1, 0.2, 0.3, 0.4])
for raw in (
    {"type": "mean_cut", "parameters": {}},
    {"type": "cvar", "parameters": {"alpha": 0.5}},
    {"type": "gibbs", "parameters": {"eta": 1.3}},
    {"type": "best_of_k", "parameters": {"k": 3}},
    {"type": "threshold_probability", "parameters": {"threshold": 1.0}},
):
    criterion = validate_objective(raw)
    expected = objective_value(cuts, masses, criterion)
    actual = objective_value_gpu(cuts, masses, criterion)
    print(criterion["type"], actual, "error", abs(actual - expected))
    assert abs(actual - expected) < 1e-12"""
    ),
    md("## Семплер и обе ориентации пары"),
    code(
        """pair_cuts = cut_diagonal_gpu(graph, "spin_flip_pairs")
pair_state, pair_masses = state_at_angles_gpu(
    pair_cuts, [0.3, 0.2], 1, n_vertices=len(graph),
    representation="spin_flip_pairs", dtype="complex128",
)
import cupy as cp
rng = cp.random.RandomState(19)
sampled = sample_full_indices_gpu(
    pair_masses, 20000, n_vertices=len(graph),
    representation="spin_flip_pairs", rng=rng,
)
counts = cp.bincount(sampled, minlength=1 << len(graph)).astype(float) / 20000
expected_physical = cp.concatenate([pair_masses / 2, pair_masses[::-1] / 2])
print("physical mass max error:", float(cp.max(cp.abs(counts - expected_physical))))
assert float(cp.max(cp.abs(counts - expected_physical))) < 0.02"""
    ),
    md("## QAOAGPU: обучение, корреляции и readout"),
    code(
        """gpu_config = tools.load_json(PROJECT / "configs/solvers/qaoa_gpu_pair.json")
gpu_config.pop("save", None)
gpu_config["parameters"]["max_vertices"] = 10
gpu_config["optimizer"]["parameters"]["max_steps"] = 2
gpu_config["collect"]["correlations"] = True
solver = QAOAGPU(gpu_config)
solution = solver.solve(graph, report=0, logs=0)
print("solution:", solution)
print("cut:", solver.cut_value, "status:", solver.status)
print("state:", solver.state_representation["type"])
print("resources:", solver.resources)
assert set(solution) == set(graph.nodes)
assert solver.resources["shots_used"] == 0
assert solver.state_representation["type"] == "spin_flip_pairs\""""
    ),
    md("## Readout best-of-shots"),
    code(
        """shot_config = tools.load_json(PROJECT / "configs/solvers/qaoa_gpu_cvar_shots.json")
shot_config.pop("save", None)
shot_config["parameters"]["max_vertices"] = 10
shot_config["optimizer"]["parameters"]["max_steps"] = 1
shot_config["limits"]["shots_total"] = 2000
shot_solver = QAOAGPU(shot_config)
shot_solution = shot_solver.solve(graph, report=0, logs=0)
print("shots:", shot_solver.resources["shots_used"], "status:", shot_solver.status)
assert set(shot_solution) == set(graph.nodes)
assert shot_solver.resources["shots_used"] <= 2000"""
    ),
    md("## GPU-RQAOA и GPU-LS"),
    code(
        """rq_config = tools.load_json(PROJECT / "configs/solvers/rqaoa_gpu.json")
rq_config.pop("save", None)
rq_solver = RQAOAGPU(rq_config)
rq_solution = rq_solver.solve(graph, report=0, logs=0)
print("RQAOA:", rq_solver.status, rq_solver.cut_value,
      "reductions:", rq_solver.reductions_completed)
assert set(rq_solution) == set(graph.nodes)

ls_config = tools.load_json(PROJECT / "configs/solvers/ls_rqaoa_gpu.json")
ls_config.pop("save", None)
ls_config["parameters"]["max_reductions"] = 4
ls_config["subgraph_builder"]["parameters"]["max_size"] = 4
ls_solver = LSRQAOAGPU(ls_config)
ls_solution = ls_solver.solve(graph, report=0, logs=0)
print("LS:", ls_solver.status, ls_solver.cut_value,
      "owner:", ls_solver.resources["num_treads_applied"])
assert set(ls_solution) == set(graph.nodes)
assert ls_solver.resources["backend"] == "gpu_owner\""""
    ),
    md("## Сохранение и CPU-анализ"),
    code(
        """with tempfile.TemporaryDirectory(prefix="gpu_qaoa_") as tmp:
    save_config = dict(gpu_config)
    save_config["save"] = {
        "path": str(Path(tmp) / "results"),
        "atts": ["solution", "cut_value", "optimal_angles", "correlations", "state_representation"],
        "filename": "gpu_notebook",
        "save_original_data": [],
        "makedir": True,
        "savedir": True,
    }
    saved = tools.save_result(solver, save_config["save"])
    loaded = tools.load_result(saved.primary)
    print("saved:", saved.primary)
    print("loaded algorithm:", loaded.get("algorithm_type"))
    assert loaded.get("algorithm_type") == "qaoa_gpu"
    figure = tools.plot_optimization(solver) if solver.optimizer_history else None
    print("plot:", type(figure).__name__ if figure is not None else "no history")"""
    ),
    md("## Итог"),
    code(
        """print("version:", QA_max_cut_lib.__version__)
print("GPU:", info["name"], "CC", info["compute_capability"])
print("full/pair CPU agreement: ok")
print("five criteria: ok")
print("sampler orientations: ok")
print("GPU RQAOA/LS: ok")"""
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={"kernelspec": {"name": "qa-max-cut-lib-gpu", "display_name": "QA max cut lib GPU", "language": "python"}},
)
path = Path(__file__).resolve().parents[1] / "notebooks/07_nvidia_gpu_qaoa.ipynb"
nbf.write(notebook, str(path))
print(path)
