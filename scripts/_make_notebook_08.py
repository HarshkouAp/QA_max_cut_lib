"""Build the executable Stage 07 hierarchy notebook."""

from pathlib import Path
import nbformat as nbf


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


cells = [
    md(
        """# QAOA-in-QAOA: block patterns and global orientations

**Версия 0.1.0a7.** Ноутбук показывает базовую иерархию: локальный рисунок
спинов блока, одну свободную ориентацию блока, точный метаграф и
восстановление исходных вершин.

Это не LS-RQAOA: здесь выбирается не корреляция пары, а целый дискретный
рисунок блока. Локальный разрез и постоянная часть `Delta b` — разные
величины."""
    ),
    code(
        """from pathlib import Path
from copy import deepcopy
import json
import subprocess
import sys
import tempfile

import numpy as np

import QA_max_cut_lib
from QA_max_cut_lib import (
    WeightedGraph, QAOAInQAOACPU, QAOAInQAOAGPU, tools,
)
from QA_max_cut_lib.solvers.hierarchy_components import project_blocks
from QA_max_cut_lib.solvers.gpu_state import gpu_available

PROJECT = Path.cwd()
if PROJECT.name == "notebooks":
    PROJECT = PROJECT.parent
print(QA_max_cut_lib.__version__)
print("package:", Path(QA_max_cut_lib.__file__).resolve())
assert QA_max_cut_lib.__version__ == "0.1.0a7"
print("gpu_available:", gpu_available())"""
    ),
    md(
        """## 1. Один блок: рисунок и переворот

Сначала проверяется точная формула на контрольном графе из задания. Блоки
`[10,30]` и `[80,101]` имеют рисунки `(+1,-1)`. Метаграф содержит вес `-2`,
`Delta b=10`, а для всех четырёх ориентаций `C_original=Delta b+C_meta`."""
    ),
    code(
        """graph = WeightedGraph(
    [(10,30,2),(80,101,3),(10,80,5),(10,101,1),(30,80,4),(30,101,-2)],
    nodes=[10,30,80,101],
)
blocks = [(10,30),(80,101)]
patterns = {10:{10:1,30:-1}, 80:{80:1,101:-1}}
meta, delta_b, records = project_blocks(graph, blocks, patterns)
print("meta nodes:", meta.nodes)
print("meta edges:", meta.edges)
print("Delta b:", delta_b)
for tau10, tau80 in ((-1,-1),(-1,1),(1,-1),(1,1)):
    solution = {
        10: patterns[10][10]*tau10, 30: patterns[10][30]*tau10,
        80: patterns[80][80]*tau80, 101: patterns[80][101]*tau80,
    }
    print(tau10, tau80, graph.cut_value(solution), delta_b + meta.cut_value({10:tau10,80:tau80}))
assert meta.edges == ((10,80,-2.0),)
assert delta_b == 10.0"""
    ),
    md(
        """## 2. Несколько уровней настоящего QAOA

На каждом уровне локальные задачи решают только индуцированный блок. После
readout рисунок нормируется по anchor, а `GraphReducer` применяет `N_l-K`
точных ограничений ко всему текущему графу."""
    ),
    code(
        """edges = [
    (0,1,1),(1,2,-2),(2,3,2),(3,4,1),(4,5,-1),(5,6,2),(6,7,1),
    (0,7,-2),(0,3,1),(1,4,2),(2,5,-1),(3,6,1),(4,7,-2),
]
graph8 = WeightedGraph(edges, nodes=list(range(8)))
config = tools.load_json(PROJECT / "configs/solvers/qaoa_in_qaoa_cpu.json")
config.pop("save", None)
config["num_treads"] = 1
config["parameters"]["finish_at"] = 2
config["local_solver"]["parameters"]["max_vertices"] = 2
config["local_solver"]["optimizer"]["parameters"]["max_steps"] = 1
config["partitioner"]["parameters"]["max_size"] = 2
solver = QAOAInQAOACPU(config)
solution = solver.solve(graph8, report=0, logs=0)
print("solution:", solution)
print("cut:", solver.cut_value, "levels:", solver.levels_completed)
print("source:", solver.solution_source, "resources:", solver.resources)
assert set(solution) == set(graph8.nodes)
assert solver.levels_completed >= 2
assert solver.solution_source == "exact_finisher"
assert abs(solver.cut_value - graph8.cut_value(solution)) < 1e-9"""
    ),
    md(
        """## 3. CPU-параллелизм через CLI

Jupyter на Windows не всегда безопасно запускает `spawn`-пул из главного
ноутбука. Поэтому реальный параллельный путь проверяется импортируемым CLI в
отдельном процессе. Это тот же coordinator, а не другой алгоритм."""
    ),
    code(
        """parallel_config = deepcopy(config)
parallel_config["num_treads"] = 2
parallel_config["parameters"]["max_levels"] = 2
with tempfile.TemporaryDirectory(prefix="hierarchy_cli_") as tmp:
    cfg_path = Path(tmp) / "parallel.json"
    cfg_path.write_text(json.dumps(parallel_config), encoding="utf-8")
    command = [
        sys.executable, "-m", "QA_max_cut_lib.solvers.qaoa_in_qaoa_cpu",
        "--graph", str(PROJECT / "configs/graphs/complete_signed_7.json"),
        "--config", str(cfg_path), "--report", "1", "--logs", "0",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    print(completed.stdout)
    assert completed.returncode == 0, completed.stderr
    assert "qaoa_in_qaoa_cpu" in completed.stdout
    assert '"num_treads_applied": 2' in completed.stdout"""
    ),
    md(
        """## 4. Общий бюджет readout

Для `best_of_shots` минимум задачи равен её M. Все минимумы следующего уровня
резервируются до запуска. Резерв не удваивается для каждого блока."""
    ),
    code(
        """shot_config = tools.load_json(PROJECT / "configs/solvers/qaoa_in_qaoa_shots.json")
shot_config.pop("save", None)
shot_config["num_treads"] = 1
shot_config["partitioner"]["parameters"]["max_size"] = 2
shot_config["local_solver"]["parameters"]["max_vertices"] = 2
shot_config["local_solver"]["readout"]["shots"] = 8
shot_config["limits"]["shots_total"] = 32
shot_solver = QAOAInQAOACPU(shot_config)
shot_solution = shot_solver.solve(graph8, report=0, logs=0)
print("shots:", shot_solver.resources["shots_used"])
print("readout:", shot_solver.resources["readout_shots"])
print("calls:", shot_solver.local_solver_calls)
assert set(shot_solution) == set(graph8.nodes)
assert shot_solver.resources["shots_used"] <= 32
assert shot_solver.resources["readout_shots"] > 0"""
    ),
    md(
        """## 5. Необязательный NVIDIA-раздел

Раздел исполняется только при доступном CuPy и NVIDIA. При отсутствии
устройства он честно сообщает, что GPU-часть не проверялась. Локальные блоки
обслуживает один владелец устройства."""
    ),
    code(
        """if gpu_available():
    gpu_config = tools.load_json(PROJECT / "configs/solvers/qaoa_in_qaoa_gpu.json")
    gpu_config.pop("save", None)
    gpu_config["num_treads"] = 2
    gpu_config["partitioner"]["parameters"]["max_size"] = 2
    gpu_config["local_solver"]["parameters"]["max_vertices"] = 2
    gpu_config["local_solver"]["optimizer"]["parameters"]["max_steps"] = 1
    gpu_solver = QAOAInQAOAGPU(gpu_config)
    gpu_solution = gpu_solver.solve(graph8, report=0, logs=0)
    print("GPU:", gpu_solver.status, gpu_solver.levels_completed, gpu_solver.cut_value)
    print("GPU resources:", gpu_solver.resources)
    assert set(gpu_solution) == set(graph8.nodes)
    assert gpu_solver.resources["num_treads_applied"] == 1
    assert gpu_solver.resources["backend"] == "cupy"
else:
    print("GPU QAOA-in-QAOA not checked: no compatible CuPy/NVIDIA device.")"""
    ),
    md(
        """## 6. Восстановление, дерево и графики

Дерево имеет составные узлы `(level,anchor)`, а листья — исходные вершины.
Углы разных блоков не усредняются; график 3D использует уровень иерархии как
Z."""
    ),
    code(
        """print("tree roots:", solver.hierarchy_tree["roots"])
print("levels:", solver.levels_completed, "primitive reductions:",
      solver.primitive_reductions_completed)
print("b:", solver.final_cut_offset, "Ising constant:", solver.final_ising_constant)
overview = tools.plot_hierarchy_overview(solver, {"style": {"figsize": [7,5]}})
parameters = tools.plot_hierarchy_parameters_3d(solver, {"style": {"figsize": [7,5]}})
tree = tools.plot_hierarchy_tree(solver, {"style": {"figsize": [7,5]}})
print("plots:", type(overview).__name__, type(parameters).__name__, type(tree).__name__)
assert overview.qa_record.algorithm_type == "qaoa_in_qaoa_hierarchy_overview\""""
    ),
    md("## 7. Сохранение и повторное чтение"),
    code(
        """with tempfile.TemporaryDirectory(prefix="hierarchy_save_") as tmp:
    save = {
        "path": str(Path(tmp) / "results"),
        "atts": ["solution", "cut_value", "hierarchy_history", "block_patterns",
                 "parameter_trajectory", "hierarchy_tree"],
        "filename": "hierarchy",
        "save_original_data": [],
        "makedir": True,
        "savedir": True,
    }
    saved = tools.save_result(solver, save)
    loaded = tools.load_result(saved.primary)
    print("saved:", saved.primary)
    print("loaded:", loaded.get("algorithm_type"))
    assert loaded.get("algorithm_type") == "qaoa_in_qaoa_cpu"
    assert loaded.get("hierarchy_tree") is not None"""
    ),
    md("## 8. Итог"),
    code(
        """print("version:", QA_max_cut_lib.__version__)
print("control projection: ok")
print("multiple hierarchy levels: ok")
print("CPU parallel CLI: ok")
print("readout budget: ok")
print("tree/plots/save: ok")
print("GPU branch checked:", gpu_available())"""
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={"kernelspec": {"name": "qa-max-cut-lib-gpu", "display_name": "QA max cut lib GPU", "language": "python"}},
)
path = Path(__file__).resolve().parents[1] / "notebooks/08_qaoa_in_qaoa.ipynb"
nbf.write(notebook, str(path))
print(path)
