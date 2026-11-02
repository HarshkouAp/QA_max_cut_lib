import json
import os
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
nb = nbf.v4.new_notebook()
cells = [
    nbf.v4.new_markdown_cell(
        """# 15. Блочный mean-field QAOA

Локальные full-state QAOA на блоках, обмен m/q и два расписания:
Gauss–Seidel и Jacobi. Выход — sign_means и полный cut."""
    ),
    nbf.v4.new_code_cell(
        """import json
from pathlib import Path
from QA_max_cut_lib import WeightedGraph
from QA_max_cut_lib.laboratory import BlockMeanFieldQAOACPU
from QA_max_cut_lib.solvers.block_mean_field import build_block_problem, damped_moments

ROOT = Path.cwd()
if not (ROOT / "configs").exists():
    ROOT = ROOT.parent
graph = WeightedGraph(
    [(10, 30, 5), (10, 80, 1), (30, 80, 1), (80, 101, 2), (101, 200, 3), (30, 200, -2)],
    nodes=[-7, 10, 30, 80, 101, 200],
)
config = json.loads((ROOT / "configs/solvers/block_mean_field_jacobi_cpu.json").read_text())
config["output"]["report"] = 0
print("partition:", config["partition"]["parameters"]["blocks"])"""
    ),
    nbf.v4.new_markdown_cell(
        """## Поля одного блока

Внешние means дают `h=(1/2) sum w m`; `c_aff` считается отдельно."""
    ),
    nbf.v4.new_code_cell(
        """means = {node: 0.1 for node in graph.nodes}
problem = build_block_problem(graph, [30, 80], means, block_id=1)
print("h:", problem.fields)
print("J:", problem.couplings)
print("c_aff:", problem.constant)"""
    ),
    nbf.v4.new_markdown_cell(
        """## Один Jacobi run"""
    ),
    nbf.v4.new_code_cell(
        """jacobi = BlockMeanFieldQAOACPU(config)
solution = jacobi.solve(graph)
print("solution:", solution)
print("full cut:", jacobi.cut_value)
print("model mean cut:", jacobi.result["model_mean_cut"])
print("sweeps:", jacobi.result["completed_sweeps"], "shots:", jacobi.result["resources"]["shots_used"])"""
    ),
    nbf.v4.new_markdown_cell(
        """## Gauss–Seidel

Тот же partition и budget, но поля каждого следующего блока строятся
по уже опубликованному bank."""
    ),
    nbf.v4.new_code_cell(
        """gs_config = json.loads((ROOT / "configs/solvers/block_mean_field_gauss_seidel_cpu.json").read_text())
gs_config["output"]["report"] = 0
gs = BlockMeanFieldQAOACPU(gs_config)
gs_solution = gs.solve(graph)
print("solution:", gs_solution)
print("full cut:", gs.cut_value)
print("model mean cut:", gs.result["model_mean_cut"])
print("sweeps:", gs.result["completed_sweeps"], "shots:", gs.result["resources"]["shots_used"])"""
    ),
    nbf.v4.new_markdown_cell(
        """## Согласованное damping

Одна convex mixture применяется и к m, и к внутреннему q. Сырой q с
уже сглаженным m не является физической парной моделью."""
    ),
    nbf.v4.new_code_cell(
        """m, q = damped_moments(
    {0: 1.0, 1: 1.0}, {(0, 1): 1.0},
    {0: -1.0, 1: 1.0}, {(0, 1): -1.0}, 0.5,
)
print("damped m:", m)
print("damped q:", q)"""
    ),
    nbf.v4.new_markdown_cell(
        """## Ограничение

Модельный mean cut и cut округлённого решения — разные числа. В этом
run последний может быть хуже; это честное свойство sign-rounded
messages, а не ошибка readout."""
    ),
]
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = ROOT / "notebooks" / "15_block_mean_field_exchange.ipynb"
nbf.write(nb, out)
print(out)
