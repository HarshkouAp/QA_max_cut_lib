import json
import os
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
nb = nbf.v4.new_notebook()
cells = []
cells.append(nbf.v4.new_markdown_cell("""# 14. Условный QAOA и environment LS

Внешние спины фиксированы. Активный блок обучается полным CPU statevector;
локальные поля не попадают в глобальный Max-Cut граф."""))
cells.append(nbf.v4.new_code_cell("""import json
from pathlib import Path
from QA_max_cut_lib import WeightedGraph
from QA_max_cut_lib.solvers.conditional import build_conditional_problem
from QA_max_cut_lib.laboratory import ConditionalQAOACPU, EnvironmentLSRQAOACPU

ROOT = Path.cwd()
if not (ROOT / "configs").exists():
    ROOT = ROOT.parent
import os
os.chdir(ROOT)
graph = WeightedGraph(
    [(10, 30, 2), (10, 80, 3), (30, 80, -4), (30, 101, 5), (80, 101, -2)],
    nodes=[-7, 10, 30, 80, 101],
)
reference = {-7: -1, 10: 1, 30: 1, 80: 1, 101: -1}
problem = build_conditional_problem(graph, [10, 30], reference)
print("J:", problem.couplings)
print("h:", problem.fields)
print("c_aff:", problem.constant)
print("C_TT:", problem.outside_cut)
print("A_S:", problem.affected_cuts)
print("C_full:", problem.original_cuts)"""))
cells.append(nbf.v4.new_markdown_cell("""## Standalone QAOA на одном блоке"""))
cells.append(nbf.v4.new_code_cell("""config = json.loads((ROOT / "configs/solvers/conditional_qaoa_cpu.json").read_text())
config["output"]["report"] = 0
solver = ConditionalQAOACPU(config)
solution = solver.solve(graph)
print(solution)
print("full cut:", solver.cut_value)
print("affected:", solver.affected_cut_value, "outside:", solver.outside_cut)
print("shots:", solver.result["resources"]["shots_used"])"""))
cells.append(nbf.v4.new_markdown_cell("""## Full-graph limit

При `S=V` поля нулевые; схема должна совпасть с обычным `QAOACPU`."""))
cells.append(nbf.v4.new_code_cell("""import numpy as np
from QA_max_cut_lib import QAOACPU

full = WeightedGraph([(0, 1, 2), (1, 2, -1), (0, 2, 3)], nodes=[0, 1, 2])
base = {
    "schema_version": 1, "seed": 2, "parameters": {"layers": 1},
    "objective": {"type": "mean_cut"}, "estimation": {"mode": "exact"},
    "optimizer": {"type": "adam", "parameters": {"max_steps": 0}},
    "initialization": {"type": "fixed_angles", "parameters": {"values": [0.2, 0.3]}},
    "readout": {"mode": "most_probable"}, "limits": {"shots_total": 0},
    "collect": {"probabilities": True}, "output": {"report": 0},
}
ordinary = QAOACPU({**base, "type": "qaoa_cpu"})
ordinary.solve(full)
cc = dict(config)
cc["parameters"] = {"active_vertices": [0, 1, 2], "layers": 1, "max_active_vertices": 8, "dtype": "complex128"}
cc["environment"] = {"type": "fixed_spins", "source": {"type": "all_plus"}}
cc["initialization"] = {"type": "fixed_angles", "parameters": {"values": [0.2, 0.3]}}
conditional = ConditionalQAOACPU(cc)
conditional.solve(full)
print("max probability error:", float(np.max(np.abs(ordinary.probabilities - conditional.probabilities))))"""))
cells.append(nbf.v4.new_markdown_cell("""## Environment LS

Один frozen reference на шаг, затем global reducer и retain survivor."""))
cells.append(nbf.v4.new_code_cell("""ls_config = json.loads((ROOT / "configs/solvers/environment_ls_rqaoa_cpu.json").read_text())
ls_config["output"]["report"] = 0
ls_graph = WeightedGraph([(10, 30, 2), (10, 80, 3), (30, 80, 5)], nodes=[10, 30, 80, 101])
ls = EnvironmentLSRQAOACPU(ls_config)
ls_solution = ls.solve(ls_graph)
print(ls_solution)
print("cut:", ls.cut_value, "reductions:", ls.result["reductions_completed"])
print("source:", ls.result["solution_source"], "shots:", ls.result["resources"]["shots_used"])"""))
cells.append(nbf.v4.new_markdown_cell("""## Граница

`h` — из фиксированного окружения, а не self-loop и не атрибут графа.
Soft means, correlation-weighted fields и итеративный обмен `〈Z〉` —
отдельные deferred-стратегии, не эта recipe."""))
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = ROOT / "notebooks" / "14_conditional_fields_and_environment_ls.ipynb"
nbf.write(nb, out)
print(out)
