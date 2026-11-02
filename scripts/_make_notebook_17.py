import json
import os
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
nb = nbf.v4.new_notebook()
cells = [
    nbf.v4.new_markdown_cell(
        """# 17. Взаимная информация: QMI и Z-MI

Две разные меры, partial trace без global density matrix, masks,
final-angle prefixes и passive observer."""
    ),
    nbf.v4.new_code_cell(
        """import json, math
from pathlib import Path
import numpy as np
from QA_max_cut_lib import WeightedGraph
from QA_max_cut_lib.laboratory import QAOAWithMIObserverCPU
from QA_max_cut_lib.tools.mutual_information import *
ROOT = Path.cwd()
if not (ROOT / "configs").exists():
    ROOT = ROOT.parent"""
    ),
    nbf.v4.new_markdown_cell(
        """## A. Product, Bell Φ/Ψ, CZ

QMI и Z-MI — разные величины; CZ plus имеет такие же Z probabilities,
как product plus, но другую QMI."""
    ),
    nbf.v4.new_code_cell(
        """plus = np.ones(4) / 2
bell_phi = np.array([1, 0, 0, 1]) / math.sqrt(2)
bell_psi = np.array([0, 1, 1, 0]) / math.sqrt(2)
cz_plus = np.array([1, 1, 1, -1]) / 2
for name, psi in [("plus", plus), ("Φ", bell_phi), ("Ψ", bell_psi), ("CZ+", cz_plus)]:
    print(name, "QMI", quantum_mutual_information(psi, [0, 1], (0, 1)), "Z-MI", classical_from_probabilities(abs(psi) ** 2, [0, 1], (0, 1)))"""
    ),
    nbf.v4.new_markdown_cell(
        """## B/C. Mixed pair RDM и entropy bug"""
    ),
    nbf.v4.new_code_cell(
        """ghz = np.zeros(8); ghz[0] = ghz[7] = 1 / math.sqrt(2)
print("GHZ pair QMI:", quantum_mutual_information(ghz, [0, 1, 2], (0, 1)))
state = np.zeros(8, dtype=complex); state[0] = state[5] = 1 / math.sqrt(2)
print("Bell(A,E) QMI A,B:", quantum_mutual_information(state, [0, 1, 2], (0, 1)))
print("S_B:", von_neumann_entropy(reduced_density_matrix(state, [1], [0, 1, 2])))"""
    ),
    nbf.v4.new_markdown_cell(
        """## D. Complex RDM и labels"""
    ),
    nbf.v4.new_code_cell(
        """state = np.zeros(16, dtype=complex); state[0] = state[5] = 0.5; state[2] = state[7] = 0.5j
labels = [80, -7, 101, 10]
print(reduced_density_matrix(state, [1], labels))
print("pair QMI:", quantum_mutual_information(state, labels, (80, 101)))"""
    ),
    nbf.v4.new_markdown_cell(
        """## E/F. Counts, masks, honest zero"""
    ),
    nbf.v4.new_code_cell(
        """print("counts MI:", classical_from_counts([[3, 1], [1, 3]]))
print("summary:", map_summary(np.array([2.0, 0.0, 0.0]), np.array([True, True, True])))"""
    ),
    nbf.v4.new_markdown_cell(
        """## Passive QAOA observer

Baseline и observer имеют одни и те же selected angles, shots и solution;
добавляются только MI snapshots."""
    ),
    nbf.v4.new_code_cell(
        """from QA_max_cut_lib.generators.graphs import GraphFactory
from QA_max_cut_lib.solvers.qaoa_cpu import QAOACPU
graph = GraphFactory.from_recipe(json.loads((ROOT / "configs/graphs/block_mean_field_6.json").read_text()))
config = json.loads((ROOT / "configs/solvers/qaoa_mi_cpu.json").read_text())
config["output"]["report"] = 0
plain_config = dict(config); plain_config["type"] = "qaoa_cpu"; plain_config.pop("diagnostics", None)
plain = QAOACPU(plain_config); plain_solution = plain.solve(graph)
observer = QAOAWithMIObserverCPU(config); observed_solution = observer.solve(graph)
print("same solution:", observed_solution == plain_solution)
print("shots:", observer.result["resources"]["shots_used"], plain.result["resources"]["shots_used"])
print("frames:", observer.result["mi_summary"]["snapshot_count"])
print("first frame summary:", observer.result["mi_summary"]["quantum"][0])"""
    ),
    nbf.v4.new_markdown_cell(
        """## Ограничение

QMI — exact simulator diagnostic; Z-MI из finite shots является
эмпирической оценкой и не заменяет QMI. Старые frames RQAOA относятся
к своему graph version и не переносятся на новый compact index."""
    ),
]
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = ROOT / "notebooks" / "17_mutual_information_diagnostics.ipynb"
nbf.write(nb, out)
print(out)
