from pathlib import Path
from nbformat import v4 as nbf
import nbformat

cells = []
def md(text):
    cells.append(nbf.new_markdown_cell(text))
def code(text):
    cells.append(nbf.new_code_cell(text))

md("""# Checkpoint и resume QAOA

Этот ноутбук показывает:
- разницу между новым `solve()` и продолжением `resume()`;
- согласованную паузу после двух ADAM-итераций;
- inspect без запуска solver;
- продолжение в новом объекте и сравнение с непрерывным запуском;
- консервативный charge после незавершённой попытки;
- capability matrix для остальных реализаций.

Основной пример — CPU и малый statevector. GPU checkpoint в `0.1.0a10`
намеренно не заявлен: CuPy RNG требует отдельного переносимого протокола.
""")

code("""from copy import deepcopy
from pathlib import Path
import os
import tempfile

ROOT = Path.cwd()
if not (ROOT / "experiments").exists():
    ROOT = ROOT.parent
os.chdir(ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from QA_max_cut_lib import GraphFactory, QAOACPU, RunControl, inspect_checkpoint, restore, tools
from QA_max_cut_lib.core.checkpoint import ResourceJournal
from QA_max_cut_lib.core.checkpoint_capabilities import capabilities

graph = GraphFactory.from_recipe(tools.load_json("configs/graphs/complete_signed_7.json"))
base_config = tools.load_json("configs/solvers/qaoa_cpu_checkpoint.json")
root = Path(tempfile.mkdtemp(prefix="qa_a10_notebook_"))
checkpoint_config = deepcopy(base_config)
checkpoint_config["checkpoint"]["path"] = str(root)
checkpoint_config["output"] = {"report": 0, "logs": 0, "color": "never"}
print("store:", root)
""")

md("""## A. Непрерывный запуск и пауза

Непрерывный и паузный запуски используют одну и ту же конфигурацию и seed.
Ожидаемые training shots: 320 после двух итераций, held readout: 64.
""")

code("""continuous_config = deepcopy(checkpoint_config)
continuous_config.pop("checkpoint", None)
continuous = QAOACPU(continuous_config)
continuous_solution = continuous.solve(graph)
print("continuous:", continuous.status, continuous.resources["shots_used"], continuous.cut_value)

paused = QAOACPU(checkpoint_config)
paused_solution = paused.solve(graph, control=RunControl(pause_after={"optimizer_iterations": 2}))
print("paused:", paused.status, paused.termination_reason)
print("resources:", paused.resources)
print("checkpoint:", paused.last_checkpoint.path)
info = inspect_checkpoint(paused.last_checkpoint.path)
print("phase:", info["phase"], "next:", info["next_operation"])
assert paused.status == "paused"
assert paused.resources["known_spent"] == 320
assert paused.resources["readout_reserved"] == 64
""")

md("""## B. Продолжение в новом объекте

`restore()` не запускает solver. Только `resume()` продолжает canonical run.
""")

code("""continued = restore(paused.last_checkpoint.path)
final_solution = continued.resume()
print("resumed:", continued.status, continued.resources["shots_used"], continued.cut_value)
print("same run id:", continued.run_id == paused.run_id)
print("same solution:", final_solution == continuous_solution)
print("max angle error:", float(np.max(np.abs(continued.optimal_angles - continuous.optimal_angles))))
assert final_solution == continuous_solution
assert np.allclose(continued.optimal_angles, continuous.optimal_angles)
assert continued.resources["objective_evaluations"] == continuous.resources["objective_evaluations"]
assert continued.resources["shots_used"] == continuous.resources["shots_used"] == 704
""")

md("""## C. Незавершённая попытка и консервативный charge

BEGIN без COMMIT/RESOLVE означает, что измерение могло состояться. Журнал
списывает upper bound один раз и закрывает операцию. Это не выдаётся за
доказанное число физических shots.
""")

code("""root_b = Path(tempfile.mkdtemp(prefix="qa_a10_b_"))
config_b = deepcopy(checkpoint_config)
config_b["checkpoint"]["path"] = str(root_b)
solver_b = QAOACPU(config_b)
solver_b.solve(graph, control=RunControl(pause_after={"optimizer_iterations": 2}))
run_dir = Path(solver_b.last_checkpoint.path).parent
journal = ResourceJournal(run_dir)
journal.begin(f"{solver_b.run_id}:iteration:3", attempt_id="crash-demo", upper_bound=160, kind="optimizer_iteration")
recovered = restore(solver_b.last_checkpoint.path)
recovered.resume()
print("recovered:", recovered.status, recovered.resources)
assert recovered.resources["known_spent"] == 544
assert recovered.resources["uncertain_upper_charge"] == 160
assert recovered.resources["total_charged"] == 704
""")

md("""## D. Capability matrix

Неподдержанный компонент отклоняется явно; он не превращается в фиктивный
частичный checkpoint.
""")

code("""table = capabilities()
for name in ("qaoa_cpu", "qaoa_gpu", "rqaoa_cpu", "ls_rqaoa_cpu", "qaoa_in_qaoa_cpu", "exact_cpu", "gw_cpu", "simulated_annealing_cpu"):
    item = table[name]
    print(f"{name:24s} supported={item['supports_checkpoint']!s:5s} boundaries={item['boundaries']}")
""")

md("""## E. Размер поколений и бюджеты

Checkpoint содержит manifest, state.json и четыре числовых `.npy` sidecar для
`theta/moment/variance/best_theta`. Большой statevector по умолчанию не
сохраняется.
""")

code("""run_dir = Path(continued.last_checkpoint.path).parent
gen_sizes = []
for gen in sorted(run_dir.glob("gen_*")):
    size = sum(p.stat().st_size for p in gen.iterdir() if p.is_file())
    gen_sizes.append((gen.name, size))
print("generations:", gen_sizes)
print("total bytes:", sum(v for _, v in gen_sizes))
""")

md("""## F. График lineage и расхода

График строится по сохранённым manifests; solver не вызывается.
""")

code("""import json
manifest_rows = []
for gen in sorted(run_dir.glob("gen_*")):
    data = json.loads((gen / "manifest.json").read_text(encoding="utf-8"))
    manifest_rows.append((data["generation"], data.get("phase"), data.get("journal_cutoff", {}).get("known_spent", 0)))
fig, ax = plt.subplots(figsize=(9, 4))
xs = [row[0] for row in manifest_rows]
ys = [row[2] for row in manifest_rows]
ax.step(xs, ys, where="post", marker="o", color="#2563eb")
ax.set_xlabel("generation")
ax.set_ylabel("known training shots")
ax.set_title("Checkpoint lineage and accepted resource prefix")
ax.grid(alpha=0.25)
fig.tight_layout()
figure_path = Path("docs/notebook_previews/11_checkpoints_and_resume.png")
figure_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(figure_path, dpi=140, facecolor="white")
plt.close(fig)
print("saved:", figure_path)
""")

md("""## Итог

- `solve()` всегда новый запуск.
- pause → inspect → restore → resume продолжает тот же run.
- После двух итераций: 320 training shots, 64 held, 320 available.
- Completed readout не повторяет M измерений.
- При аварии uncertain charge не равен нулю и не удваивается при reconciliation.
- GPU/RQAOA/LS/hierarchy checkpoint в a10 явно не заявлены и требуют
  соответствующих codecs.
""")

nb = nbf.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3.12 (QA_max_cut_lib)", "language": "python", "name": "qa-max-cut-lib"}, "language_info": {"name": "python", "version": "3.12"}})
Path("notebooks/11_checkpoints_and_resume.ipynb").write_text(nbformat.writes(nb), encoding="utf-8")
print("wrote notebooks/11_checkpoints_and_resume.ipynb")
