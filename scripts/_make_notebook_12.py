from pathlib import Path
import nbformat
from nbformat import v4 as nbf

cells = []
def md(text): cells.append(nbf.new_markdown_cell(text))
def code(text): cells.append(nbf.new_code_cell(text))

md("""# Laboratory, warm-start и первый перенос

Этот ноутбук показывает:
- полный каталог исходного архива и его статусы;
- source aliases и отложенные варианты;
- seed spins → q → product state;
- два разных warm-start миксера;
- полный запуск с бюджетом 544;
- checkpoint/resume;
- реальный classical source;
- малую контролируемую серию H.

Основной пример использует только NumPy и `provided_spins`, поэтому
работает в CPU-профиле без CVXPY и CUDA.
""")

code("""from pathlib import Path
import copy
import json
import os
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if not (ROOT / "configs").exists():
    ROOT = ROOT.parent
os.chdir(ROOT)

from QA_max_cut_lib import GraphFactory, RunControl, WeightedGraph, restore, tools
from QA_max_cut_lib.laboratory import WarmStartQAOACPU, catalog_summary, get_file, get_variant, list_variants
from QA_max_cut_lib.solvers.warm_start import product_state, state_at_warm_angles
from QA_max_cut_lib.solvers.qaoa_state import cut_diagonal

print(catalog_summary())
for card in list_variants():
    print(card["variant_id"], card["runnable"], card["status"]["implementation"])
assert len(list_variants()) >= 6
""")

md("""## 1. Inventory и aliases

Все 89 Python-членов архива имеют file record. Пара `Diplom/Mk_5.py` и
`Laboratory/LS_RQAOA_fast_5.py` имеет одинаковый AST; это provisional
equivalence, а не доказательство идентичной trajectory.
""")

code("""left = get_file("Diplom/Mk_5.py")
right = get_file("Laboratory/LS_RQAOA_fast_5.py")
print(left["path"], left["sha256"])
print(right["path"], right["sha256"])
print("same AST:", left["ast_sha256"] == right["ast_sha256"])
print("same bytes:", left["sha256"] == right["sha256"])
deferred = get_variant("warm_start_rqaoa")
print("WS_RQAOA:", deferred["status"])
assert left["ast_sha256"] == right["ast_sha256"]
assert left["sha256"] != right["sha256"]
""")

md("""## 2. Спины, вероятности и product state A

Узлы 10,30, ребро веса 2, seed `(+1,-1)`, epsilon=1/4. Ожидаемые
вероятности: `(3/16, 1/16, 9/16, 3/16)`, initial expected C=5/4.
""")

code("""graph = WeightedGraph([(10, 30, 2)], nodes=[10, 30])
q_raw = np.array([0.0, 1.0])
q = np.array([0.25, 0.75])
state = product_state(q)
probabilities = np.abs(state) ** 2
cuts = cut_diagonal(graph)
print("amplitudes:", state)
print("probabilities:", probabilities)
print("initial C:", float(cuts @ probabilities))
print("source cut:", graph.cut_value({10: 1, 30: -1}))
assert np.allclose(probabilities, [3/16, 1/16, 9/16, 3/16])
assert np.isclose(cuts @ probabilities, 1.25)
""")

md("""## 3. Два warm-start миксера

При epsilon=1/2 `rounded_modified` должен совпасть с обычным X-миксером
при том же beta, а `warm_aligned` — при `-beta`. Здесь сравниваются
не только энергии, но и полные распределения.
""")

code("""from QA_max_cut_lib import QAOACPU

params = {"layers": 1, "max_vertices": 8, "dtype": "complex128"}
base = {
    "type": "qaoa_cpu", "seed": 1, "parameters": params,
    "objective": {"type": "mean_cut"},
    "estimation": {"mode": "exact"},
    "optimizer": {"type": "adam", "parameters": {"max_steps": 0}},
    "initialization": {"type": "fixed_angles", "parameters": {"values": [0.4, 0.7]}},
    "readout": {"mode": "most_probable"}, "limits": {"shots_total": 0},
    "collect": {"probabilities": True}, "output": {"report": 0, "logs": 0},
}
warm = copy.deepcopy(base)
warm["type"] = "laboratory.warm_start_qaoa_cpu"
warm["warm_start"] = {
    "epsilon": 0.5,
    "source": {"type": "provided_spins", "parameters": {"entries": [{"node": 10, "spin": 1}, {"node": 30, "spin": -1}]}},
    "mixer": {"type": "rounded_modified"},
}
warm["parameters"] = params
warm_solver = WarmStartQAOACPU(warm)
warm_solver.solve(graph)
standard_solver = QAOACPU(base)
standard_solver.solve(graph)
print("max probability difference:", float(np.max(np.abs(warm_solver.probabilities - standard_solver.probabilities))))
assert np.allclose(warm_solver.probabilities, standard_solver.probabilities, atol=1e-12)
""")

md("""## 4. Полный provided-spin запуск

p=1, три ADAM-итерации, 32 shots/evaluation, 64 readout: полный бюджет
`3*5*32 + 64 = 544`. Source cut и expected product cut — разные поля.
""")

code("""config = tools.load_json("configs/solvers/warm_start_qaoa_cpu_provided.json")
config = copy.deepcopy(config)
config["checkpoint"]["path"] = str(Path(tempfile.mkdtemp(prefix="qa_a11_catalog_")) / "run")
config["output"] = {"report": 0, "logs": 0, "color": "never"}
g4 = GraphFactory.from_recipe(tools.load_json("configs/graphs/warm_start_4.json"))
provided = WarmStartQAOACPU(config)
solution = provided.solve(g4)
print("solution:", solution)
print("cut:", provided.cut_value, "source_cut:", provided.source_cut)
print("initial expected cut:", provided.initial_expected_cut)
print("mixer:", provided.mixer_id)
print("resources:", provided.resources)
assert provided.resources["shots_used"] == 544
assert provided.source_cut is not None
""")

md("""## 5. Pause → restore → resume

Пауза после одной завершённой итерации: 160 training shots и 64 held.
Продолжение в новом объекте получает тот же canonical run и полный бюджет.
""")

code("""pause_config = copy.deepcopy(config)
pause_config["checkpoint"] = {
    "enabled": True,
    "path": str(Path(tempfile.mkdtemp(prefix="qa_a11_pause_")) / "run"),
    "every": {"optimizer_iterations": 1},
}
paused = WarmStartQAOACPU(pause_config)
paused.solve(g4, control=RunControl(pause_after={"optimizer_iterations": 1}))
print("paused:", paused.status, paused.resources)
path = paused.last_checkpoint.path
continued = restore(path)
final = continued.resume()
print("final:", final, continued.resources)
print("same angles:", np.allclose(continued.optimal_angles, provided.optimal_angles))
assert paused.status == "paused"
assert continued.status == "completed"
assert continued.resources["shots_used"] == 544
""")

md("""## 6. Реальный classical source

GW и SA запускаются лениво только при выборе source. Их reads/hyperplanes
и время остаются отдельной `source_cost`; quantum shots — отдельно.
""")

code("""classical_config = tools.load_json("configs/solvers/warm_start_qaoa_cpu_gw.json")
classical_config = copy.deepcopy(classical_config)
classical_config["output"] = {"report": 0, "logs": 0, "color": "never"}
gw = WarmStartQAOACPU(classical_config)
gw.solve(g4)
print("source:", gw.source_type, "source_cut:", gw.source_cut)
print("source_cost:", gw.source_cost)
print("quantum_shots:", gw.resources["shots_used"])
assert gw.source_cost is not None
assert gw.resources["shots_used"] == 224
""")

md("""## 7. Малая серия H

3 графа × 2 algorithm seeds × 7 конфигураций (base + 2 epsilon × 3 mixer) =
42 запуска. Это проверка бюджета и честного сравнения, не доказательство
преимущества warm-start.
""")

code("""spec = tools.load_json("experiments/benchmark_warm_start.json")
spec.pop("save", None)
series = tools.ExperimentRunner(spec).run()
print(series.summary)
print("planned/valid:", series.analysis["planned"], series.analysis["valid"])
print("quantum shots per run: 544")
print("declared series shots:", 42 * 544)
assert series.analysis["planned"] == 42
assert series.analysis["valid"] == 42
assert series.analysis["failed"] == 0
""")

md("""## 8. Исходное и конечное распределение

Подписи явно разделяют source cut, initial expected cut и конечный cut.
График строится из реальных массивов этого запуска; классический source
повторно не вызывается.
""")

code("""fig, ax = plt.subplots(figsize=(10, 4))
width = 0.36
index = np.arange(len(provided.probabilities))
ax.bar(index - width / 2, np.abs(product_state(provided.warm_start_data["q_clipped"])) ** 2, width, label="initial product")
ax.bar(index + width / 2, provided.probabilities, width, label="final QAOA")
ax.set_xlabel("basis index (little-endian node order)")
ax.set_ylabel("probability")
ax.set_title("Warm start: source cut=%.1f, initial E[C]=%.3f, final C=%.3f" % (provided.source_cut, provided.initial_expected_cut, provided.cut_value))
ax.set_xticks(index)
ax.grid(axis="y", alpha=0.25)
ax.legend()
fig.tight_layout()
figure_path = Path("docs/notebook_previews/12_laboratory_catalog_and_warm_start.png")
figure_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(figure_path, dpi=140, facecolor="white")
plt.close(fig)
print("saved:", figure_path)
""")

md("""## 9. Сравнение H

Таблица ниже — solver-only качество и quantum shots. `source_cost` для
GW/SA не приписывается конфигурациям без classical source.
""")

code("""rows = []
for name, data in series.analysis["by_solver"].items():
    rows.append((name, data["cut"]["mean"], data["success"], data["valid"]))
for row in sorted(rows):
    print("%-24s mean_cut=%.3f success=%.3f n=%d" % row)
assert len(rows) == 7
""")

md("""## Итог

- Каталог покрывает 89 исходных файлов; deferred-карточки видны.
- Product state A воспроизводит `(3/16,1/16,9/16,3/16)` и initial C=5/4.
- `rounded_modified` и `warm_aligned` имеют разные знаки beta.
- Полный provided запуск использует 544 shots.
- Pause/resume продолжает canonical run.
- GW source имеет отдельную стоимость preprocessing.
- H — 42 реальных запуска, не заявление о превосходстве warm-start.
""")

nb = nbf.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3.12 (QA_max_cut_lib)", "language": "python", "name": "qa-max-cut-lib"}, "language_info": {"name": "python", "version": "3.12"}})
Path("notebooks/12_laboratory_catalog_and_warm_start.ipynb").write_text(nbformat.writes(nb), encoding="utf-8")
print("wrote notebook 12")
