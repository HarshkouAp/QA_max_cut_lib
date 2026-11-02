from pathlib import Path
import nbformat
from nbformat import v4 as nbf

cells = []
def md(text): cells.append(nbf.new_markdown_cell(text))
def code(text): cells.append(nbf.new_code_cell(text))

md("""# Серии экспериментов и ParallelismProfiler

Учебный CPU-ноутбук этапа 09. Он показывает plan → runner → exact-эталон →
profile полного малого grid → глобальную рекомендацию → графики. Все
результаты читаются из сохранённых records; профиль не запускает solver
повторно при построении графика.
""")

code("""from pathlib import Path
import json
import os
import tempfile

ROOT = Path.cwd()
if not (ROOT / "experiments").exists():
    ROOT = ROOT.parent
os.chdir(ROOT)

from QA_max_cut_lib import BenchmarkSpec, ExperimentRunner, ParallelismProfiler, tools

spec = tools.load_json("experiments/benchmark_small.json")
plan = tools.plan_experiments(spec)
print(plan.summary)
print(json.dumps(plan.counts, ensure_ascii=False, indent=2))
assert plan.counts["cases"] == 28
assert plan.counts["trials"] == 56
""")

md("""## Малая серия

`benchmark_small.json` содержит два графа (8 и 10 вершин), четыре mean-конфигурации,
восемь cvar-конфигураций и две SA-конфигурации. Exact maximum считается отдельно.
""")

code("""run_spec = json.loads(Path("experiments/benchmark_small.json").read_text(encoding="utf-8"))
root = Path(tempfile.mkdtemp(prefix="qa_stage09_notebook_"))
run_spec["save"] = {
    "path": str(root),
    "filename": "small_comparison",
    "makedir": True,
    "savedir": True,
    "atts": ["trials", "references", "summary"],
    "save_original_data": [],
}
series = ExperimentRunner(run_spec).run()
print(series.summary)
print(json.dumps(series.analysis["by_solver"], ensure_ascii=False, indent=2)[:1600])
assert series.analysis["planned"] == 56
assert series.analysis["valid"] == 56
assert series.analysis["failed"] == 0
assert len(series.references) == 2
""")

md("""## Точный эталон и качество

ReferencePolicy сохраняет exact maximum отдельным record. `optimality_gap`
считается как `C_star - C`; `ratio` не подменяет эту разность.
""")

code("""for ref in series.references:
    print(ref["kind"], ref["status"], ref["cut_value"], ref["proof"])
assert all(ref["proof"] == "exhaustive" for ref in series.references)

first = series.analysis["trials"][0]
print("first gap:", first["metrics"]["optimality_gap"])
print("q_range requires both extrema:", first["metrics"]["q_range_reason"])
""")

md("""## Параллельный профиль

Профиль использует тот же spec, один выбранный seed и полный grid всех cases.
Числа ниже — реальные измерения этой машины; они не являются универсальным
бенчмарком.
""")

code("""profile_cfg = json.loads(Path("tools/parallelism_sweep.json").read_text(encoding="utf-8"))
profile_cfg.pop("save", None)
profile = ParallelismProfiler(run_spec, profile_cfg).run()
print(profile.summary)
print("recommendation:", profile.analysis["recommendation"])
print("fit:", profile.analysis["fit_amdahl"])
assert profile.plan.points == 84
assert len(profile.measured_records) == 252
assert len(profile.warmup_records) == 84
""")

md("""## График и автономный HTML

Профиль уже сохранён в памяти; график читает только records/analysis.
""")

code("""out = root / "figures"
fig = tools.plot_parallelism(profile, {
    "style": {"figsize": [15, 8]},
    "save": {"path": str(out), "filename": "parallelism", "format": "png", "DPI": 100, "makedir": True},
})
print("figure:", out / "parallelism_parallelism.png")
html_files = tools.parallelism_html(profile, {
    "style": {"figsize": [12, 7]},
    "save": {"path": str(out), "filename": "parallelism_report", "makedir": True},
})
print("html:", html_files.primary)
""")

md("""## Глобальная рекомендация не является голосованием локальных минимумов

Синтетический контроль D: A хорош при n=4, B при n=1, но общий минимум при n=2.
""")

code("""curve = [
    {"n": 1, "total_seconds": 26.0, "complete": True},
    {"n": 2, "total_seconds": 24.0, "complete": True},
    {"n": 4, "total_seconds": 29.0, "complete": True},
]
rec = tools.recommend_total_curve(curve, tolerance_rel=0.03)
print(rec)
assert rec["raw_argmin"] == 2 and rec["recommended_num_treads"] == 2
""")

md("""## Итог

- Полный product: 28 cases, 56 scientific trials, 2 exact references.
- Профиль: 84 case/candidate points, 84 warmup + 252 measured calls.
- Рекомендация учитывает весь набор cases и не выдаёт failed/missing за ноль.
- Графики и HTML используют сохранённые records и не вызывают solver.
- GPU-профиль в этом CPU-ноутбуке не запускается; аппаратная часть отмечена
  отдельно в отчёте.
""")

nb = nbf.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3.12 (QA_max_cut_lib)", "language": "python", "name": "qa-max-cut-lib"}, "language_info": {"name": "python", "version": "3.12"}})
Path("notebooks/10_experiments_and_parallelism.ipynb").write_text(nbformat.writes(nb), encoding="utf-8")
print("wrote notebook 10")
