import os
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
nb = nbf.v4.new_notebook()
cells = [
    nbf.v4.new_markdown_cell("# 18. Incremental LS-RQAOA и MK-стратегии\n\nЭпоха, банк, age, четыре score и частичное обновление."),
    nbf.v4.new_code_cell("""import json
from itertools import combinations
from pathlib import Path
import numpy as np
from QA_max_cut_lib import WeightedGraph
from QA_max_cut_lib.generators.incremental import survivor_neighbor_cover
from QA_max_cut_lib.solvers.incremental_ls import IncrementalLSRQAOACPU
ROOT = Path.cwd()
if not (ROOT / 'configs').exists(): ROOT = ROOT.parent
labels = [-7,10,30,80,101,205]
pairs = list(combinations(labels, 2))
graph = WeightedGraph([(u,v,(2*i+1)/8) for i,(u,v) in enumerate(pairs)], nodes=labels)"""),
    nbf.v4.new_markdown_cell("## Одна эпоха: mean, zero и missing"),
    nbf.v4.new_code_cell("""config = json.loads((ROOT/'configs/solvers/incremental_ls_current.json').read_text())
config['output'] = {'report': 0}
config['limits']['max_reductions'] = 1
solver = IncrementalLSRQAOACPU(config); solver.solve(graph, num_treads=1)
e = solver.result['epoch_history'][0]
print('fresh pairs:', len(e['fresh_pairs']), 'allowed:', e['allowed_pairs'])
print('selection:', e['selection'])"""),
    nbf.v4.new_markdown_cell("## Частичное обновление вокруг retained center"),
    nbf.v4.new_code_cell("""center = solver.result['selection_history'][0]['pair'][1]
cover = survivor_neighbor_cover(graph, center, max_size=3, growth_metric='count', version=1)
for task in cover: print(task['task_id'], task['nodes'])"""),
    nbf.v4.new_markdown_cell("## Полный run с bank keep_unobserved"),
    nbf.v4.new_code_cell("""config = json.loads((ROOT/'configs/solvers/incremental_ls_keep.json').read_text())
config['output'] = {'report': 0}
solver = IncrementalLSRQAOACPU(config); solution = solver.solve(graph, num_treads=1)
print('cut', solver.cut_value, 'reductions', solver.result['reductions_completed'])
print('tasks', solver.result['resources']['local_tasks_total'], 'shots', solver.result['resources']['shots_used'])
print('selections', [e['selection']['pair'] for e in solver.result['epoch_history']])"""),
    nbf.v4.new_markdown_cell("## Стоимость и ограничения"),
    nbf.v4.new_code_cell("""print(solver.result['resources'])
print('bank entries:', len(solver.result['final_bank']))
print('checkpoint:', solver.supports_checkpoint)"""),
    nbf.v4.new_markdown_cell("## Пять score: значения различаются\n\nScore не является физическим c и не является вероятностью."),
    nbf.v4.new_code_cell("""# Fixture B: scores are intentionally different for four rules.
# See tests/test_incremental_ls.py for the exact winner assertions.""")
]
nb['cells'] = cells
nb['metadata']['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3'}
out = ROOT/'notebooks'/'18_incremental_ls_mk_strategies.ipynb'
nbf.write(nb, out)
print(out)
