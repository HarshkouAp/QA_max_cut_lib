import os
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
nb=nbf.v4.new_notebook()
cells=[
nbf.v4.new_markdown_cell("# 00. Quickstart\n\nCPU quickstart: graph, solve, independent cut check, Tools."),
nbf.v4.new_code_cell("""import json, tempfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from QA_max_cut_lib import WeightedGraph, QAOACPU, ExactCPU, tools
ROOT=Path.cwd()
if not (ROOT/'configs').exists(): ROOT=ROOT.parent
graph=WeightedGraph([(10,30,1.5),(30,80,-0.5),(10,80,2.0)], nodes=[10,30,80,101])
reference=ExactCPU(); ref_solution=reference.solve(graph, report=0); print('reference',ref_solution,reference.cut_value)"""),
nbf.v4.new_markdown_cell("## QAOA configuration and first solve"),
nbf.v4.new_code_cell("""config={'seed':42,'parameters':{'layers':2},'objective':{'type':'mean_cut'},
 'estimation':{'mode':'exact'},'optimizer':{'parameters':{'max_steps':3}},
 'readout':{'mode':'most_probable'},'limits':{'shots_total':0},
 'collect':{'optimizer_history':True,'probabilities':True}}
solver=QAOACPU(config); solution=solver.solve(graph, report=0, logs=0)
print('solution',solution); print('cut',solver.cut_value,'energy',solver.solution_energy)
assert solver.cut_value==graph.cut_value(solution)
assert {int(k): int(v) for k,v in solver.result['solution'].items()}==solution"""),
nbf.v4.new_markdown_cell("## Tools: graph and optimization trace"),
nbf.v4.new_code_cell("""fig=tools.plot_graph_solution(solver); plt.close(fig)
fig=tools.plot_optimization(solver); plt.close(fig)
print('probability entries',len(solver.probabilities))"""),
nbf.v4.new_markdown_cell("## Second solve on a changed graph"),
nbf.v4.new_code_cell("""changed=graph.copy(); changed.set_weight(10,30,0.25)
new_solution=solver.solve(changed, report=0, logs=0)
print('new cut',solver.cut_value); print('previous result unchanged',solver.result['graph']['fingerprint']!=graph.snapshot()['fingerprint'])"""),
nbf.v4.new_markdown_cell("## Save and no-overwrite"),
nbf.v4.new_code_cell("""from QA_max_cut_lib.core.saving import save_result
out=Path(tempfile.mkdtemp(prefix='qa_quickstart_'))
record=solver.last_run
path=save_result(record, {'path':str(out),'filename':'quickstart','atts':['solution'],'save_original_data':[]})
print('saved',path)
try:
 save_result(record, {'path':str(out),'filename':'quickstart','atts':['solution'],'save_original_data':[]})
except Exception as exc:
 print('no-overwrite:',type(exc).__name__)"""),
nbf.v4.new_markdown_cell("## GPU\n\nGPU cells are intentionally not run in this CPU quickstart. See `docs/GPU.md`; G8 remains a release gate."),
]
nb['cells']=cells; nb['metadata']['kernelspec']={'name':'python3','display_name':'Python 3'}
out=ROOT/'notebooks'/'00_quickstart.ipynb'; nbf.write(nb,out); print(out)
