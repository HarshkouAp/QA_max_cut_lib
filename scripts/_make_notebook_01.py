import os
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]; os.chdir(ROOT)
nb=nbf.v4.new_notebook()
cells=[
nbf.v4.new_markdown_cell("# 01. Full guide (CPU slice)\n\nПубличные четыре раздела, lifecycle, exact/QAOA/RQAOA, генераторы/plans, save/checkpoint. GPU и незакрытые Laboratory controllers отмечены отдельно."),
nbf.v4.new_code_cell("""import json, tempfile
from pathlib import Path
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from QA_max_cut_lib import WeightedGraph, ExactCPU, QAOACPU, RQAOACPU, RunControl, restore
from QA_max_cut_lib.generators import GraphFactory
from QA_max_cut_lib.generators.plans import build_overlap_edge_partition, build_connected_bfs_partition
from QA_max_cut_lib.tools.partition_analysis import analyze_plan
ROOT=Path.cwd()
if not (ROOT/'configs').exists(): ROOT=ROOT.parent"""),
nbf.v4.new_markdown_cell("## Exact reference"),
nbf.v4.new_code_cell("""graph=WeightedGraph([(10,30,2),(30,80,-1),(10,80,3)],nodes=[10,30,80,101])
exact=ExactCPU(); sol=exact.solve(graph,report=0); print(sol, exact.cut_value)"""),
nbf.v4.new_markdown_cell("## CPU QAOA criteria/readout"),
nbf.v4.new_code_cell("""config=json.loads((ROOT/'configs/solvers/qaoa_cpu.json').read_text())
config['parameters']['layers']=2; config['optimizer']['parameters']['max_steps']=3
config['initialization']={'type':'fixed_angles','parameters':{'values':[0.3,0.2,0.1,0.4]}}
config['estimation']={'mode':'exact'}; config['readout']={'mode':'most_probable'}
config['limits']={'shots_total':0}; config['output']={'report':0}
q=QAOACPU(config); s=q.solve(graph); print(s,q.cut_value,q.resources)"""),
nbf.v4.new_markdown_cell("## Recursive RQAOA (small residual)"),
nbf.v4.new_code_cell("""rconfig=json.loads((ROOT/'configs/solvers/rqaoa_cpu.json').read_text())
rconfig['parameters']={'finish_at':1,'max_reductions':1}; rconfig['limits']={'shots_total':64}
rconfig['output']={'report':0}; rconfig['correlation_source']['parameters']['max_vertices']=4
rconfig['correlation_source']['estimation']={'mode':'exact'}; rconfig['correlation_source']['correlations']={'mode':'exact'}
r=RQAOACPU(rconfig); rs=r.solve(graph); print(rs,r.cut_value,r.reductions_completed)"""),
nbf.v4.new_markdown_cell("## Graph recipe and typed plan"),
nbf.v4.new_code_cell("""g=GraphFactory.from_recipe(json.loads((ROOT/'configs/graphs/connected_path_fill_6.json').read_text()))
p=build_connected_bfs_partition(g,max_size=3,seed=2,merge_singletons=True)
a=analyze_plan(g,p); print(len(g),len(g.edges),p.plan_kind,a['vertex_coverage'],a['edge_coverage'])"""),
nbf.v4.new_markdown_cell("## Edge-subset plan has an explicit edge domain"),
nbf.v4.new_code_cell("""e=build_overlap_edge_partition(g,max_size=3,seed=0)
ea=analyze_plan(g,e); print(e.completion_status,ea['edge_coverage'],ea['edge_exclusive'])
print([(part.nodes,part.edge_domain,len(part.edges)) for part in e.parts[:2]])"""),
nbf.v4.new_markdown_cell("## Checkpoint/resume QAOA"),
nbf.v4.new_code_cell("""checkpoint_config=dict(config); checkpoint_config['checkpoint']={'enabled':True,'path':tempfile.mkdtemp(prefix='qa_ckpt_'),'every':{'optimizer_iterations':1},'durable_resource_journal':True}
ck=QAOACPU(checkpoint_config); ck.solve(graph,control=RunControl(pause_after={'optimizer_iterations':1}))
print('paused',ck.status,ck.phase if hasattr(ck,'phase') else '',ck.last_checkpoint is not None)
resumed=restore(ck.last_checkpoint.path); resumed.resume(); print('resumed',resumed.status,resumed.reductions_completed if hasattr(resumed,'reductions_completed') else '')"""),
nbf.v4.new_markdown_cell("## Release blockers\n\nBatch/backtracking/hierarchical controllers and full 89-source closure are not claimed here; see `RELEASE_READINESS.md`."),
]
nb['cells']=cells; nb['metadata']['kernelspec']={'name':'python3','display_name':'Python 3'}
out=ROOT/'notebooks'/'01_full_guide.ipynb'; nbf.write(nb,out); print(out)
