import os
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
nb = nbf.v4.new_notebook()
cells = [
    nbf.v4.new_markdown_cell("# 19. Генераторы, typed plans и coverage\n\nRecipe → graph → plan → независимый анализ."),
    nbf.v4.new_code_cell("""import json
from pathlib import Path
import numpy as np
from QA_max_cut_lib import WeightedGraph
from QA_max_cut_lib.generators import GraphFactory, resolve_bench297
from QA_max_cut_lib.generators.plans import *
from QA_max_cut_lib.generators.structured import *
from QA_max_cut_lib.tools.partition_analysis import analyze_plan, hop_metrics
ROOT = Path.cwd()
if not (ROOT/'configs').exists(): ROOT = ROOT.parent
recipe = json.loads((ROOT/'configs/graphs/connected_path_fill_6.json').read_text())
graph = GraphFactory.from_recipe(recipe)
print(len(graph), len(graph.edges), graph.snapshot()['recipe']['resolved']['topology_hash'][:12])"""),
    nbf.v4.new_markdown_cell("## Density G(n,m) отличается от Bernoulli G(n,p)"),
    nbf.v4.new_code_cell("""g = GraphFactory.from_recipe({'schema_version':2,'type':'gnm_density_v1','seed':1,'parameters':{'n_vertices':4,'density':0.5}})
print('density edges:', len(g.edges), '(exact floor(0.5*6))')"""),
    nbf.v4.new_markdown_cell("## Benchmark297 decoder"),
    nbf.v4.new_code_cell("""for q in (17, 296): print(q, resolve_bench297(q))"""),
    nbf.v4.new_markdown_cell("## Wishart integer witness"),
    nbf.v4.new_code_cell("""w, record = build_wishart_from_factor(np.array([[1.],[1.],[2.]]), [1,1,-1], labels=[80,-7,10])
print(w.edges); print(record['candidate_cut'], record['stored_graph_optimum_verified'])"""),
    nbf.v4.new_markdown_cell("## Hebbian counterexample"),
    nbf.v4.new_code_cell("""h, hr = build_hebbian_patterns([[1,1,1]]*5 + [[1,1,-1]])
print(h.edges); print(hr['candidates'][-1])"""),
    nbf.v4.new_markdown_cell("## Edge partition: assigned domain не induced"),
    nbf.v4.new_code_cell("""fixture = WeightedGraph([(-7,10,2),(10,30,-3),(-7,30,5)], nodes=[-7,10,30,80])
plan = build_overlap_edge_partition(fixture, max_size=3, seed=0)
analysis = analyze_plan(fixture, plan)
print(plan.plan_kind, analysis['edge_coverage'], analysis['pair_coverage'])
for part in plan.parts: print(part.nodes, part.edge_domain, len(part.edges))"""),
    nbf.v4.new_markdown_cell("## Hop metrics"),
    nbf.v4.new_code_cell("""path = WeightedGraph([(0,1,7),(1,2,-4)], nodes=[0,1,2])
print(hop_metrics(path))"""),
    nbf.v4.new_markdown_cell("## 48-case series"),
    nbf.v4.new_code_cell("""series = json.loads((ROOT/'results/generators_series/stage18_48_cases.json').read_text())
print(series['trials'], series['quantum_shots'])
from collections import Counter
print(Counter((row['builder'], row['completion']) for row in series['rows']))"""),
]
nb['cells'] = cells
nb['metadata']['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3'}
out = ROOT/'notebooks'/'19_generators_and_partition_coverage.ipynb'
nbf.write(nb, out)
print(out)
