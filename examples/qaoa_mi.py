"""Passive QAOA mutual-information example."""
from __future__ import annotations
import json
from pathlib import Path
from QA_max_cut_lib.generators.graphs import GraphFactory
from QA_max_cut_lib.laboratory import QAOAWithMIObserverCPU
ROOT=Path(__file__).resolve().parents[1]
def main():
    graph=GraphFactory.from_recipe(json.loads((ROOT/'configs/graphs/block_mean_field_6.json').read_text()))
    config=json.loads((ROOT/'configs/solvers/qaoa_mi_cpu.json').read_text()); config['output']['report']=0
    solver=QAOAWithMIObserverCPU(config); solution=solver.solve(graph)
    print('solution:',solution); print('cut:',solver.cut_value)
    print('frames:',solver.result['mi_summary']['snapshot_count'])
    print('shots:',solver.result['resources']['shots_used']); return 0
if __name__=='__main__': raise SystemExit(main())
