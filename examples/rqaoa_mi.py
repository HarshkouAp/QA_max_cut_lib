"""Passive RQAOA mutual-information example."""
from __future__ import annotations
import json
from pathlib import Path
from QA_max_cut_lib.generators.graphs import GraphFactory
from QA_max_cut_lib.laboratory import RQAOAWithMIObserverCPU
ROOT=Path(__file__).resolve().parents[1]
def main():
    graph=GraphFactory.from_recipe(json.loads((ROOT/'configs/graphs/warm_start_4.json').read_text()))
    config=json.loads((ROOT/'configs/solvers/rqaoa_mi_cpu.json').read_text()); config['output']['report']=0
    solver=RQAOAWithMIObserverCPU(config); solution=solver.solve(graph)
    print('solution:',solution); print('cut:',solver.cut_value)
    print('frames:',len(solver.result['mi_frames'])); print('shots:',solver.result['resources']['shots_used']); return 0
if __name__=='__main__': raise SystemExit(main())
