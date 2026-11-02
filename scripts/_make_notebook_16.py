import json
import os
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
nb = nbf.v4.new_notebook()
cells = [
    nbf.v4.new_markdown_cell(
        """# 16. Distributed LS-RQAOA

Mean-field episodes на текущем graph, fresh final correlations, одна
редукция и transport/repartition."""
    ),
    nbf.v4.new_code_cell(
        """import json, math
from pathlib import Path
from QA_max_cut_lib import WeightedGraph
from QA_max_cut_lib.laboratory import DistributedLSRQAOACPU
ROOT = Path.cwd()
if not (ROOT / "configs").exists():
    ROOT = ROOT.parent
config = json.loads((ROOT / "configs/solvers/distributed_ls_jacobi_cpu.json").read_text())
config["checkpoint"]["enabled"] = False
config["output"]["report"] = 0
print("schedule:", config["correlation_source"]["mean_field"]["schedule"])
print("transport:", config["messages"]["transport"]["type"])"""
    ),
    nbf.v4.new_markdown_cell(
        """## Fixture C: fresh c, не bank q

Полный двухузловой block, initial means=(1,1), θ=(-π/4, π/8),
max_sweeps=0/exact. Fresh extraction должен дать c=-1."""
    ),
    nbf.v4.new_code_cell(
        """c = json.loads((ROOT / "configs/solvers/distributed_ls_jacobi_cpu.json").read_text())
c["parameters"] = {"finish_at": 1, "max_reductions": 1}
c["partition_builder"] = {"type": "greedy_abs_weight_partition", "parameters": {"max_size": 3, "rounds": 1, "full_graph_when_fits": True}}
c["messages"] = {"initial": {"type": "provided_means", "parameters": {"entries": [{"node": 10, "mean": 1.0}, {"node": 30, "mean": 1.0}]}}, "episode_mean_aggregation": "uniform", "transport": {"type": "carry_survivor_means", "partition": "rebuild", "block_moments": "product_reset", "angles": "restart"}}
s = c["correlation_source"]
s["mean_field"].update({"max_sweeps": 0, "moment_damping": 0.0, "angle_damping": 1.0})
s["mean_field"]["schedule"]["type"] = "jacobi"
s["initialization"]["parameters"]["values"] = [-math.pi / 4, math.pi / 8]
s["local_solver"]["estimation"] = {"mode": "exact"}
s["local_solver"]["optimizer"]["parameters"]["max_steps"] = 0
s["moments"] = {"mode": "exact", "statistics": ["z_all", "zz_internal_edges"], "shared_sample": True}
s["final_correlations"] = {"mode": "exact"}
c["limits"] = {"shots_total": 0, "max_block_updates_total": 4}
c["checkpoint"] = {"enabled": False}
c["collect"] = {"reduction_history": True, "final_correlation_tables": True, "episode_history": True}
c["output"] = {"report": 0, "logs": 0}
small = WeightedGraph([(10, 30, 2)], nodes=[10, 30])
solver = DistributedLSRQAOACPU(c)
solution = solver.solve(small)
print("solution:", solution, "cut:", solver.cut_value)
print("fresh c:", solver.result["final_correlation_tables"])"""
    ),
    nbf.v4.new_markdown_cell(
        """## Основной six-node run

Greedy partitions, один episode на текущий graph, exact finisher при
`finish_at=4`."""
    ),
    nbf.v4.new_code_cell(
        """graph = WeightedGraph([(10,30,5),(10,80,1),(30,80,1),(80,101,2),(101,200,3),(30,200,-2)], nodes=[-7,10,30,80,101,200])
main = DistributedLSRQAOACPU(config)
solution = main.solve(graph)
print("solution:", solution)
print("cut:", main.cut_value)
print("reductions:", main.result["reductions_completed"], "source:", main.result["solution_source"])
print("episodes:", len(main.result["episode_history"] or []), "shots:", main.result["resources"]["shots_used"])"""
    ),
    nbf.v4.new_markdown_cell(
        """## Reducer и transport

Survivor mean сохраняется, q нового product model сбрасывается."""
    ),
    nbf.v4.new_code_cell(
        """from QA_max_cut_lib.core.reduction import GraphReducer
from QA_max_cut_lib.solvers.block_mean_field import build_block_problem
rgraph = WeightedGraph([(10,30,2),(10,80,3),(30,80,5)], nodes=[10,30,80,101])
reducer = GraphReducer(); state = reducer.start(rgraph)
reducer.reduce(state, removed=80, retained=30, sign=-1, correlation=-0.6)
carried = {10: 0.2, 30: -0.4, 101: 0.6}
problem = build_block_problem(state.current_graph(), [10,30], carried)
print("new edges:", state.current_graph().edges)
print("offset:", state.cut_offset)
print("carried q:", carried[10] * carried[30])
print("local J:", problem.couplings)"""
    ),
    nbf.v4.new_markdown_cell(
        """## Граница

Fresh c — физическая local ZZ-оценка. Damped model q и последний raw
receipt не считаются той же величиной. Transport не является exact
conditional projection."""
    ),
]
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = ROOT / "notebooks" / "16_distributed_ls_rqaoa.ipynb"
nbf.write(nb, out)
print(out)
