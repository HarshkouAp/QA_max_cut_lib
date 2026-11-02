import json
from itertools import combinations
from QA_max_cut_lib import WeightedGraph
from QA_max_cut_lib.solvers.incremental_ls import IncrementalLSRQAOACPU

if __name__ == "__main__":
    labels = [-7, 10, 30, 80, 101, 205]
    pairs = list(combinations(labels, 2))
    graph = WeightedGraph([(u, v, (2 * i + 1) / 8) for i, (u, v) in enumerate(pairs)], nodes=labels)
    config = json.load(open("configs/solvers/incremental_ls_keep.json"))
    config["output"] = {"report": 0}
    config["num_treads"] = 2
    solver = IncrementalLSRQAOACPU(config)
    solver.solve(graph, num_treads=2)
    print(solver.result["status"], solver.result["resources"]["shots_used"], solver.result["resources"]["num_treads_applied"], solver.cut_value)
