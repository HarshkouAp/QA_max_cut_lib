import json
from itertools import combinations
from QA_max_cut_lib import WeightedGraph
from QA_max_cut_lib.solvers.incremental_ls import IncrementalLSRQAOACPU

if __name__ == "__main__":
    labels = [-7, 10, 30, 80, 101, 205]
    pairs = list(combinations(labels, 2))
    graph = WeightedGraph([(u, v, (2 * i + 1) / 8) for i, (u, v) in enumerate(pairs)], nodes=labels)
    outputs = []
    for threads in (1, 2):
        config = json.load(open("configs/solvers/incremental_ls_keep.json"))
        config["output"] = {"report": 0}
        config["num_treads"] = threads
        solver = IncrementalLSRQAOACPU(config)
        solution = solver.solve(graph, num_treads=threads)
        outputs.append((solution, solver.result["selection_history"], solver.result["final_bank"], solver.cut_value))
        print(threads, solver.cut_value, solver.result["resources"]["shots_used"])
    print("same solution", outputs[0][0] == outputs[1][0])
    print("same selections", outputs[0][1] == outputs[1][1])
    print("same bank", outputs[0][2] == outputs[1][2])
