from pathlib import Path
import sys
from QA_max_cut_lib import GraphFactory, RunControl, tools
from QA_max_cut_lib.laboratory import WarmStartQAOACPU

if len(sys.argv) == 2:
    graph = GraphFactory.from_recipe(tools.load_json("configs/graphs/warm_start_4.json"))
    config = tools.load_json("configs/solvers/warm_start_qaoa_cpu_provided.json")
    config["checkpoint"] = {"enabled": True, "path": "results/checkpoints/process", "every": {"optimizer_iterations": 1}}
    config["output"] = {"report": 0, "logs": 0}
    solver = WarmStartQAOACPU(config)
    solver.solve(graph, control=RunControl(pause_after={"optimizer_iterations": 1}))
    Path(sys.argv[1]).write_text(str(solver.last_checkpoint.path), encoding="utf-8")
    print("paused", solver.status, solver.resources["known_spent"])
else:
    from QA_max_cut_lib import restore
    path = Path(sys.argv[2]).read_text(encoding="utf-8")
    solver = restore(path)
    final = solver.resume()
    print("resumed", solver.status, solver.resources["shots_used"], final)
