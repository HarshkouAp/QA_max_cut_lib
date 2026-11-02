"""Recipe, typed plan, and independent coverage example."""
from __future__ import annotations
import json
from pathlib import Path
from QA_max_cut_lib.generators import GraphFactory
from QA_max_cut_lib.generators.plans import build_overlap_edge_partition, build_edge_coverage_scan
from QA_max_cut_lib.tools.partition_analysis import analyze_plan

ROOT = Path(__file__).resolve().parents[1]

def main():
    recipe = json.loads((ROOT / "configs/graphs/connected_path_fill_6.json").read_text())
    graph = GraphFactory.from_recipe(recipe)
    print("graph:", len(graph), "vertices", len(graph.edges), "edges")
    plan = build_overlap_edge_partition(graph, max_size=3, seed=0)
    analysis = analyze_plan(graph, plan)
    print("plan:", plan.plan_kind, plan.completion_status)
    print("coverage: vertices", analysis["vertex_coverage"], "edges", analysis["edge_coverage"])
    scan = build_edge_coverage_scan(graph, max_tasks=8, target_edge_coverage=1.0)
    print("scan:", scan.completion_status, len(scan.parts))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
