"""Run the 48 small generation/plan/analysis cases from stage 18."""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

from QA_max_cut_lib import WeightedGraph
from QA_max_cut_lib.generators import GraphFactory
from QA_max_cut_lib.generators.plans import (
    build_connected_bfs_partition,
    build_edge_coverage_scan,
    build_overlap_edge_partition,
    build_random_partition_plan,
)
from QA_max_cut_lib.tools.partition_analysis import analyze_plan

LABELS = [-7, 0, 10, 30, 80, 101, 205, 1000]


def weighted(edges, labels):
    return WeightedGraph(
        [(u, v, ((-1) ** index) * (2 * index + 1) / 8) for index, (u, v) in enumerate(edges)],
        nodes=labels,
    )


def graphs():
    complete = weighted(list(combinations(LABELS, 2)), LABELS)
    cycle_edges = [(LABELS[i], LABELS[(i + 1) % len(LABELS)]) for i in range(len(LABELS))]
    cycle = weighted(cycle_edges, LABELS)
    star = weighted([(-7, node) for node in LABELS[1:]], LABELS)
    sparse = weighted([(-7, 0), (10, 30), (80, 101)], LABELS)
    fixture = WeightedGraph(
        [(-7, 10, 2), (10, 30, -3), (-7, 30, 5)],
        nodes=[-7, 10, 30, 80],
    )
    recipe = json.loads(Path("configs/graphs/connected_path_fill_6.json").read_text())
    recipe["parameters"]["n_vertices"] = 8
    recipe["parameters"]["nodes"] = LABELS
    recipe["parameters"]["n_edges"] = 12
    recipe["seed"] = 18
    recipe["weights"]["seed"] = 118
    filled = GraphFactory.from_recipe(recipe)
    return [
        ("complete", complete),
        ("cycle", cycle),
        ("star", star),
        ("three_edges", sparse),
        ("fixture", fixture),
        ("path_fill", filled),
    ]


def builders(seed):
    return {
        "random": lambda graph: build_random_partition_plan(graph, max_size=3, seed=seed),
        "bfs": lambda graph: build_connected_bfs_partition(graph, max_size=3, seed=seed, merge_singletons=True),
        "scan": lambda graph: build_edge_coverage_scan(
            graph,
            target_edge_coverage=1.0,
            max_tasks=256,
            max_episodes=32,
            cover_isolates=True,
            anchor_policy="random",
            local_builder={"max_size": 3, "goal": "cover_neighbors", "growth_metric": "count"},
            seed=seed,
        ),
        "overlap": lambda graph: build_overlap_edge_partition(
            graph, max_size=3, seed=seed, growth_metric="abs_weight",
            lambda_internal=1.0, lambda_overlap=1.0, seed_strategy="max_overlap",
        ),
    }


def main():
    rows = []
    for graph_name, graph in graphs():
        for seed in (0, 1):
            for builder_name, builder in builders(seed).items():
                plan = builder(graph)
                analysis = analyze_plan(graph, plan, seed=seed)
                rows.append(
                    {
                        "graph": graph_name,
                        "seed": seed,
                        "builder": builder_name,
                        "plan_kind": plan.plan_kind,
                        "completion": plan.completion_status,
                        "parts": len(plan.parts),
                        "max_part_size": max(part.size for part in plan.parts) if plan.parts else 0,
                        "vertex_coverage": analysis["vertex_coverage"],
                        "edge_coverage": analysis["edge_coverage"],
                        "pair_coverage": analysis["pair_coverage"],
                        "abs_weight_coverage": analysis["abs_weight_coverage"],
                        "edge_multiplicity_max": analysis["repeated_edge_assignments"],
                        "mean_hop": analysis["hop"]["all_pairs_mean_hop_distance"],
                    }
                )
    out = {
        "trials": len(rows),
        "quantum_shots": 0,
        "rows": rows,
    }
    path = Path("results/generators_series/stage18_48_cases.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print("trials", len(rows), "path", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
