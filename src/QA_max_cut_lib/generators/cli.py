"""Small reproducible-generator CLI.

Examples:
    python -m QA_max_cut_lib.generators.cli resolve recipe.json
    python -m QA_max_cut_lib.generators.cli generate recipe.json --out graph.json
    python -m QA_max_cut_lib.generators.cli bench297 --id 17
    python -m QA_max_cut_lib.generators.cli analyze graph.json plan.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmarks import resolve_bench16, resolve_bench297
from .graphs import GraphFactory
from .plans import SubproblemPlan
from ..tools.partition_analysis import analyze_plan


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dump(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description="QA_max_cut_lib generators")
    sub = parser.add_subparsers(dest="command", required=True)
    resolve = sub.add_parser("resolve")
    resolve.add_argument("recipe")
    generate = sub.add_parser("generate")
    generate.add_argument("recipe")
    generate.add_argument("--out")
    bench16 = sub.add_parser("bench16")
    bench16.add_argument("--id", type=int, required=True)
    bench297 = sub.add_parser("bench297")
    bench297.add_argument("--id", type=int, required=True)
    bench297.add_argument("--base-seed", type=int, default=42)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("graph")
    analyze.add_argument("plan")
    args = parser.parse_args(argv)
    if args.command == "resolve":
        graph = GraphFactory.from_recipe(_load(args.recipe))
        _dump(graph.snapshot()["recipe"])
        return 0
    if args.command == "generate":
        graph = GraphFactory.from_recipe(_load(args.recipe))
        snapshot = graph.snapshot()
        if args.out:
            Path(args.out).write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            _dump(snapshot)
        return 0
    if args.command == "bench16":
        _dump(resolve_bench16(args.id))
        return 0
    if args.command == "bench297":
        _dump(resolve_bench297(args.id, args.base_seed))
        return 0
    if args.command == "analyze":
        from ..core.graph import WeightedGraph

        graph = WeightedGraph.from_snapshot(_load(args.graph))
        plan = SubproblemPlan.from_dict(_load(args.plan))
        _dump(analyze_plan(graph, plan))
        return 0
    parser.error("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
