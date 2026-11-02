"""Parallelism-profile tools and CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..core.experiment_spec import BenchmarkSpec, load_benchmark_spec
from ..core.profile import (
    ParallelismProfiler,
    ProfileSpec,
    ProfilePlan,
    build_profile_plan,
    fit_amdahl,
    karp_flatt,
    recommend_total_curve,
)
from .experiment_plots import parallelism_html, plot_parallelism

__all__ = [
    "ProfileSpec",
    "ProfilePlan",
    "build_profile_plan",
    "parallelism_profile",
    "plot_parallelism",
    "parallelism_html",
    "fit_amdahl",
    "karp_flatt",
    "recommend_total_curve",
]


def _spec(value):
    if isinstance(value, BenchmarkSpec):
        return value
    if isinstance(value, (str, Path)):
        return load_benchmark_spec(value)
    return BenchmarkSpec(value)


def parallelism_profile(benchmark, config):
    """Run a full-grid parallelism profile."""
    return ParallelismProfiler(_spec(benchmark), config).run()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Profile CPU budgets on a benchmark plan.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        spec = load_benchmark_spec(args.benchmark)
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if args.plan_only:
            plan = build_profile_plan(spec, config)
            print(plan.summary)
            if args.json:
                print(json.dumps(plan.result, ensure_ascii=False, indent=2))
            return 0
        profile = ParallelismProfiler(spec, config).run()
        print(profile.summary)
        print(json.dumps(profile.analysis["recommendation"], ensure_ascii=False, sort_keys=True))
        return 1 if profile.analysis.get("errors") else 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
