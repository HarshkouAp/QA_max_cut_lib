"""Public experiment tools and CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..core.experiment_spec import BenchmarkSpec, load_benchmark_spec, plan_experiments as _plan
from ..core.runner import ExperimentRunner
from ..core.series_analysis import analyze_series
from ..core.series_store import load_saved_series
from .experiment_plots import plot_series_coverage, plot_series_quality

__all__ = [
    "BenchmarkSpec",
    "load_benchmark_spec",
    "plan_experiments",
    "ExperimentRunner",
    "run_experiments",
    "analyze_series",
    "load_saved_series",
    "plot_series_coverage",
    "plot_series_quality",
]


def _spec(value):
    if isinstance(value, BenchmarkSpec):
        return value
    if isinstance(value, (str, Path)):
        return load_benchmark_spec(value)
    return BenchmarkSpec(value)


def plan_experiments(spec):
    """Return a dry-run plan.  No solver or export is called."""
    return _plan(_spec(spec))


def run_experiments(spec, *, reuse_completed=False):
    """Run a benchmark spec and return its independent series."""
    return ExperimentRunner(_spec(spec), reuse_completed=reuse_completed).run()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plan and run reproducible Max-Cut experiments.")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="expand a benchmark spec without running solvers")
    plan.add_argument("--config", required=True)
    plan.add_argument("--json", action="store_true", help="print the full plan JSON")
    run = sub.add_parser("run", help="execute a benchmark spec")
    run.add_argument("--config", required=True)
    run.add_argument("--reuse-completed", action="store_true")
    args = parser.parse_args(argv)
    try:
        spec = load_benchmark_spec(args.config)
        if args.command == "plan":
            result = plan_experiments(spec)
            if args.json:
                print(json.dumps(result.result, ensure_ascii=False, indent=2))
            else:
                print(result.summary)
                print(json.dumps(result.counts, ensure_ascii=False, sort_keys=True))
                print("rejected:", len(result.rejected))
            return 0
        series = ExperimentRunner(spec, reuse_completed=args.reuse_completed).run()
        print(series.summary)
        print(json.dumps(series.analysis["by_solver"], ensure_ascii=False, sort_keys=True))
        return 1 if series.analysis["failed"] else 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
