"""Inspect and resume QA_max_cut_lib checkpoints without rerunning a solver."""

from __future__ import annotations

import argparse
import json
import sys

from ..core.checkpoint import inspect_checkpoint, restore

__all__ = ["inspect_checkpoint", "restore_checkpoint", "main"]


def restore_checkpoint(path, **kwargs):
    """Restore a solver object from a checkpoint without executing it."""
    return restore(path, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect or resume a QA_max_cut_lib checkpoint.")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="read manifest metadata without a solver")
    inspect.add_argument("--checkpoint", required=True)
    resume = sub.add_parser("resume", help="restore and continue a managed run")
    resume.add_argument("--checkpoint", required=True)
    resume.add_argument("--report", type=int, default=0, choices=(0, 1, 2))
    resume.add_argument("--logs", type=int, default=0, choices=(0, 1, 2))
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            print(json.dumps(inspect_checkpoint(args.checkpoint), ensure_ascii=False, indent=2))
            return 0
        solver = restore(args.checkpoint)
        solution = solver.resume()
        result = solver.result
        print(json.dumps(
            {
                "run_id": result.get("run_id"),
                "status": result.get("status"),
                "termination_reason": result.get("termination_reason"),
                "cut_value": result.get("cut_value"),
                "solution": solution,
                "resources": result.get("resources"),
                "checkpoint": result.get("checkpoint_info"),
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
