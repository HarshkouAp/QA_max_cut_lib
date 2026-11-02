"""Format progress and reports independently of stored result values."""

import json
import sys
from ..core.records import resolve_record


def _print(text, color, code="36"):
    enabled = color == "always" or (color == "auto" and sys.stdout.isatty())
    print((f"\033[{code}m" + text + "\033[0m") if enabled else text)


def log_message(config, message, *, detail=False):
    """Print one progress event at the configured log level."""
    output = config["output"]
    if output["logs"] >= (2 if detail else 1):
        _print(message, output["color"], "36")


def print_report(source, level=1, color="auto"):
    """Print a short or detailed report. Never calculate missing data."""
    if level not in (0, 1, 2):
        raise ValueError("Report level must be 0, 1, or 2.")
    if level == 0:
        return
    record = resolve_record(source)
    _print(f"QA_max_cut_lib | {record.algorithm_type} | {record.status}", color, "32")
    _print(
        f"Cut: {record.cut_value:.9g}   Energy: {record.solution_energy:.9g}   "
        f"Time: {record.timing['compute_seconds']:.4f} s",
        color,
        "37",
    )
    print("Resources:", json.dumps(record.resources, sort_keys=True))
    if "reductions_completed" in record.field_names:
        print(
            "RQAOA:",
            f"reductions={record.get('reductions_completed')}",
            f"source={record.get('solution_source')}",
            f"cut_offset={record.get('cut_offset')}",
            f"ising_constant={record.get('ising_constant')}",
        )
    if level == 2:
        print("Termination:", record.termination_reason)
        if "remaining_node_order" in record.field_names:
            print("Remaining nodes:", record.get("remaining_node_order"))
        print("Solution:", record.solution)
        print(
            "Configuration:",
            json.dumps(record.config, sort_keys=True, ensure_ascii=False),
        )
