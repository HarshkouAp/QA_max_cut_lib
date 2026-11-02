"""Command-line access to the versioned Laboratory catalog."""

from __future__ import annotations

import argparse
import json

from .catalog import catalog_summary, get_variant, list_variants


def _print_json(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m QA_max_cut_lib.laboratory",
        description="List and inspect the reviewed historical variant catalog.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_list = sub.add_parser("list", help="list catalog variants")
    p_list.add_argument("--implemented-only", action="store_true")
    p_list.add_argument("--json", action="store_true")
    p_show = sub.add_parser("show", help="show one variant card")
    p_show.add_argument("--id", required=True, dest="variant_id")
    p_show.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "list":
        cards = list_variants(include_deferred=not args.implemented_only)
        if args.json:
            _print_json(cards)
        else:
            for card in cards:
                print(
                    f"{card['variant_id']:42s} runnable={card['runnable']!s:5s} "
                    f"implementation={card['status']['implementation']} "
                    f"relation={card.get('historical_relation', '-')}"
                )
        return 0
    card = get_variant(args.variant_id)
    if args.json:
        _print_json(card)
    else:
        print(json.dumps(card, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
