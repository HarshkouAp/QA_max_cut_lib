#!/usr/bin/env python
"""Fail when a documentation file still contains Cyrillic prose."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    ROOT / "README.md",
    *ROOT.glob("*.md"),
    ROOT / "docs" / "index.rst",
    *ROOT.glob("docs/*.md"),
    *ROOT.glob("docs/*.rst"),
    *ROOT.glob("docs/reference/*.md"),
]


def has_cyrillic(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return any("\u0400" <= char <= "\u04ff" for char in text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    offenders = sorted({path for path in TARGETS if path.is_file() and has_cyrillic(path)})
    if offenders:
        for path in offenders:
            print(path.relative_to(ROOT))
        return 1
    print("documentation language check: English only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
