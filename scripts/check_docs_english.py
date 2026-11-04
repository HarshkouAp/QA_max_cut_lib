#!/usr/bin/env python
"""Fail when a documentation artifact still contains Cyrillic prose."""

from __future__ import annotations

import argparse
import json
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
    *ROOT.glob("docs/notebook_previews/*.html"),
]


def text_of(path: Path) -> str:
    if path.suffix == ".ipynb":
        notebook = json.loads(path.read_text(encoding="utf-8"))
        return "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "markdown"
        )
    return path.read_text(encoding="utf-8", errors="ignore")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    targets = [path for path in TARGETS if path.is_file()]
    targets.extend(sorted((ROOT / "notebooks").glob("*.ipynb")))
    offenders = sorted(
        {
            path
            for path in targets
            if any("\u0400" <= char <= "\u04ff" for char in text_of(path))
        }
    )
    if offenders:
        for path in offenders:
            print(path.relative_to(ROOT))
        return 1
    print(f"documentation language check: English only ({len(targets)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
