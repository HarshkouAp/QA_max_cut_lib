"""Create the compact delivery archive for QA_max_cut_lib 0.1.0a11."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("C:/Users/Alexey/Downloads/QA_max_cut_lib_0.1.0a11.zip")
EXCLUDE_PARTS = {
    ".venv", ".venv-stage02", ".venv-wheelcheck", "build", "_build",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
} 
EXCLUDE_PREFIXES = ("notebooks/_outputs/", "results/series/small_comparison/")
INCLUDE_EXTRA = [
    "dist/qa_max_cut_lib-0.1.0a11-py3-none-any.whl",
    "dist/qa_max_cut_lib-0.1.0a11.tar.gz",
    "dist/SHA256SUMS.txt",
    "results/checkpoints/process",
    "results/series/warm_start_comparison",
]


def tracked():
    text = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-files", "-z"]
    ).decode("utf-8")
    return [item for item in text.split("\0") if item]


def add_file(zf, path: Path, arc: str):
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return
    if any(arc.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
        return
    zf.write(path, arc)


with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for rel in tracked():
        path = ROOT / rel
        if path.is_file():
            add_file(zf, path, rel)
    for rel in INCLUDE_EXTRA:
        path = ROOT / rel
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    add_file(zf, child, child.relative_to(ROOT).as_posix())
        elif path.is_file():
            add_file(zf, path, rel)

size = OUT.stat().st_size
digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
print(OUT)
print("bytes", size)
print("sha256", digest)
