"""Create the compact delivery archive for QA_max_cut_lib 0.1.0a18."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("C:/Users/Alexey/Downloads/QA_max_cut_lib_0.1.0a18.zip")
EXCLUDE_PARTS = {
    ".venv", ".venv-stage02", ".venv-wheelcheck", "build", "_build",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
}
EXCLUDE_PREFIXES = ("notebooks/_outputs/",)
INCLUDE_EXTRA = [
    "dist/qa_max_cut_lib-0.1.0a18-py3-none-any.whl",
    "dist/qa_max_cut_lib-0.1.0a18.tar.gz",
    "dist/SHA256SUMS.txt",
    "results/generators_series/stage18_48_cases.json",
]


def tracked():
    text = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z"]).decode("utf-8")
    return [item for item in text.split("\0") if item]


def add(zf, path, arc):
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return
    if any(arc.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
        return
    zf.write(path, arc)


with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for rel in tracked():
        path = ROOT / rel
        if path.is_file():
            add(zf, path, rel)
    for rel in INCLUDE_EXTRA:
        path = ROOT / rel
        if path.is_file():
            add(zf, path, rel)

print(OUT)
print("bytes", OUT.stat().st_size)
print("sha256", hashlib.sha256(OUT.read_bytes()).hexdigest())
