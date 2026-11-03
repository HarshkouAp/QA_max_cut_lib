"""Build the 0.1.0rc1 delivery archive and manifest."""
from __future__ import annotations
import hashlib
from pathlib import Path
import json
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("C:/Users/Alexey/Downloads/QA_max_cut_lib_0.1.0rc2.zip")
EXCLUDE = {".venv", ".venv-stage02", ".venv-wheelcheck", "build", "_build", "__pycache__", ".pytest_cache"}
EXTRA = [
    "dist/qa_max_cut_lib-0.1.0rc2-py3-none-any.whl",
    "dist/qa_max_cut_lib-0.1.0rc2.tar.gz",
    "dist/SHA256SUMS.txt",
    "results/generators_series/stage18_48_cases.json",
    "RELEASE_READINESS.md",
    "REQUIREMENTS_TRACEABILITY.json",
    "SOURCE_CLOSURE.json",
]


def tracked():
    text = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z"]).decode()
    return [item for item in text.split("\0") if item]


def selected():
    seen = set()
    items = []
    for rel in tracked() + EXTRA:
        path = ROOT / rel
        if not path.is_file():
            continue
        arc = rel.replace("\\", "/")
        if arc in seen:
            continue
        if any(part in EXCLUDE for part in path.parts):
            continue
        seen.add(arc)
        items.append((path, arc))
    return items


items = selected()
manifest = {
    "schema_version": 1,
    "package": "QA_max_cut_lib",
    "version": "0.1.0rc2",
    "release_status": "release_candidate",
    "release_profile": "CPU verified; GPU and full controller closure blocked",
    "files": {},
}
for path, arc in items:
    data = path.read_bytes()
    manifest["files"][arc] = {
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for path, arc in items:
        zf.write(path, arc)
    zf.writestr("MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

print(OUT)
print("bytes", OUT.stat().st_size)
print("sha256", hashlib.sha256(OUT.read_bytes()).hexdigest())
print("files", len(items) + 1)
