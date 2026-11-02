"""Build the versioned Laboratory inventory from the read-only archive.

This developer utility only reads and parses Python text.  It never imports,
executes, or rewrites an archived module.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "reference" / "RQAOA_main"
ARCHIVE_SHA = "b4c684a5d4da469adc25bd99a405e402892d1a2524465e4c97ab6229c71a4aed"
OUT = ROOT / "src" / "QA_max_cut_lib" / "laboratory" / "data"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ast_hash(tree: ast.AST) -> str:
    payload = ast.dump(tree, include_attributes=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def top_level(tree: ast.Module):
    names = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            names.append({"kind": "class", "name": node.name, "line": node.lineno})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append({"kind": "function", "name": node.name, "line": node.lineno})
    return names


def imports(tree: ast.Module):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            out.append(node.module or "")
    return sorted(set(item for item in out if item))


def side_effect_hints(tree: ast.Module):
    hints = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            hints.append({"line": node.lineno, "hint": "top_level_call"})
        elif isinstance(node, ast.If):
            test = ast.unparse(node.test)
            if "__name__" in test and "__main__" in test:
                hints.append({"line": node.lineno, "hint": "main_guard_present"})
    return hints


def role_for(rel: str) -> str:
    if rel.startswith("Other/WS_QAOA"):
        return "warm_start_source"
    if rel in ("Diplom/Mk_5.py", "Laboratory/LS_RQAOA_fast_5.py"):
        return "legacy_ls_rqaoa_alias"
    if rel.startswith("Other/"):
        return "diagnostic_or_example"
    if rel.startswith("Laboratory/"):
        return "laboratory_family_source"
    if rel.startswith("Solvers/"):
        return "legacy_solver_source"
    if rel.startswith("Tools/"):
        return "tool_or_diagnostic"
    if rel.startswith("Generators/"):
        return "generator"
    if rel.startswith("ENV/"):
        return "environment_component"
    if rel.startswith("RQAOA_exp/"):
        return "research_experiment"
    if rel.startswith("Diplom/"):
        return "historical_solver_source"
    return "unresolved"


def variant_for(rel: str) -> str:
    if rel.startswith("Other/WS_QAOA"):
        return "warm_start_product_qaoa_cpu_v1"
    if rel in ("Diplom/Mk_5.py", "Laboratory/LS_RQAOA_fast_5.py"):
        return "legacy_ls_rqaoa_fast_alias"
    if rel.startswith("Solvers/WS_RQAOA"):
        return "warm_start_rqaoa"
    if rel.startswith("Tools/MI_"):
        return "mutual_information_diagnostics"
    if rel.startswith("Laboratory/"):
        return "laboratory_ls_families"
    return "deferred_environment_and_generators"


def main():
    records = []
    dumps = {}
    for path in sorted(ARCHIVE.rglob("*.py")):
        rel = path.relative_to(ARCHIVE).as_posix()
        data = path.read_bytes()
        text = data.decode("utf-8", errors="replace")
        error = None
        tree = None
        try:
            tree = ast.parse(text, filename=rel)
        except SyntaxError as exc:
            error = {"type": type(exc).__name__, "message": exc.msg, "line": exc.lineno}
        dump_hash = ast_hash(tree) if tree is not None else None
        record = {
            "schema_version": 1,
            "path": rel,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "lines": text.count("\n") + (1 if text and not text.endswith("\n") else 0),
            "parser": "python-3.12.14",
            "parse_error": error,
            "ast_sha256": dump_hash,
            "top_level": top_level(tree) if tree is not None else [],
            "imports": imports(tree) if tree is not None else [],
            "side_effect_hints": side_effect_hints(tree) if tree is not None else [],
            "role": role_for(rel),
            "variant_id": variant_for(rel),
            "review_status": "partially_reviewed" if rel.startswith("Other/WS_QAOA") else "inventoried",
            "runtime_status": "not_run",
            "note": (
                "Reviewed for the first warm-start transfer."
                if rel.startswith("Other/WS_QAOA")
                else "Catalogued from source text; scientific audit deferred."
            ),
        }
        records.append(record)
        if dump_hash:
            dumps.setdefault(dump_hash, []).append(rel)

    aliases = [
        {"ast_sha256": key, "paths": paths, "evidence": "identical AST dump without attributes"}
        for key, paths in dumps.items()
        if len(paths) > 1
    ]
    groups = {}
    for record in records:
        groups.setdefault(record["path"].split("/", 1)[0], 0)
        groups[record["path"].split("/", 1)[0]] += 1
    inventory = {
        "schema_version": 1,
        "archive_sha256": ARCHIVE_SHA,
        "member_count": len(records),
        "parser": "python-3.12.14",
        "directory_counts": groups,
        "files": records,
        "ast_alias_groups": aliases,
        "coverage": {
            "python_members_expected": 89,
            "python_members_recorded": len(records),
            "python_members_parsed": sum(record["parse_error"] is None for record in records),
            "parse_failures": sum(record["parse_error"] is not None for record in records),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote", OUT / "inventory.json", "files", len(records), "aliases", aliases)
    if len(records) != 89:
        raise SystemExit("inventory member count mismatch")
    if sum(record["parse_error"] is None for record in records) != 89:
        raise SystemExit("inventory parse coverage mismatch")


if __name__ == "__main__":
    main()
