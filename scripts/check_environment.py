"""Check the local QA_max_cut_lib environment without importing optional stacks.

Run from the project root:
    .venv/Scripts/python.exe scripts/check_environment.py
"""

from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def line(name: str, value: object) -> None:
    print(f"{name:<28} {value}")


def package_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception as error:  # noqa: BLE001 - report, do not hide
        return f"NOT AVAILABLE ({type(error).__name__}: {error})"
    return getattr(module, "__version__", "installed")


def main() -> int:
    print("QA_max_cut_lib environment check")
    print("=" * 60)
    line("project", PROJECT)
    line("python", sys.version.split()[0])
    line("executable", sys.executable)
    line("platform", platform.platform())

    try:
        import psutil
        line("cpu (physical/logical)", f"{psutil.cpu_count(False)} / {psutil.cpu_count(True)}")
        line("ram_gb", round(psutil.virtual_memory().total / 2**30, 1))
    except Exception as error:  # noqa: BLE001
        line("cpu/ram", f"NOT AVAILABLE ({error})")

    print("-" * 60)
    for name in ("numpy", "scipy", "pandas", "matplotlib", "seaborn", "plotly",
                 "networkx", "cvxpy", "scs", "dimod", "dwave.samplers",
                 "joblib", "threadpoolctl", "pytest"):
        line(name, package_version(name))

    print("-" * 60)
    try:
        sys.path.insert(0, str(PROJECT / "src"))
        import QA_max_cut_lib
        from QA_max_cut_lib import ExactCPU, WeightedGraph

        line("QA_max_cut_lib", QA_max_cut_lib.__version__)
        solver = ExactCPU()
        solution = solver.solve(WeightedGraph([(0, 1, 1.0), (1, 2, -2.0)], nodes=[0, 1, 2, 5]))
        line("smoke exact cut", f"{solver.cut_value} (solution {solution})")
    except Exception as error:  # noqa: BLE001
        line("QA_max_cut_lib smoke", f"FAILED ({type(error).__name__}: {error})")
        return 1

    print("-" * 60)
    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        result = subprocess.run(
            [nvidia, "--query-gpu=name,driver_version,memory.total,memory.free",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        )
        line("nvidia-smi", result.stdout.strip() or result.stderr.strip())
    else:
        line("nvidia-smi", "not found")
    line("cupy", package_version("cupy"))
    line("latex", shutil.which("latex") or "not found")
    line("pdflatex", shutil.which("pdflatex") or "not found")

    print("=" * 60)
    print("Environment check finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
