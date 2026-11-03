# Installation

## CPU

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[cpu]"
```

Windows:

```bat
.venv\Scripts\activate
python -m pip install -e ".[cpu]"
```

The CPU path does not require CuPy/CUDA, CVXPY, dimod, Qiskit or TeX.
Classical SDP/SA extras are separate. Notebooks and Sphinx are optional
extras. NVIDIA profile is described in `docs/GPU.md`; it is not validated
by a CPU-only install.

## Smoke test

```bash
python -c "import QA_max_cut_lib; print(QA_max_cut_lib.__version__)"
python examples/generators_and_plans.py
```
