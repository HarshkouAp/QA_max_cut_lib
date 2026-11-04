# Installation: choose a profile and verify it

This page takes you from a clean Python environment to a checked QA_max_cut_lib
`0.1.0rc2` installation. You will be able to install a CPU-only or an NVIDIA
profile, confirm the version and the public example, and know which optional
pieces each profile does and does not provide.

The package requires Python 3.10 or newer. The commands below were checked with
Python 3.12.14 on Windows 10; the CPU core is pure Python plus NumPy and SciPy.

## Install the CPU profile

The import-time core needs only:

- `numpy>=1.26,<3`
- `scipy>=1.11`

A bare `pip install -e .` is enough to import the package, solve with `ExactCPU`,
and run the statevector QAOA. Most users want the `cpu` extra, which adds graph
and table tooling, validation, plotting, and CPU profiling helpers:

```bash
python -m pip install -e ".[cpu]"
```

The `cpu` extra installs NetworkX, pandas, jsonschema, Rich, Matplotlib, Seaborn,
Plotly, joblib, threadpoolctl, psutil, python-louvain, and the classical stack
(CVXPY, SCS, dimod, dwave-samplers).

Run the command from the repository root. Importing the package does not load
Matplotlib, CuPy, CVXPY, or dimod; those libraries are imported only by the code
that needs them.

## Add the classical baselines

`GoemansWilliamsonCPU` and `SimulatedAnnealingCPU` need CVXPY, SCS, dimod, and
dwave-samplers. The `cpu` extra already lists them at their minimum versions;
the `classical` extra raises the CVXPY floor to 1.6 and installs only the
classical stack:

```bash
python -m pip install -e ".[cpu,classical]"
```

Both extras are CPU-only. A missing classical package is reported when you
select `gw_cpu` or `simulated_annealing_cpu`; it does not break the package
import, `ExactCPU`, or `QAOACPU`.

## Install one NVIDIA profile

GPU support is optional and installed as exactly one CuPy variant:

```bash
python -m pip install -e ".[cpu,gpu-cu13]"     # CUDA 13.x
python -m pip install -e ".[cpu,gpu-cu12]"     # CUDA 12.x
```

The profiles use `cupy-cuda13x[ctk]` and `cupy-cuda12x[ctk]`. The `ctk` extra
places the CUDA user-space components in the Python environment; the NVIDIA
driver remains a system dependency. Do not install both CuPy variants in one
environment.

Pick the profile that matches the installed driver and the device. CUDA 13
drops support for Maxwell, Pascal, and Volta GPUs, so a newer profile does not
make an older card compatible. The upstream rule is in the
[CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).

## Verify the installation

Check the version first:

```bash
python -c "import QA_max_cut_lib; print(QA_max_cut_lib.__version__)"
```

Expected output:

```text
0.1.0rc2
```

Then run the public example:

```bash
python examples/public_quickstart.py
```

Expected output:

```text
QA_max_cut_lib 0.1.0rc2
Exact optimum                    3.500000
Expected QAOA cut               3.044365
Best cut in 128 simulated shots 3.500000
Final-state optimal mass        0.805755
Readout spins                   {10: 1, 30: -1, 80: -1, 101: 1}
```

If the version or these numbers differ, stop here: the rest of the documentation
assumes this build. `QUICKSTART.md` explains what each number means and how to
recompute the cut by hand.

## Verify an NVIDIA device

Installing CuPy successfully is not proof that a kernel can run. Print the
device properties and the CUDA runtime version:

```bash
python -c "import cupy as cp; p = cp.cuda.runtime.getDeviceProperties(0); print(cp.__version__, p['name'].decode(), p['major'], p['minor'], cp.cuda.runtime.runtimeGetVersion())"
```

On the machine used for the GPU acceptance run this prints:

```text
14.2.0 NVIDIA GeForce RTX 3050 Laptop GPU 8 6 13020
```

A clear `cupy.cuda.runtime.CUDARuntimeError` here means the driver or the device
does not match the installed profile. The CPU solvers remain usable; the GPU
path fails explicitly and never falls back to NumPy. See `GPU.md` for the
supported representations and memory limits.

## Text rendering: MathText or LaTeX

Plotting selects the text engine explicitly through `style.text_engine`:

- `"mathtext"` (default) uses Computer Modern from Matplotlib and renders math
  labels through MathText. It needs no system TeX installation.
- `"latex"` sets `text.usetex=True`. It requires a working LaTeX installation,
  its formats and packages, and `dvipng` for raster Agg output. Installing only
  the `latex` binary is not sufficient.

The library does not switch engines automatically when LaTeX fails; the error
is reported. Requirements per export format are listed in the
[Matplotlib usetex guide](https://matplotlib.org/stable/users/explain/text/usetex.html).

## Threads and measurement hygiene

`num_treads` is a CPU worker budget, never a CUDA launch dimension. For this
release:

- sequential solvers accept only `1` and raise `ValueError` for larger values;
- parallel CPU controllers accept a larger budget and report it;
- the budget does not limit the native BLAS threads by itself.

For strict timing measurements, set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
and `MKL_NUM_THREADS` before Python starts, or wrap the measured call with
`threadpoolctl`. Experiment records keep the actual BLAS library and resource
counters; environment variables alone are not treated as evidence.

## Filesystem requirements for saving

Result export publishes new files by creating hard links on the destination
filesystem. The behavior is checked on Linux and on Windows 10/NTFS, including
simultaneous exports. Filesystems without hard-link support were not tested.
When a name already exists, the export raises `FileExistsError`; there is no
automatic overwrite or rename. Do not point `save.path` at a filesystem that
cannot create hard links.

## Reproduce the development environment with uv

The repository was developed with `uv`:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[cpu,notebooks,dev,docs]"
```

Add `[notebooks]` for Jupyter, `[docs]` for Sphinx, and `[classical]` for GW/SA
series. The source lists are `requirements/cpu.in`, `requirements/notebooks.in`,
`requirements/dev.in`, and `requirements/docs.in`. Resolved versions for the
local Windows build are in `requirements/lock-cpu-py312-win.txt`.

On Linux or macOS, replace `.venv/Scripts/python.exe` with
`.venv/bin/python`. A published wheel is installed with its file name, for
example:

```bash
python -m pip install qa_max_cut_lib-0.1.0rc2-py3-none-any.whl
```

## What the Laboratory sources need

The catalog and `provided_spins` / `provided_probabilities` warm-start sources
need only the CPU profile. A `classical_solver` source that selects GW or SA
needs the `classical` extra. No CuPy, Qiskit, or archive RAR is required to read
the catalog or to run the provided warm-start sources.

## Limitations

- Only Python 3.10+ is supported; the validated interpreter is 3.12.14.
- The GPU profiles are NVIDIA-only and were checked on one RTX 3050 Laptop GPU
  with 4 GiB of memory.
- Docker, Conda-specific recipes, and offline mirrors are not documented here;
  the commands use pip and uv only.
