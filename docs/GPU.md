# NVIDIA GPU QAOA: environment, representations, and memory

QA_max_cut_lib `0.1.0rc2` runs QAOA, RQAOA, and LS-RQAOA on an NVIDIA GPU
through CuPy. CPU solvers and result loading never require CUDA or CuPy. This
page shows how to choose exactly one GPU profile, which state representation to
use, what the sampler and gradients do, and how the memory preflight rejects a
task before allocating it.

By the end you can run a small QAOA on the device and read its
`state_representation` block.

## One environment, one CuPy package

A single environment must contain exactly one of:

- `gpu-cu13` — `cupy-cuda13x[ctk]`, CUDA 13.x;
- `gpu-cu12` — `cupy-cuda12x[ctk]`, CUDA 12.x.

A compatible NVIDIA driver is a system dependency. Installing `cu13` does not
make an older card compatible: CUDA 13 drops Maxwell, Pascal, and Volta.

The verified local device for this release:

| Field | Value |
|---|---|
| Device | NVIDIA GeForce RTX 3050 Laptop GPU |
| Compute capability | 8.6 |
| VRAM | 4 GiB (about 3.2–3.4 GiB free during tests) |
| Driver | 591.74 |
| CUDA runtime of the process | 13020 |
| CuPy | 14.2.0 |
| Python | 3.12.14 |

The RTX 5060 Laptop named in the task description is a target, not a measured
device of this machine. V100, RTX 5060, and other cards are not claimed as
verified without a separate run.

Check a device before trusting a profile:

```bash
python -c "import cupy as cp; p = cp.cuda.runtime.getDeviceProperties(0); print(cp.__version__, p['name'].decode(), p['major'], p['minor'], cp.cuda.runtime.runtimeGetVersion())"
```

Importing CuPy is not a kernel test. A clear `CUDARuntimeError` or a failed
elementwise kernel means the driver, the device, and the profile disagree.

## State representations

`representation="full"` stores the physical state `psi` of length `2**N`.
`representation="spin_flip_pairs"` stores `chi` of length `2**(N-1)`:

\[
|[r]\rangle=\frac{|r\rangle+|\bar r\rangle}{\sqrt2},\qquad
\chi_r=\sqrt2\,\psi_r,\qquad q_r=|\chi_r|^2 .
\]

The representative is chosen so that the last bit of `node_order[N-1]` is zero.
Reconstruction is:

```python
psi = numpy.concatenate([chi, chi[::-1]]) / numpy.sqrt(2)
```

The pair mass `q_r` belongs to two physical rows, each with probability
`q_r / 2`. `state_representation` reports the storage type, normalization,
`node_order`, the representative rule, and the meaning of the masses, so a
consumer does not have to infer them from the array length.

Use `spin_flip_pairs` for larger graphs: it halves the state memory. Use `full`
when a consumer needs the physical amplitudes directly.

## Entry points

| Class | `type` | Role |
|---|---|---|
| `QAOAGPU` | `qaoa_gpu` | full solver: spins, readout, `state_representation` |
| `QAOAGPUCorrelationSource` | `qaoa_statevector_gpu` | training and physical ZZ correlations for RQAOA/LS |
| `RQAOAGPU` | `rqaoa_gpu` | shared RQAOA coordinator with a GPU source |
| `LSRQAOAGPU` | `ls_rqaoa_gpu` | LS with `execution.backend="gpu_owner"`; local tasks run one at a time on one device |

All solvers produce the common `RunRecord`. GPU arrays are converted to
portable NumPy arrays when the record is collected; live device buffers are not
part of the result.

## Run a small GPU QAOA

```python
from QA_max_cut_lib import QAOAGPU, WeightedGraph

graph = WeightedGraph(edges=[(0, 1, 1.0), (1, 2, -0.5), (0, 2, 2.0)], nodes=[0, 1, 2])
solver = QAOAGPU({
    "seed": 3,
    "parameters": {
        "layers": 1,
        "representation": "spin_flip_pairs",
        "dtype": "complex64",
        "max_vertices": 12,
    },
    "objective": {"type": "mean_cut"},
    "estimation": {"mode": "exact"},
    "optimizer": {"parameters": {"max_steps": 2}},
    "readout": {"mode": "best_of_shots", "shots": 32},
    "limits": {"shots_total": 32},
})
solution = solver.solve(graph)
print(solution)                       # {0: 1, 1: 1, 2: -1}
print(solver.cut_value)               # 1.5
print(solver.result["state_representation"]["type"])  # spin_flip_pairs
```

On the acceptance machine this prints cut `1.5`. The cut is data-dependent; the
stable part of the example is the schema, not the number on another device.

Inspect these fields in `state_representation`:

| Field | Meaning |
|---|---|
| `type` | `full` or `spin_flip_pairs` |
| `n_vertices`, `node_order` | graph size and bit order |
| `dtype` | `complex64` or `complex128` |
| `amplitude_kind` | `statevector` or `normalized_pair_chi` |
| `probability_kind` | physical probabilities or representative pair masses |
| `representative_rule` | `last_node_order_bit_zero` for pairs |
| `normalization` | how masses map to physical probabilities |
| `device` | device name, capability, free/total memory, runtime, CuPy version |

## Objectives, gradients, and shots

All five criteria are evaluated on the device from `C(r)` and `q_r`:
`mean_cut`, `cvar`, `gibbs`, `best_of_k`, and `threshold_probability`.

- `gradient.type="adjoint"` is available for Exact Mean Cut only; it requires
  `estimation.mode="exact"`.
- Central differences are available for every criterion.
- Adjoint is not applied to CVaR, Gibbs, Best-of-K, Threshold Probability, or
  shot-based estimation.
- One full central-difference iteration uses `4p+1` objective evaluations; an
  adjoint pass is not reported as that many experimental evaluations.

The sampler performs a CDF and `searchsorted` on the device. In pair mode it
first samples `r` by `q`, then samples the orientation independently, and
returns both physical rows. The shot budget is checked by the coordinator before
measurement; the readout reservation is taken before training starts.

## Memory preflight

`memory_preflight` estimates the state, diagonal, masses/probabilities, and one
temporary buffer, then compares the estimate with:

- free device memory;
- `gpu.memory.fraction_of_available` (default `0.8`);
- `gpu.memory.reserve_bytes` (default `268435456`, 256 MiB).

The estimate counts only this run's large arrays. It does not treat another
process's CuPy pool as exclusive and cannot guarantee the absence of an OOM when
other consumers share the device.

`parameters.max_vertices` is a user ceiling, not a placement promise:

- `QAOAGPU` default `30`;
- `QAOAGPUCorrelationSource` default `20`;
- hard cap `30` for both.

An 8 GiB card is not promised to fit an arbitrary `N=30` state for every dtype
and criterion. Halving the state with `spin_flip_pairs` or using `complex64` is
the first adjustment.

## Failure behavior

- If CuPy or a device is absent, the GPU path raises a clear error.
- If the preflight rejects a size, it raises `MemoryError` with the estimated and
  workspace byte counts.
- The library never switches a GPU request to NumPy silently.
- CPU solvers remain available in the same environment.

## Limitations

Local fields, warm-start, an alternative mixer, multi-GPU execution, distributed
runs, CUDA Graphs, Walsh tiles, and `batch>1` are not implemented. The direct
`pairwise` mixer is the verified variant; faster mixer variants need a separate
check. GPU LS does not create CUDA contexts in multiple processes: local tasks
are served sequentially by one owner device.
