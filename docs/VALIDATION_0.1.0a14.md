# Verification 0.1.0a14

Date: 2 November 2026.

## Base

- `QA_max_cut_lib 0.1.0a14`, Python 3.12.14, NumPy 2.5.4.
- CPU: `184 passed, 1 skipped, 1 warning`.
- Sphinx `-W`: success.
- Wheel installed into `.venv` and `.venv-wheelcheck`; examples/CLI run
  outside the `src` tree.
- GPU: not re-run; field/mean-field GPU modes are not claimed.

## A — model cut and rounding

Two singleton blocks, edge of weight 2, means `(0.2,-0.4)`:

```text
model mean cut = 1.08
rounded solution = (+1,-1)
full cut = 2
```

The sum of the two local affected means is `2.16`; that is not the global cut.

## B — damping

Old `(m,q)=((1,1),1)`, candidate `z=(-1,1)`, `r=-1`, `lambda=1/2` give
`m'=(0,1)`, `q'=0` and pair probabilities `(1/2,0,1/2,0)`. A raw `q=-1`
with an already smoothed `m` gives a negative pair probability and is
rejected.

## D — applied angles

One spin, `h=1`, `theta_old=(0,0)`, candidate `(-pi/4,pi/4)`,
`kappa=1/4`: applied `(-pi/16,pi/16)` gives
`z=-(2-sqrt(2))/4 ≈ -0.1464466094`.

## Budget / status

- `B=47`, 3 mandatory moment jobs of 16 each: the sweep does not start,
  `moment_reserve_unavailable`, 0 shots.
- `B=48`: all three jobs perform moments only, `shots_used=48`,
  `training_shots=0`.
- target Jacobi 3 blocks × 3 sweeps × (`2×5×8+16`)=864 shots; the actual run
  used 864 and 9 logical jobs.
- `max_sweeps=0`: 0 local jobs, sign-rounded result and full cut.
- empty graph: `{}` without quantum work.

## Checkpoint

A compact checkpoint is implemented at the boundary of a full sweep:
`initialized` and `gauss_seidel_sweep_commit`/`jacobi_sweep_commit`.
`restore(path)` + `resume()` preserve m/q/theta, budget, RNG, histories and do
not repeat already completed sweeps. The check was performed in the same
process; a separate cross-process test and fault injection are **not claimed**
in this version. A mid-sweep child checkpoint and a replay receipt are also
not claimed.

## Not run

- overlapping blocks, adaptive damping, nonlinear/correlation-weighted
  fields, GPU/chi/analytic path;
- a full Jacobi multiprocessing pool with memory-limited resident jobs;
- GS/Jacobi completion permutations at production sizes;
- the 16-trial comparative series and the P=1/2/4 profile;
- fault injection I before/after BEGIN/receipt/commit.
