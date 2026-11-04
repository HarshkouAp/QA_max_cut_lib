# Verification 0.1.0a16

Date: 2 November 2026.

## Base

- `QA_max_cut_lib 0.1.0a16`, Python 3.12.14, NumPy 2.5.4.
- CPU: `198 passed, 1 skipped, 1 warning`.
- Sphinx `-W`: success.
- Wheel installed into `.venv` and `.venv-wheelcheck`; CLI/examples outside
  `src`.

## A/B/C/D

- Product plus: `I_Q=0`, `I_Z=0`, `ZZ=0`.
- Bell Φ/Ψ: `I_Q=2 bits`, `I_Z=1 bit`; the ZZ sign differs, the MI does not.
- CZ plus: `I_Q=2`, `I_Z=0`.
- GHZ: `S_i=1`, pair `I_Q=I_Z=1`.
- Bell(A,E)⊗|0>_B: the QMI of pair A,B = 0; the correct S_B=0 (the old
  s2=s1 bug would have given a false value).
- Complex RDM: labels `80,-7,101,10`, `ρ_{-7}=[[0.5,-0.5i],[0.5i,0.5]]`.

## E/F/G

- Counts `[[3,1],[1,3]]`, M=8: `I_Z=0.18872187554086717` bits.
- Mask/statistics: values `(2,0,0)` give mean `2/3`, min `0`, max `2`,
  population std `2√2/3`; real zeros are not filtered out.
- Prefix p=3, θ=(0.1,0.2,0.3,0.4,0.5,0.6): layer 2 uses
  `(0.1,0.2,0.4,0.5)`; the theta halves are not mixed.

## Observer passivity

QAOA target: baseline `qaoa_cpu` and `laboratory.qaoa_mi_cpu` on the same
graph/seed give identical solution, cut, selected angles and shots.
The observer adds MI frames, but not a second readout and not new samples.
For p=2/two iterations/M_train=16/readout=64 the actual spend is 352 shots,
3 prefix snapshots, 15 all-pairs per snapshot.

Control p=1/three iterations/M_train=32/readout=64: baseline and observer give
the same solution/angles and **544 shots**.

RQAOA observer: full-state CPU source, frames before reducer commit, 2
reduction steps, `max_abs_correlation` unchanged. Correlation/reuse
provenance is stored separately.

## Not run

- a real cross-process restore of analysis I;
- fault-injection/unknown-charge observer phase;
- the full 12-trial series and the CPU profile;
- GPU/tomography/MI-guided selector/reconstruction;
- all production pair-scope/analysis-limit combinations.

## Artifacts

- wheel: `qa_max_cut_lib-0.1.0a16-py3-none-any.whl`, SHA-256
  `c5884c7f02552b99a044b64faa126cb877a63fe91e5275c1fe359dbfa5b28d2d`;
- sdist: `qa_max_cut_lib-0.1.0a16.tar.gz`, SHA-256
  `9de6baac3e88eeb6eb8d523309a05ee1a56cb6ef261bca453270adac4418a7b9`.
