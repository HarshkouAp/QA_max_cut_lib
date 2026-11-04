# QA_max_cut_lib

**A Python research library for weighted Max-Cut with LS-RQAOA, QAOA, and
classical baselines.**

The LS-RQAOA family presented here was developed by **Alexey Gorshkov**.
The base method builds on Recursive QAOA, introduced by Bravyi et al.
It estimates the spin correlations used to choose each reduction pair on a
working subgraph.
[Original RQAOA paper](https://arxiv.org/abs/1910.08980)

[Release 0.1.0rc2](https://github.com/HarshkouAp/QA_max_cut_lib/releases/tag/v0.1.0rc2)

## Run the CPU example

With Python 3.10 or newer, run these commands from the repository root.

```bash
python -m pip install ".[cpu]"
python examples/public_quickstart.py
```

![A signed weighted graph and the depth-one QAOA distribution of cut values](docs/_static/signed-max-cut.svg)

In this QAOA example, the exact optimum is **3.500**, the expected cut in the
final state is **3.044**, and the best cut in 128 simulated measurements is
**3.500**.

## Documentation

[Quickstart](docs/QUICKSTART.md) ·
[Choose a solver](docs/CHOOSE_ALGORITHM.md) ·
[Mathematics](docs/MATHEMATICS.md) ·
[API](docs/API.md)
