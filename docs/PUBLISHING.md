# Publishing: prepare a reviewable 0.1.0rc2 release

This page is a checklist for publishing QA_max_cut_lib `0.1.0rc2` from a commit
whose documentation, examples, and tests describe that same commit. It covers
the files that ship together, the claims that must be rechecked, and the order
in which a public version is opened.

By the end you can rebuild the documentation with warnings as errors, verify the
scientific control points, and decide whether the artifact is ready for a tag.

## What ships together

| File | Role |
|---|---|
| `README.md` | repository landing page, installation, and the example result |
| `examples/public_quickstart.py` | reproducible calculation and source of the illustration |
| `docs/weighted_max_cut.rst` | scientific chapter with the derivation and reference checks |
| `docs/_static/signed-max-cut.svg` | signed graph and cut-value distribution |
| `docs/DOCUMENTATION_STYLE.md` | authoring rules used for the public chapters |

Keep author information and links to verified features from the current README.
The Sphinx root `docs/index.rst` already places `weighted_max_cut`, the
quickstart, concepts, algorithm choice, API, installation, configuration,
saving, checkpoints, experiments, GPU, and Laboratory pages in the reader path.
Do not add a page to the toctree unless it exists and builds.

After changing the example or the figure, rebuild both:

```bash
python -m pip install ".[cpu,docs]"
python examples/public_quickstart.py --figure docs/_static/signed-max-cut.svg
python -m sphinx -W -b html docs docs/_build/html
```

A Sphinx warning is an error for this project. Build from the same source tree
that produced the wheel; do not mix a released wheel with a newer `docs/`
directory.

## Recheck the scientific claims

For every method page, state the task and the input limits, the sign convention,
the physical model, the training criterion, the readout rule, and the cost. Give
one small example with an analytical answer or an enumeration, a link to the
primary source, and a configuration that reproduces it. Compare methods on the
same graphs, with an explicit budget, hardware, and spread over independent
runs. Report the selected cut weight, the expected weight, and the probability of
reaching a stated quality separately.

Three control points cover this chapter set:

1. **One edge.** Check the energy sign, amplitudes, expected cut, and
   correlator against the analytical formula. `tests/test_science.py` contains
   `test_one_edge_and_physical_correlations` and
   `test_qaoa_against_dense_matrix_exponential` for this purpose.
2. **Signed graph.** Compare the simulator distribution with enumeration over
   basis states and check the returned partition. The exact tests use
   independent assignments.
3. **Reduction.** Enumerate the allowed spins for both constraint signs and
   check the constant offset on every partition. For the recursive solver,
   repeat the check through its public API.

A claim that is not backed by one of these checks, a committed test, or a
recorded run is a limitation, not a result.

## Open the verified version

1. **Audit the source and the history.** Check sources, Git history,
   dependencies, and rights for every artifact that will become public. Remove
   secrets, local environments, and accidental experiment output. Agree on the
   project license and add `LICENSE`. Fill `CITATION.cff` with confirmed author
   names, the version, and the repository URL. This release candidate has no
   public distribution license yet and is not on PyPI; do not imply otherwise.
2. **Build and install in isolation.** In CI, build the wheel and sdist, install
   the wheel in a clean environment, and run the tests and the public example
   outside the source directory. Build Sphinx with `-W` from the same version.
   Check the claimed platforms and install profiles, the documentation link, the
   formulas, and the mobile view of the landing page.
3. **Publish the reviewed history.** Push the agreed history to the chosen
   GitHub repository. After CI passes, publish Sphinx through GitHub Pages, link
   the README to the site, and create a tag and a GitHub Release for the verified
   commit. For a first preview, use the version recorded in the wheel metadata.
   If PyPI is required, verify it with a separate installation scenario.

GitHub references: [importing an existing project](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github),
[licensing a repository](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository),
[citation files](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files),
and [Pages with custom workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

## Verification record for this set

The checks below were run on Windows 10 with Python 3.12.14, the installed
`QA_max_cut_lib 0.1.0rc2` wheel, NumPy 2.5.4, SciPy 1.18.1, and Sphinx 9.1.1.
The public example was run outside the package source tree.

- `218 passed` CPU tests; `10 passed` GPU tests on the local RTX 3050.
- Sphinx `-W` builds successfully.
- `examples/public_quickstart.py`: exact optimum `3.5`, expected QAOA cut
  `3.044365`, best cut in 128 shots `3.5`, optimal mass `0.805755`.
- `test_one_edge_and_physical_correlations` checks the one-edge state,
  expected cut, and correlator against the analytical values;
  `test_qaoa_against_dense_matrix_exponential` compares the state with a dense
  matrix exponential for one and two layers, including a negative edge weight.
- `tests/test_reduction.py` and `tests/test_rqaoa.py` assert the reduction
  offsets (`cut_offset`, `ising_constant`) on explicit partitions.
- Shot-budget behavior is covered by
  `test_shot_budget_seed_reuse_and_result_independence` and
  `test_budget_can_only_cover_readout_and_zero_training`.

Repeat the list on the commit that will be tagged. A number copied from this page
without a run on that commit is not acceptance evidence.

## Release gate

`0.1.0rc2` is a release candidate, not `0.1.0`. The remaining blocker is R19:
not every materially different branch of the 89-file archive has been ported.
`RELEASE_READINESS.md` and `SOURCE_CLOSURE.json` record the exact list. Do not
label a build `0.1.0`, and do not describe the Laboratory as a complete legacy
migration, until that gate is closed or explicitly excluded.

## Limitations

- The documentation build requires the `docs` extra; it is not part of the CPU
  runtime profile.
- The wheel-name and version checks use the built artifact, not the working tree.
- A GitHub Pages deployment and a PyPI upload are separate release actions; one
  does not verify the other.
