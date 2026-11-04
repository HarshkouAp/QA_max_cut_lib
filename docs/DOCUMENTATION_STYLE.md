# Documentation authoring rules

This page is for authors and reviewers. It records the rules used for the public
chapters. A reader-facing page starts with a task, follows one example, and ends
with a check that can be repeated.

## Reader journey

1. **First result** — `QUICKSTART.md`: installation, a short script, expected
   output, and an explanation of the result.
2. **Concepts** — `CONCEPTS.md`: graph, spins, cut, energy, bit order, and the
   readout rule.
3. **Scientific model** — `weighted_max_cut.rst` and `MATHEMATICS.md`: formulas,
   assumptions, and exact checks.
4. **Method choice** — `CHOOSE_ALGORITHM.md`: limitations, cost, and suitable
   operating modes.
5. **Practice** — installation, configuration, saving, checkpoints, GPU,
   experiments, and Laboratory.
6. **Reference** — `API.md` and the generated `api.rst`: exact types, defaults,
   constraints, and exceptions.
7. **Evidence** — `VALIDATION*.md`, `GIT_STAGE*.md`, and performance files. These
   are acceptance records, not the introductory path.

## Page rules

- A heading names an action or a reader question, not a stage number.
- The opening paragraph states what is given, what is sought, and what the
  reader will be able to do or check.
- Introduce a symbol before its first meaningful use; keep one meaning per
  symbol throughout the page.
- For every formula, state what is substituted, what it returns, and how to read
  its sign.
- Keep a mathematical statement, a model assumption, and an implementation
  choice visibly distinct.
- Use one running example where practical; give its input, command, seed, and
  checkable result.
- Put a limitation next to the conclusion or command it affects.
- Keep parameter tables, defaults, and exceptions in the reference; a method
  chapter links to the reference instead of duplicating it.
- Do not present a historical acceptance log as a current user guide.

## Version and evidence

Public pages describe `0.1.0rc2`. Numbers come from an actual run or test; an
analytical answer is labelled as expected. An unverified fact is not published
as a result. The remaining R19 blocker is described in `RELEASE_READINESS.md`
and is not hidden in introductory chapters.

## Build check

```bash
.venv-stage02/Scripts/python.exe -m pytest -q
.venv-stage02/Scripts/sphinx-build.exe -W -b html docs docs/_build/html
```

A Sphinx warning is an error for this project.
