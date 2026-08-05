# Contributing to cyclops

Thanks for your interest in contributing to cyclops!

To submit PRs, please fill out the PR template along with the PR. If the PR
fixes an issue, don't forget to link the PR to the issue!

## Setting up your environment

cyclops uses [uv](https://docs.astral.sh/uv/getting-started/installation/) to
manage dependencies. Once uv is installed, set up a development environment
with the test dependency group and activate it:

```bash
uv sync --group test
source .venv/bin/activate
```

Some modules have optional dependencies (e.g. `torch`, `xgboost`, `monai`,
`alibi-detect`) that aren't installed by default - see the table in
[README.md](README.md) for the full list of extras. To work on a module that
needs one of these, install the matching extra, e.g.:

```bash
uv sync --group test --extra alibi-detect
```

Install the pre-commit hooks so code-style issues are caught before you push:

```bash
pre-commit install
```

## Running tests

Run the unit test suite with:

```bash
python -m pytest -m "not integration_test"
```

Tests marked `@pytest.mark.integration_test` require external
infrastructure (e.g. a live database via
[cycquery](https://github.com/VectorInstitute/cycquery)) that isn't
available in a plain checkout, which is why they're excluded above and not
run in CI. Only run them locally if you have that infrastructure set up.

Pass `-k <pattern>` to scope a run to a subset of tests, and
`--cov=cyclops` to see coverage for the code you changed.

## Coding guidelines

For code style, we recommend the [PEP 8 style guide](https://peps.python.org/pep-0008/).

For docstrings we use [numpy format](https://numpydoc.readthedocs.io/en/latest/format.html).

We use [ruff](https://docs.astral.sh/ruff/) for code formatting and static code
analysis. Ruff checks various rules including [flake8](https://docs.astral.sh/ruff/faq/#how-does-ruff-compare-to-flake8). The pre-commit hooks show errors which you need to fix before submitting a PR.

Last but not the least, we use type hints in our code which is then checked using
[mypy](https://mypy.readthedocs.io/en/stable/).

You can run all pre-commit checks (ruff, ruff-format, mypy, notebook
stripping) against the whole repository at once with:

```bash
pre-commit run --all-files
```

## Repository layout

- `cyclops/data` - dataset construction, loading, and slicing
- `cyclops/models` - scikit-learn and PyTorch model wrappers and implementations
- `cyclops/tasks` - task formulations (e.g. binary/multi-label classification) tying data and models together
- `cyclops/evaluate` - metrics and fairness evaluation for clinical prediction tasks
- `cyclops/monitor` - dataset shift / drift detection for deployed models
- `cyclops/report` - model report card generation
- `cyclops/utils` - small shared utilities used across the other modules

Each module has a corresponding test package under `tests/cyclops/`.

## Code of Conduct

Participation in this project is governed by our
[Code of Conduct](CODE_OF_CONDUCT.md).
