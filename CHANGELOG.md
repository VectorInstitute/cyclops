# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `cyclops.monitor`: `Detector.detect_shift_by_subgroup()` runs the fitted
  drift tester independently on each subgroup of a target dataset (e.g.
  age band, sex, hospital site, defined via the existing `SliceSpec`),
  instead of only testing the aggregate population - a model can look
  stable overall while drifting badly for a specific subgroup, which
  matters for health-equity-aware monitoring. Includes Bonferroni
  correction across subgroups and a minimum-sample-size guard.
- `cyclops.monitor`: `DCTester.explain_shift()` explains a detected shift
  using SHAP on the domain classifier trained by `tester_method="classifier"`,
  returning features ranked by how strongly they indicate a sample belongs
  to the shifted distribution.

### Fixed

- `cyclops.monitor`: `errorfill()` crashed on the default `color=None` because
  matplotlib removed `ax._get_lines.prop_cycler`.
- `cyclops.monitor`: `TSTester.test_shift()` mutated `p_val_threshold` in
  place on every call, so the Bonferroni correction compounded across
  repeated calls (e.g. in `Detector`'s sweep loops) instead of being
  computed fresh each time; also fixed an `UnboundLocalError` when the
  input isn't a plain `np.ndarray`.
- `cyclops.monitor`: `ContextMMDWrapper` was missing the
  `preprocess_at_init` argument in its positional argument list to
  alibi-detect's `ContextMMDDrift`, silently shifting every later
  argument by one slot. `ContextMMDWrapper` and `LKWrapper` now pass
  keyword arguments so future alibi-detect signature changes fail loudly
  instead of silently misaligning.
- `cyclops.monitor`: `Reductor` raised `TypeError` when torchvision wasn't
  installed and `transforms` was passed, because `isinstance(transforms,
  Compose)` was called with `Compose is None`.
- `cyclops.monitor`: removed `plot_label_distribution`, an unreachable,
  untested, and uncalled function with a use-before-assignment bug.
- `cyclops.report`: `ModelCardReport.export()` raised `IndexError` when no
  `PerformanceMetric` had been logged.
- `cyclops.report`: `_process_metric_name()` raised `UnboundLocalError`
  for any metric `type` not prefixed with `Binary`/`Multiclass`/
  `Multilabel` (e.g. a custom metric name).
- `cyclops.report`: `export()`'s default output filename was a static
  `model_card.html`/`.json`, so repeated calls into the same
  `output_dir` silently overwrote prior reports and broke trend/history
  comparisons. The default filename is now timestamped per call.
- `cyclops.report`: `Citation.content` (raw BibTeX text) was rendered
  with Jinja's `|safe` filter, bypassing autoescaping for no reason.
- `cyclops.utils`: `exchange_extension()` dropped the filename entirely
  for paths with no existing extension (e.g. `"myfile"` -> `".csv"`
  instead of `"myfile.csv"`).

### Changed

- `cyclops.monitor.utils`: removed ~470 lines of unused temporal-modeling
  scaffolding (`Data`, `get_data`, `run_model`, `get_serving_data`,
  `scale`, `daterange`, `get_obj_from_str`, `load_model`/`save_model`,
  `print_metrics_binary`, `load_ckp`, `get_device`, `get_temporal_model`,
  `Loader`, and a stray `__main__` demo block) that was neither exported,
  imported elsewhere in the repo, nor tested.

### CI / infra

- Added a CodeQL workflow for Python static security scanning.
- Added a `uv` ecosystem entry to Dependabot so `pyproject.toml`/
  `uv.lock` dependencies get automated update PRs.
- Fixed the README's "integration tests" badge, which linked to a
  workflow file that no longer exists; replaced with the (existing,
  previously unlinked) unit tests badge.
- Expanded `CONTRIBUTING.md` with environment setup, test-running, and
  repository layout sections.

[Unreleased]: https://github.com/VectorInstitute/cyclops/compare/v0.2.12...HEAD
