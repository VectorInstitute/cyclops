# Changelog

All notable changes to this project will be documented in this file.

Currently this project is early in development, so the semantic versioning is more to just keep track of development changes, not releases.

## [0.0.1] - 2022-01-13 (6e5b1a00cddfe4cdaf58b41d82ad948a204dbc1d)
This is first development version after Maria's handover of codebase.

### Added
- Code formatting using black
- Cleanup README
- Installation of dependencies using both pip and Conda

### Fixed
- Jupyter notebooks sourcing environment variables using python-dotenv package
- postgres username automatically sourced using $USER env variable


## [0.0.2] - 2022-02-01 (d397ae701a2b67ad023f0d4fd4c9d6d015e55d29)

### Added
- Add pre-commit hooks for static code analysis, unit tests
- Add docs generation using Sphinx, just uses README now, API docs yet to add
- Move luigi pipeline stuff to cyclops.workflow (yet to be integrated, so
this feature is broken ATM.
- Add basic SQL querier, works for admin+diagnosis data
- Move data-pipeline utils functions to cyclops.processors, clean up some
functions, and add tests for them.
- Add logging, wherever it makes sense. Freeze requirements.

### Changed
- Split config files, so it becomes easier to manage.
- Move model implementation, and training/validation specific scripts to models.
