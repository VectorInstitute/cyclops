![cyclops Logo](https://github.com/VectorInstitute/cyclops/blob/main/docs/source/theme/static/cyclops_logo-dark.png?raw=true)

--------------------------------------------------------------------------------

[![PyPI](https://img.shields.io/pypi/v/pycyclops)](https://pypi.org/project/pycyclops)
[![Code checks](https://github.com/VectorInstitute/cyclops/actions/workflows/code_checks.yml/badge.svg)](https://github.com/VectorInstitute/cyclops/actions/workflows/code_checks.yml)
[![docs](https://github.com/VectorInstitute/cyclops/actions/workflows/docs_deploy.yml/badge.svg)](https://github.com/VectorInstitute/cyclops/actions/workflows/docs_deploy.yml)
[![codecov](https://codecov.io/gh/VectorInstitute/cyclops/branch/main/graph/badge.svg)](https://codecov.io/gh/VectorInstitute/cyclops)
[![license](https://img.shields.io/github/license/VectorInstitute/cyclops.svg)](https://github.com/VectorInstitute/cyclops/blob/main/LICENSE)

``cyclops`` is a framework for facilitating research and deployment of ML models
in the health (or clinical) setting. It provides three high-level features:


* Data Querying and Processing
* Rigorous Evaluation
* Drift Detection toolkit

``cyclops`` also provides a library of use-cases on clinical datasets. The implemented
use-cases include:

* Mortality decompensation prediction


## 🐣 Getting Started

### Setup Python virtual environment and install dependencies

The development environment has been tested on ``python = 3.9.7``.

The python virtual environment can be setup using
[poetry](https://python-poetry.org/docs/#installation). Hence, make sure it is
installed and then run:

```bash
poetry install
source $(poetry env info --path)/bin/activate
```

## 📚 [Documentation](https://vectorinstitute.github.io/cyclops/)

## 🎓 Notebooks

To use jupyter notebooks, the python virtual environment can be installed and
used inside an Ipython kernel. After activating the virtual environment, run:

```bash
python3 -m ipykernel install --user --name <name_of_kernel>
```

Now, you can navigate to the notebook's ``Kernel`` tab and set it as
``<name_of_kernel>``.

Tutorial notebooks in ``tutorials`` can be useful to view the
functionality of the framework.
