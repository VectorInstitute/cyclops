![cyclops Logo](https://github.com/VectorInstitute/cyclops/blob/main/docs/source/theme/static/cyclops_logo-dark.png?raw=true)

--------------------------------------------------------------------------------

[![Code checks](https://github.com/VectorInstitute/cyclops/actions/workflows/code_checks.yml/badge.svg)](https://github.com/VectorInstitute/cyclops/actions/workflows/code_checks.yml)
[![Documentation and Coverage Report](https://github.com/VectorInstitute/cyclops/actions/workflows/docs_deploy.yml/badge.svg)](https://github.com/VectorInstitute/cyclops/actions/workflows/docs_deploy.yml)

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

> ⚠️ ``poetry`` is the preferred installation method since it also installs
the ``cyclops`` package, and is tested. There is also an ``environment.yaml``
and ``requirements.txt`` to install dependencies using
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) or
[pip](https://pypi.org/project/pip/), however is not tested frequently.


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
