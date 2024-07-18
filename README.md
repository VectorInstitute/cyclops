![cyclops Logo](https://github.com/VectorInstitute/cyclops/blob/main/docs/source/_static/cyclops_logo-dark.png?raw=true)

--------------------------------------------------------------------------------

[![PyPI](https://img.shields.io/pypi/v/pycyclops)](https://pypi.org/project/pycyclops)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pycyclops)
[![code checks](https://github.com/VectorInstitute/cyclops/actions/workflows/code_checks.yml/badge.svg)](https://github.com/VectorInstitute/cyclops/actions/workflows/code_checks.yml)
[![integration tests](https://github.com/VectorInstitute/cyclops/actions/workflows/integration_tests.yml/badge.svg)](https://github.com/VectorInstitute/cyclops/actions/workflows/integration_tests.yml)
[![docs](https://github.com/VectorInstitute/cyclops/actions/workflows/docs.yml/badge.svg)](https://github.com/VectorInstitute/cyclops/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/VectorInstitute/cyclops/branch/main/graph/badge.svg)](https://codecov.io/gh/VectorInstitute/cyclops)
[![docker](https://github.com/VectorInstitute/cyclops/actions/workflows/docker.yml/badge.svg)](https://hub.docker.com/r/vectorinstitute/cyclops)

``cyclops`` is a toolkit for facilitating research and deployment of ML models for healthcare. It provides a few high-level APIs namely:

* `data` - Create datasets for training, inference and evaluation. We use the popular 🤗 [datasets](https://github.com/huggingface/datasets) to efficiently load and slice different modalities of data
* `models` - Use common model implementations using [scikit-learn](https://scikit-learn.org/stable/) and [PyTorch](https://pytorch.org/)
* `tasks` - Use common ML task formulations such as binary classification or multi-label classification on tabular, time-series and image data
* `evaluate` - Evaluate models on clinical prediction tasks
* `monitor` - Detect dataset shift relevant for clinical use cases
* `report` - Create [model report cards](https://vectorinstitute.github.io/cyclops/api/tutorials/kaggle/heart_failure_report_periodic.html) for clinical ML models

``cyclops`` also provides example end-to-end use case implementations on clinical datasets such as

* [NIH chest x-ray](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community)
* [MIMIC-IV](https://physionet.org/content/mimiciv/2.0/)


## 🐣 Getting Started

### Installing cyclops using pip

```bash
python3 -m pip install pycyclops
```

`cyclops` has many optional dependencies that are used for specific functionality. For
example, the [monai](https://github.com/Project-MONAI/MONAI) library is used for loading
DICOM images to create datasets. Hence, `monai` can be installed using
``python3 -m pip install pycyclops[monai]``. Specific sets of dependencies are listed
below.


| Dependency       | pip extra       | Notes                                                                                                        |
| ----------       | ---------       | -----                                                                                                        |
| xgboost          | xgboost         | Allows use of [XGBoost](https://xgboost.readthedocs.io/en/stable/) model                                     |
| torch            | torch           | Allows use of [PyTorch](https://pytorch.org/) models                                                         |
| torchvision      | torchvision     | Allows use of [Torchvision](https://pytorch.org/vision/stable/index.html) library                            |
| torchxrayvision  | torchxrayvision | Uses [TorchXRayVision](https://mlmed.org/torchxrayvision/) library                                           |
| monai            | monai           | Uses [MONAI](https://github.com/Project-MONAI/MONAI) to load and transform images                            |
| alibi            | alibi           | Uses [Alibi](https://docs.seldon.io/projects/alibi/en/stable/) for additional explainability functionality   |
| alibi-detect     | alibi-detect    | Uses [Alibi Detect](https://docs.seldon.io/projects/alibi-detect/en/stable/) for dataset shift detection     |


## 🧑🏿‍💻 Developing

### Using poetry

The development environment can be set up using
[poetry](https://python-poetry.org/docs/#installation). Hence, make sure it is
installed and then run:

```bash
python3 -m poetry install
source $(poetry env info --path)/bin/activate
```

In order to install dependencies for testing (codestyle, unit tests, integration tests),
run:

```bash
python3 -m poetry install --with test
```

API documentation is built using [Sphinx](https://www.sphinx-doc.org/en/master/) and
can be locally built by:

```bash
python3 -m poetry install --with docs
cd docs
make html SPHINXOPTS="-D nbsphinx_allow_errors=True"
```

### Contributing

Contributing to cyclops is welcomed.
See [Contributing](https://vectorinstitute.github.io/cyclops/api/contributing.html) for
guidelines.


## 📚 [Documentation](https://vectorinstitute.github.io/cyclops/)

If you need to build the documentations locally, make sure to install ``Pandoc`` in addition to ``docs`` poetry group.


## 📓 Notebooks

To use jupyter notebooks, the python virtual environment can be installed and
used inside an IPython kernel. After activating the virtual environment, run:

```bash
python3 -m ipykernel install --user --name <name_of_kernel>
```

Now, you can navigate to the notebook's ``Kernel`` tab and set it as
``<name_of_kernel>``.


## 🎓 Citation

Reference to cite when you use `cyclops` in a project or a research paper:

```
@article {Krishnan2022.12.02.22283021,
	author = {Krishnan, Amrit and Subasri, Vallijah and McKeen, Kaden and Kore, Ali and Ogidi, Franklin and Alinoori, Mahshid and Lalani, Nadim and Dhalla, Azra and Verma, Amol and Razak, Fahad and Pandya, Deval and Dolatabadi, Elham},
	title = {CyclOps: Cyclical development towards operationalizing ML models for health},
	elocation-id = {2022.12.02.22283021},
	year = {2022},
	doi = {10.1101/2022.12.02.22283021},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2022/12/08/2022.12.02.22283021},
	journal = {medRxiv}
}
```
