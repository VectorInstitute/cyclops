# cyclops
---------

cyclops is a framework for facilitating research and deployment of ML models
in the health (or clinical) setting. It provides three high-level features:

* Data extraction and processing
* Model training and evaluation support
* Drift detection toolkit


## Getting Started

### Setup Python virtual environment

The development environment has been tested on Python version `3.8.5`.

[Install](https://github.com/pyenv/pyenv#installation) `pyenv` if not already installed.

Assuming there is already some version of `python3` and `pyenv` installed, we can then set the Python version which will be used in the virutal environment,
```
# Install the desired Python version
pyenv install <PYTHON VERSION>

# Set/verify this version in the global environment
pyenv global <PYTHON VERSION>
python3 -V
```

There are two ways to setup the virutal environment and install package dependencies.

#### Using Anaconda/Miniconda

Assuming [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is installed, create and activate a `conda` environment,
```bash
conda env create -f environment.yml
conda activate cyclops
```

#### Using pip/Poetry

Assuming there is already some version of `pip` installed, we can create and activate a virtual environment,
```
# Install the virtualenv package
python3 -m pip install --upgrade pip
python3 -m pip install virtualenv

# Create virtual environment at desired path
python3 -m venv <PATH>

# Enter virtual environment
source <PATH>/bin/activate
```

Installing package dependencies is made easy with Poetry, which we can also install with `pip`,
```
# Install the Poetry package
python3 -m pip install poetry

# Install repository packages with Poetry
cd <CYCLOPS REPOSITORY>
poetry install
```

In the future, any updates to the packages can be made with
```
poetry update
```

## [Documentation](https://vectorinstitute.github.io/cyclops/)

## Notebooks

To use jupyter notebooks, the python virtual environment can be installed and
used inside an Ipython kernel. After activating the virtual environment, run:

```bash
python3 -m ipykernel install --user --name <name_of_kernel>
```

Now, you can navigate to the notebook's ``Kernel`` tab and set it as
``<name_of_kernel>``.

Tutorial notebooks in ``notebooks/tutorials`` can be useful to view the
functionality of the framework.
