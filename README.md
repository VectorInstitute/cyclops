
# cyclops
---------

cyclops is a framework for facilitating research and deployment of ML models 
in the health (or clinical) setting. It provides three high-level features:

* Data extraction and processing
* Model training and evaluation support
* Drift detection toolkit


## Getting Started

### Setup Python virtual environment

The development environment has been tested on ``python = 3.8.5``. 
There are two ways to setup the environment and install dependencies.

#### Using Anaconda/Miniconda

To create and activate environment, run:

```bash
conda env create -f environment.yml
conda activate cyclops
```

#### Using pip, venv and poetry

To create virtual environment and install dependencies, run:

```bash
python3 -m venv <path/to/virtual/environment>
source <path/to/virtual/environment>/bin/activate
pip install --upgrade pip
pip install poetry
poetry install
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
