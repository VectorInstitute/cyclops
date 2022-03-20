
************
Introduction
************

cyclops is a framework for facilitating research and deployment of ML models 
in the health (or clinical) setting. It provides three high-level features:

* Data extraction and processing
* Model training and evaluation support
* Drift detection toolkit

Setup
-----

Python virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

The development environment has been tested on ``python = 3.8.5``. 
There are two ways to setup the environment and install dependencies.

Using Anaconda/Miniconda
~~~~~~~~~~~~~~~~~~~~~~~~

To create and activate environment, run:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate cyclops

Using pip and venv
~~~~~~~~~~~~~~~~~~

To create virtual environment and install dependencies, run:

.. code-block:: bash

   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install poetry
   poetry install

Configuration Files
-------------------

There are four configuration files:

* ``configs/default/data.yaml``
* ``configs/default/model.yaml``
* ``configs/default/analysis.yaml``
* ``configs/default/workflow.yaml``

Each file contains the parameters for respective tasks
(\ ``data extraction``\ , ``model training and inference``\ ,
``analysis`` and ``workflows``\ ). The config parser script is ``config.py``.

Refer to ``configs/default`` for default configuration parameters. 
(Additional ones are described in ``config.py``\ ).

A copy of the config dir can be made for bootstrapping, for custom experiments
and pipelines. For example:

.. code-block:: bash

   cp -r configs/default configs/<name_of_experiment>

Edit the new configs, then the new configs can be passed to the main script using:

.. code-block:: bash

   python3 main.py -c configs/<name_of_experiment>/*.yaml

Example Notebooks
-----------------

To use the notebooks, the ``vector_delirium`` conda environment or the ``venv``
virtual environment can be installed and used inside an Ipython kernel. To use
the conda environment:

.. code-block:: bash

   conda activate <name> or <path/to/conda/env>
   python3 -m ipykernel install --user --name <name_of_kernel>

To use venv's virtual environment:

.. code-block:: bash

   source <path/to/venv>
   python3 -m ipykernel install --user --name <name_of_kernel>

Now, you can navigate to the notebook's ``Kernel`` tab and set it as
``<name_of_kernel>``.


* ``sample_code/data_layer.ipynb`` - Example to run data extraction and processing.
* ``sample_code/delirium.ipynb`` - Exploratory data analysis for developing a 
  delirium model.
