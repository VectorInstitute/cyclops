
************
Introduction
************

cyclops is a framework for facilitating research and deployment of ML models
in the health (or clinical) setting. It provides three high-level features:

* Data extraction and processing
* Model training and evaluation support
* Drift detection toolkit

Getting Started
---------------

Setup Python virtual environment and install dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The development environment has been tested on ``python = 3.9.7``.


The python virtual environment can also be setup using
`poetry <https://python-poetry.org/docs/#installation>`_. Hence, make sure it is
installed and then run:

.. code-block:: bash

    poetry install
    source $(poetry env info --path)/bin/activate


.. note::
    ``poetry`` is the preferred installation method since it also installs
    the ``cyclops`` package, and is tested. There is also an ``environment.yaml``
    and ``requirements.txt`` to install dependencies using
    `conda <https://docs.conda.io/en/latest/miniconda.html>`_ or
    `pip <https://pypi.org/project/pip/>`_ however is not tested frequently.


Notebooks
---------

To use jupyter notebooks, the python virtual environment can be installed and
used inside an Ipython kernel. After activating the virtual environment, run:

.. code-block:: bash

   python3 -m ipykernel install --user --name <name_of_kernel>

Now, you can navigate to the notebook's ``Kernel`` tab and set it as
``<name_of_kernel>``.

Tutorial notebooks in ``notebooks/tutorials`` can be useful to view the
functionality of the framework.
