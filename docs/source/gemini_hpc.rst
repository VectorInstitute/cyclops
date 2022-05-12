
Using cyclops on the GEMINI HPC
-------------------------------

Setup Python virtual environment and install dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The GEMINI HPC environment has a separate private repository to fetch and
install python packages. Hence, add the following lines to the ``pyproject.toml``
file before trying to setup the python virtual environment and install dependencies,
(i.e. before running the ``poetry install`` command).

.. code-block:: bash

    [[tool.poetry.source]]
    name = "gemini_repository"
    url = "https://packages.gemini-hpc.ca/repository/pypi/simple"
    default = true

Environment variables
^^^^^^^^^^^^^^^^^^^^^

To use querying features, which involves access to GEMINI's database, add the
GEMINI password to the environment variables:

.. code-block:: bash

   export PGPASSWORD=<your-gemini-db-password>

For convenience of use, add the above line to a ``.env`` file in the root of this repo.
The variable is automatically added to the configuration.
