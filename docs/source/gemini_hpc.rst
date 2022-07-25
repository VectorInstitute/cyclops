
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

To use the querying API which involves access to GEMINI's database, add the
GEMINI password to a ``.env`` file in the root of this repo. The password is
automatically added to cyclops's configuration and used when running queries.

For security, ensure that the permissions are set such that the repo is accessible
to only you, hence preventing other users on the HPC from accessing your password!

.. code-block:: bash

   PGPASSWORD=<your-gemini-db-password>
