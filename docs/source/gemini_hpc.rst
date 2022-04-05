
Developing on the GEMINI HPC
----------------------------

Pre-installed virtual environment paths
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For pipenv venv:

.. code-block:: bash

   source /mnt/nfs/project/delirium/dev_env/cyclops-uzNue7B5-py3.8/bin/activate

Environment variables
^^^^^^^^^^^^^^^^^^^^^

To use querying features, which involves access to GEMINI's database, add the
GEMINI password to the environment variables:

.. code-block:: bash

   export PGPASSWORD=<your-gemini-db-password>

For convenience of use, add the above line to a ``.env`` file in the root of this repo.
The variable is automatically added to the configuration.
