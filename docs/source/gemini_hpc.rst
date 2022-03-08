
Developing on the GEMINI HPC
----------------------------

Pre-installed virtual environment paths
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pre-installed environments on the HPC are available. For conda environment:

.. code-block:: bash

   conda activate /mnt/nfs/project/delirium/dev_env/conda

For pipenv venv:

.. code-block:: bash

   source /mnt/nfs/project/delirium/dev_env/venv

Environment variables
^^^^^^^^^^^^^^^^^^^^^

Add the following environment variables in order to use the framework: 

.. code-block:: bash

   export PGPASSWORD=<your-gemini-db-password>
   export PYTHONPATH="${PYTHONPATH}:${PWD}"

To do that, add a file named ``.env`` to the root of this repo and add
the above lines to the file. The variables are automatically added to the
configuration.
