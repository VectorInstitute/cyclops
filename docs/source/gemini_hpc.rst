
Developing on the GEMINI HPC
----------------------------

Environment variables
^^^^^^^^^^^^^^^^^^^^^

To use querying features, which involves access to GEMINI's database, add the
GEMINI password to the environment variables:

.. code-block:: bash

   export PGPASSWORD=<your-gemini-db-password>

For convenience of use, add the above line to a ``.env`` file in the root of this repo.
The variable is automatically added to the configuration.
