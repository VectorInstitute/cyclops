
Developing on the GEMINI HPC
----------------------------

Installing dependencies
^^^^^^^^^^^^^^^^^^^^^^^

On the GEMINI HPC, the
`poetry installation method <https://vectorinstitute.github.io/cyclops/intro.html#using-poetry>`_
is the preferred one, since the conda install is currently broken.

Environment variables
^^^^^^^^^^^^^^^^^^^^^

To use querying features, which involves access to GEMINI's database, add the
GEMINI password to the environment variables:

.. code-block:: bash

   export PGPASSWORD=<your-gemini-db-password>

For convenience of use, add the above line to a ``.env`` file in the root of this repo.
The variable is automatically added to the configuration.
