🧑🏿‍💻 Developing
=======================

Using poetry
------------

The development environment can be set up using `poetry <https://python-poetry.org/docs/#installation>`__. Hence, make sure it is installed and then run:

.. code:: bash

   python3 -m poetry install
   source $(poetry env info --path)/bin/activate

In order to install dependencies for testing (codestyle, unit tests, integration tests), run:

.. code:: bash

   python3 -m poetry install --with test

API documentation is built using `Sphinx <https://www.sphinx-doc.org/en/master/>`__ and can be locally built by:

.. code:: bash

   python3 -m poetry install --with docs
   cd docs
   make html SPHINXOPTS="-D nbsphinx_allow_errors=True"


If you need to build the documentations locally, make sure to install ``Pandoc`` in addition to ``docs`` poetry group.

Contributing
------------

Contributing to cyclops is welcomed. See `Contributing <https://vectorinstitute.github.io/cyclops/api/contributing.html>`__ for guidelines.
