.. figure:: https://github.com/VectorInstitute/cyclops/blob/main/docs/source/theme/static/cyclops_logo-dark.png?raw=true
   :alt: cyclops Logo

--------------

|PyPI| |code checks| |integration tests| |docs| |codecov| |license|

``cyclops`` is a framework for facilitating research and deployment of
ML models in the health (or clinical) setting. It provides a few
high-level APIs namely:

-  ``query`` - Querying EHR databases (such as MIMIC-IV)
-  ``process`` - Process static and temporal EHR data
-  ``evaluate`` - Evaluate models on clinical prediction tasks
-  ``monitor`` - Detect data drift relevant for clinical use cases

``cyclops`` also provides a library of use-cases on clinical datasets.
The implemented use cases include:

-  Mortality decompensation prediction

🐣 Getting Started
==================

Installing cyclops using pip
----------------------------

.. code:: bash

   python3 -m pip install pycyclops

The core package only includes support for the ``process`` API. To
install support for ``query``, ``evaluate`` and ``monitor`` APIs,
install them as additional dependencies.

To install with ``query`` API support,

.. code:: bash

   python3 -m pip install 'pycyclops[query]'

To install with ``evaluate`` API support,

.. code:: bash

   python3 -m pip install 'pycyclops[evaluate]'

To install with ``monitor`` API support,

.. code:: bash

   python3 -m pip install 'pycyclops[monitor]'

🧑🏿‍💻 Developing
=======================

The development environment has been tested on ``python = 3.9``.

The python virtual environment can be set up using
`poetry <https://python-poetry.org/docs/#installation>`__. Hence, make
sure it is installed and then run:

.. code:: bash

   python3 -m poetry install
   source $(poetry env info --path)/bin/activate

Contributing
------------

Contributing to cyclops is welcomed. See
`Contributing <CONTRIBUTING.md>`__ for guidelines.

📚 `Documentation <https://vectorinstitute.github.io/cyclops/>`__
=================================================================

📓 Notebooks
============

To use jupyter notebooks, the python virtual environment can be
installed and used inside an IPython kernel. After activating the
virtual environment, run:

.. code:: bash

   python3 -m ipykernel install --user --name <name_of_kernel>

Now, you can navigate to the notebook’s ``Kernel`` tab and set it as
``<name_of_kernel>``.

Tutorial notebooks in ``docs/source/tutorials`` can be useful to view
the functionality of the framework.

🎓 Citation
===========

Reference to cite when you use CyclOps in a project or a research paper:

::

   @article {Krishnan2022.12.02.22283021,
       author = {Krishnan, Amrit and Subasri, Vallijah and McKeen, Kaden and Kore, Ali and Ogidi, Franklin and Alinoori, Mahshid and Lalani, Nadim and Dhalla, Azra and Verma, Amol and Razak, Fahad and Pandya, Deval and Dolatabadi, Elham},
       title = {CyclOps: Cyclical development towards operationalizing ML models for health},
       elocation-id = {2022.12.02.22283021},
       year = {2022},
       doi = {10.1101/2022.12.02.22283021},
       publisher = {Cold Spring Harbor Laboratory Press},
       URL = {https://www.medrxiv.org/content/early/2022/12/08/2022.12.02.22283021},
       journal = {medRxiv}
   }

.. |PyPI| image:: https://img.shields.io/pypi/v/pycyclops
   :target: https://pypi.org/project/pycyclops
.. |code checks| image:: https://github.com/VectorInstitute/cyclops/actions/workflows/code_checks.yml/badge.svg
   :target: https://github.com/VectorInstitute/cyclops/actions/workflows/code_checks.yml
.. |integration tests| image:: https://github.com/VectorInstitute/cyclops/actions/workflows/integration_tests.yml/badge.svg
   :target: https://github.com/VectorInstitute/cyclops/actions/workflows/integration_tests.yml
.. |docs| image:: https://github.com/VectorInstitute/cyclops/actions/workflows/docs_deploy.yml/badge.svg
   :target: https://github.com/VectorInstitute/cyclops/actions/workflows/docs_deploy.yml
.. |codecov| image:: https://codecov.io/gh/VectorInstitute/cyclops/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/VectorInstitute/cyclops
.. |license| image:: https://img.shields.io/github/license/VectorInstitute/cyclops.svg
   :target: https://github.com/VectorInstitute/cyclops/blob/main/LICENSE
