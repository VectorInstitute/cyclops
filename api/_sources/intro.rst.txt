.. figure::
   https://github.com/VectorInstitute/cyclops/blob/main/docs/source/theme/static/cyclops_logo-dark.png?raw=true
   :alt: cyclops Logo

--------------

|PyPI| |PyPI - Python Version| |code checks| |integration tests| |docs|
|codecov| |docker| |license|

``cyclops`` is a toolkit for facilitating research and deployment of ML
models for healthcare. It provides a few high-level APIs namely:

-  ``data`` - Create datasets for training, inference and evaluation. We
   use the popular 🤗
   `datasets <https://github.com/huggingface/datasets>`__ to efficiently
   load and slice different modalities of data
-  ``models`` - Use common model implementations using
   `scikit-learn <https://scikit-learn.org/stable/>`__ and
   `PyTorch <https://pytorch.org/>`__
-  ``tasks`` - Use common ML task formulations such as binary
   classification or multi-label classification on tabular, time-series
   and image data
-  ``evaluate`` - Evaluate models on clinical prediction tasks
-  ``monitor`` - Detect dataset shift relevant for clinical use cases
-  ``report`` - Create `model report
   cards <https://vectorinstitute.github.io/cyclops/api/tutorials/nihcxr/nihcxr_report_periodic.html>`__
   for clinical ML models

``cyclops`` also provides example end-to-end use case implementations on
clinical datasets such as

-  `NIH chest
   x-ray <https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community>`__
-  `MIMIC-IV <https://physionet.org/content/mimiciv/2.0/>`__

🐣 Getting Started
==================

Installing cyclops using pip
----------------------------

.. code:: bash

   python3 -m pip install pycyclops

The base cyclops installation doesn’t include modelling packages.

To install additional dependencies for using models,

.. code:: bash

   python3 -m pip install 'pycyclops[models]'

🧑🏿‍💻 Developing
=======================

Using poetry
------------

The development environment can be set up using
`poetry <https://python-poetry.org/docs/#installation>`__. Hence, make
sure it is installed and then run:

.. code:: bash

   python3 -m poetry install
   source $(poetry env info --path)/bin/activate

In order to install dependencies for testing (codestyle, unit tests,
integration tests), run:

.. code:: bash

   python3 -m poetry install --with test

API documentation is built using
`Sphinx <https://www.sphinx-doc.org/en/master/>`__ and can be locally
built by:

.. code:: bash

   python3 -m poetry install --with docs
   cd docs
   make html SPHINXOPTS="-D nbsphinx_allow_errors=True"

Contributing
------------

Contributing to cyclops is welcomed. See
`Contributing <https://vectorinstitute.github.io/cyclops/api/intro.html>`__
for guidelines.

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
.. |PyPI - Python Version| image:: https://img.shields.io/pypi/pyversions/pycyclops
.. |code checks| image:: https://github.com/VectorInstitute/cyclops/actions/workflows/code_checks.yml/badge.svg
   :target: https://github.com/VectorInstitute/cyclops/actions/workflows/code_checks.yml
.. |integration tests| image:: https://github.com/VectorInstitute/cyclops/actions/workflows/integration_tests.yml/badge.svg
   :target: https://github.com/VectorInstitute/cyclops/actions/workflows/integration_tests.yml
.. |docs| image:: https://github.com/VectorInstitute/cyclops/actions/workflows/docs_deploy.yml/badge.svg
   :target: https://github.com/VectorInstitute/cyclops/actions/workflows/docs_deploy.yml
.. |codecov| image:: https://codecov.io/gh/VectorInstitute/cyclops/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/VectorInstitute/cyclops
.. |docker| image:: https://github.com/VectorInstitute/cyclops/actions/workflows/docker.yml/badge.svg
   :target: https://hub.docker.com/r/vectorinstitute/cyclops
.. |license| image:: https://img.shields.io/github/license/VectorInstitute/cyclops.svg
   :target: https://github.com/VectorInstitute/cyclops/blob/main/LICENSE
