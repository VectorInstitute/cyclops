Example use cases
=================

Tabular data
------------

Kaggle Heart Failure Prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a binary classification problem where the goal is to predict
risk of heart disease. The `heart failure dataset <https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction>`_
is available on Kaggle. The dataset contains 11 features and 1 target
variable.

.. toctree::

    tutorials/kaggle/heart_failure_prediction.ipynb

MIMICIV Mortality Prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a binary classification problem where the goal is to predict
risk of in-hospital mortality. The `MIMICIV dataset <https://mimic.mit.edu/>`_ is an EHR dataset collected from a single hospital site, which includes ICU data.

.. toctree::

    tutorials/mimiciv/mortality_prediction.ipynb

Synthea Prolonged Length of Stay Prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a binary classification problem where the goal is to predict
whether a patient will have a prolonged length of stay in the hospital
(more than 7 days). The `synthea dataset <https://github.com/synthetichealth/synthea>`_
is generated using Synthea which is a synthetic patient generator. The dataset
contains observations, medications and procedures as features.

.. toctree::

    tutorials/synthea/los_prediction.ipynb

Diabetes 130-US Hospitals for Years 1999-2008 Readmission Prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a binary classification problem where the goal is to predict
risk of readmission. The `diabetes dataset <https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008>`_
is available on UCI Machine Learning Repository. The dataset contains
47 features and 1 target variable.

.. toctree::

    tutorials/diabetes_130/readmission_prediction.ipynb

Image data
----------

NIH Chest X-ray classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This tutorial showcases the use of the ``tasks`` API to implement a chest X-ray
classification task. The dataset used is the `NIH Chest X-ray dataset <https://nihcc.app.box.com/v/ChestXray-NIHCC>`__, which contains 112,120 frontal-view X-ray images of 30,805 unique patients with 14 disease labels.

The tutorial also demonstrates the use of the ``evaluate`` API to evaluate the
performance of a model on the task.

.. toctree::

    tutorials/nihcxr/cxr_classification.ipynb
