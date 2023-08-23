Clinical use cases
==================


In-hospital mortality prediction
--------------------------------

The `in_hospital_mortality` task is a binary classification task that
predicts whether a patient (subgroup) will die in the hospital given the EMR data
collected during the first N hours of their stay in the hospital.


Kaggle MIMIC-III dataset
^^^^^^^^^^^^^^^^^^^^^^^^

This tutorial showcases the use of the ``tasks`` API to implement a mortality prediction
task. The dataset is based on a `subset <https://www.kaggle.com/datasets/saurabhshahane/in-hospital-mortality-prediction>`__
of `MIMIC-III dataset <https://physionet.org/content/mimiciii/1.4/>`__, which is a
large, freely-available database comprising deidentified health-related data of patients
who were admitted to the intensive care unit (ICU) at a large tertiary care hospital.

The tutorial also demonstrates the use of the ``evaluate`` API to evaluate the
performance of a model on the task.


.. toctree::

    tutorials/mimiciii/mortality_prediction.ipynb
    tutorials/nihcxr/cxr_classification.ipynb
