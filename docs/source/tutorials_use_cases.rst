Example use cases
=================



------------------------

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

    tutorials/kaggle/heart_failure_prediction.ipynb
    tutorials/nihcxr/cxr_classification.ipynb


Chest X-ray classification
--------------------------

The `chest_xray` task is a multi-label classification task that predicts the
presence of different thoracic diseases given a chest X-ray image.


NIH Chest X-ray dataset
^^^^^^^^^^^^^^^^^^^^^^^

This tutorial showcases the use of the ``tasks`` API to implement a chest X-ray
classification task. The dataset used is the `NIH Chest X-ray dataset <https://nihcc.app.box.com/v/ChestXray-NIHCC>`__, which contains 112,120 frontal-view X-ray images of 30,805 unique patients with 14 disease labels.

The tutorial also demonstrates the use of the ``evaluate`` API to evaluate the
performance of a model on the task.
