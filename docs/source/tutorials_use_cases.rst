Example use cases
=================


Binary classification using tabular data
----------------------------------------


Kaggle Heart Failure Prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a binary classification problem where the goal is to predict
risk of heart disease. The `dataset <https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction>`_
is available on Kaggle. The dataset contains 11 features and 1 target
variable.


.. toctree::

    tutorials/kaggle/heart_failure_prediction.ipynb


Chest X-ray classification
--------------------------

The `CXRClassificationTask` task is a multi-label classification task that predicts the
presence of different thoracic diseases given a chest X-ray image.


NIH Chest X-ray dataset
^^^^^^^^^^^^^^^^^^^^^^^

This tutorial showcases the use of the ``tasks`` API to implement a chest X-ray
classification task. The dataset used is the `NIH Chest X-ray dataset <https://nihcc.app.box.com/v/ChestXray-NIHCC>`__, which contains 112,120 frontal-view X-ray images of 30,805 unique patients with 14 disease labels.

The tutorial also demonstrates the use of the ``evaluate`` API to evaluate the
performance of a model on the task.

.. toctree::

    tutorials/nihcxr/cxr_classification.ipynb
