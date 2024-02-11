Installation
============

Using pip
---------

.. code:: bash

   python3 -m pip install pycyclops

``cyclops`` has many optional dependencies that are used for specific functionality. For example, the `monai <https://github.com/Project-MONAI/MONAI>`__ library is used for loading DICOM images to create datasets. All optional dependencies can be installed with ``pycyclops[all]``, and specific sets of dependencies are listed in the sections below.

+-----------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------+
| Dependency                  | pip extra                | Notes                                                                                                         |
+=============================+==========================+===============================================================================================================+
| xgboost                     | xgboost                  | Allows use of `XGBoost <https://xgboost.readthedocs.io/en/stable/>`__ model                                   |
+-----------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------+
| torch                       | torch                    | Allows use of `PyTorch <https://pytorch.org/>`__ models                                                       |
+-----------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------+
| torchvision                 | torchvision              | Allows use of `Torchvision <https://pytorch.org/vision/stable/index.html>`__ library                          |
+-----------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------+
| torchxrayvision             | torchxrayvision          | Uses `TorchXRayVision <https://mlmed.org/torchxrayvision/>`__ library                                         |
+-----------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------+
| monai                       | monai                    | Uses `MONAI <https://github.com/Project-MONAI/MONAI>`__ to load and transform images                          |
+-----------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------+
| alibi                       | alibi                    | Uses `Alibi <https://docs.seldon.io/projects/alibi/en/stable/>`__ for additional explainability functionality |
+-----------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------+
| alibi-detect                | alibi-detect             | Uses `Alibi Detect <https://docs.seldon.io/projects/alibi-detect/en/stable/>`__ for dataset shift detection   |
+-----------------------------+--------------------------+---------------------------------------------------------------------------------------------------------------+
