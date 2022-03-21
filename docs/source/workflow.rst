workflow
========


Configuration Files
^^^^^^^^^^^^^^^^^^^

There are four configuration files:

* ``configs/default/data.yaml``
* ``configs/default/model.yaml``
* ``configs/default/analysis.yaml``
* ``configs/default/workflow.yaml``

Each file contains the parameters for respective tasks
(\ ``data extraction``\ , ``model training and inference``\ ,
``analysis`` and ``workflows``\ ). The config parser script is ``config.py``.

Refer to ``configs/default`` for default configuration parameters. 
(Additional ones are described in ``config.py``\ ).

A copy of the config dir can be made for bootstrapping, for custom experiments
and pipelines. For example:

.. code-block:: bash

   cp -r configs/default configs/<name_of_experiment>
