.. role:: raw-html-m2r(raw)
   :format: html


Vector-Delirium
===============

Table of Contents
-----------------


#. `Setup <#setup>`_

   * `Python virtual environment <#python>`_

     * `Using Anaconda/Miniconda <#conda>`_
     * `Using pip and venv <#pipvenv>`_

   * `Pre-installed virtual environment paths <#preinstalled>`_
   * `Environment variables <#envvar>`_

#. `Configuration Files <#config>`_
#. `Running as Pipeline <#pipeline>`_
#. `Pipeline Components <#components>`_

   * `Data Extraction <#data>`_
   * `Model Training <#training>`_
   * `Prediction <#prediction>`_
   * `Analysis <#analysis>`_

#. `Sample Notebooks <#notebooks>`_
#. `Framework Design <#design>`_

Setup :raw-html-m2r:`<a name="setup"></a>`
-----------------------------------------------

Python virtual environment :raw-html-m2r:`<a name="python"></a>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The development environment has been tested on the Gemini HPC server, using
``python = 3.8.5``. There are two ways to setup the environment and install
dependencies.

Using Anaconda/Miniconda :raw-html-m2r:`<a name="conda"></a>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create and activate environment, run:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate vector_delirium

Using pip and venv :raw-html-m2r:`<a name="pipvenv"></a>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create virtual environment and install dependencies, run:

.. code-block:: bash

   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt

Pre-installed virtual environment paths :raw-html-m2r:`<a name="preinstalled"></a>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pre-installed environments on the HPC are available. For conda environment:

.. code-block:: bash

   conda activate /mnt/nfs/project/delirium/dev_env/conda

For pipenv venv:

.. code-block:: bash

   source /mnt/nfs/project/delirium/dev_env/venv

Environment variables :raw-html-m2r:`<a name="envvar"></a>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add the following environment variables in order to run luigi pipelines: 

.. code-block:: bash

   export PGPASSWORD=<your-gemini-db-password>
   export PYTHONPATH="${PYTHONPATH}:${PWD}"

To do that, add a file named ``.env`` to the root of this repo and add
the above lines to the file. The variables are automatically added to the
configuration.

Configuration Files :raw-html-m2r:`<a name="config"></a>`
--------------------------------------------------------------

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

Edit the new configs, then the new configs can be passed to the main script using:

.. code-block:: bash

   python3 main.py -c configs/<name_of_experiment>/*.yaml

Luigi batch processing is used to run the whole workflow (data extract, predict,
analyze) as a pipeline. Luigi parameters are specified in
``configs/default/workflow.yaml``.

Running as Pipeline :raw-html-m2r:`<a name="pipeline"></a>`
----------------------------------------------------------------

Luigi batch processing is used to execute the whole pipeline containing
data extraction, prediction and analysis steps.

Prerequisites
^^^^^^^^^^^^^


* configure environment variables (above)
* trained model exists and ``model_path`` parameter in the ``configs/default/model.yaml``
  specifies the path
* analysis step requires reference data csv (that includes model predictions) 
* ensure that configuration in ``configs/default/analysis.yaml`` is up to date
  (has model path, csv file with reference data/predictions) 

To run pipeline once for specific time period, run:

.. code-block:: bash

   luigi --module pipeline Analysis --date-from 2018-08-01 --date-to 2018-10-01 --local-scheduler

Simulation runs ML pipeline for each month in the specified interval
(date-from to date-to). To run the simulation, run:

.. code-block:: bash

   luigi --module pipeline_simulation Simulation --date-from 2018-01-01 --date-to 2020-06-01 --local-scheduler

Pipeline Tasks :raw-html-m2r:`<a name="components"></a>`
-------------------------------------------------------------

In addition to Luigi pipeline, each of the components of the pipeline can be run
on it's own from command line or from Jupyter notebook.
To run each task from the command line:

.. code-block:: bash

   python3 main.py --<action> <optional parameter overwrites>

where ``<action>``\ : extract, train, predict, analyze
To get a full list of possible arguments, run:

.. code-block:: bash

   python3 main.py -h

Data Extraction  :raw-html-m2r:`<a name="data"></a>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Examples of running data extraction from command line:

1) To extract all available records from the database and save to csv
(or change parameters in the ``configs/default/data.yaml``\ ):

.. code-block:: bash

   python3 main.py --extract -r -w --output_folder './_out' --pop_size 0

2) To extract 20,000 records, save to file, split into train, test and val sets
by ``hospital_id`` column:

.. code-block:: bash

   python3 main.py --extract -r --pop_size 20000 -w --output_full_path='./data.csv' --split_column hospital_id --test_split 3 --val_split 7

Model Training  :raw-html-m2r:`<a name="training"></a>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To train a model:

.. code-block:: bash

   python3 main.py --train --input './data.csv'

Prediction  :raw-html-m2r:`<a name="prediction"></a>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run prediction:

.. code-block:: bash

   python3 main.py --predict --input './data.csv' --result_output './result.csv'

Analysis  :raw-html-m2r:`<a name="analysis"></a>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run dataset drift analysis, update ``configs/default/analysis.yaml`` configuration with
preferred options. If ``slice`` option is not specified, report compares the data
provided in ``reference`` and ``test`` files (specified in config file or as command
line options). 

.. code-block:: bash

   python3 main.py --analyze -html

Alternatively, can use ``slice`` option to specify a column to slice data for
analysis. In this case data is read from single file specified by ``input``
parameter and ``data_ref`` and ``data_eval`` need to be specified for the two slices
to be compared. For example,

.. code-block:: bash

   python3 main.py --analyze --slice year --data_ref 2015 --data_eval 2016 -html

To run performance analysis (result files should be first generated by running
prediction step):

.. code-block:: bash

   python3 main.py --analyze --type performance --reference './ref_results.csv' --test './test_results.csv -html

For both reports, html flag stands for ``generate HTML report``\ , if not provided
JSON report is generated instead.

Sample Notebooks  :raw-html-m2r:`<a name="notebooks"></a>`
---------------------------------------------------------------

To use the notebooks, the ``vector_delirium`` conda environment or the ``venv``
virtual environment can be installed and used inside an Ipython kernel. To use
the conda environment:

.. code-block:: bash

   conda activate <name> or <path/to/conda/env>
   python3 -m ipykernel install --user --name <name_of_kernel>

To use venv's virtual environment:

.. code-block:: bash

   source <path/to/venv>
   python3 -m ipykernel install --user --name <name_of_kernel>

Now, you can navigate to the notebook's ``Kernel`` tab and set it as
``<name_of_kernel>``.


* ``sample_code/data_extraction.ipynb`` - notebook to illustrate data extraction
* ``sample_code/training_demo.ipynb`` - sample model training notebook
* ``sample_code/mlflow_demo.ipynb`` - training progress and validation results
  are logged to MLFlow, this notebook illustrates how to use them to monitor
  training.
* ``sample_code/analysis_demo.ipynb`` - shows how to generate Evidently reports; as
  well to plot results of pipeline simulation

Framework Design :raw-html-m2r:`<a name="design"></a>`
-----------------------------------------------------------


.. raw:: html

   <p float="left">
     <img src="./assets/evaluation_framework_on_gemini.png" />
   </p>

