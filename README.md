# Vector-Delirium

## Table of Contents
1. [Setup](#setup)
2. [Configuration Files](#config)
3. [Running as Pipeline](#pipeline)
4. [Pipeline Components](#components)
    * [Data Extraction](#data)
    * [Model Training](#training)
    * [Prediction](#prediction)
    * [Analysis](#analysis)
    * [Sample Notebooks](#notebooks)


## Setup: <a name="setup"></a>

To install prerequisites run:
`conda create --name <env> --file requirements.txt`

Environment Variables:
Add the following two environment variables to your profile in order to run full pipeline: 
```
export PGPASSWORD=<your-gemini-db-password>
export LUIGI_CONFIG_PATH="$HOME/vector-delirium/config/gemini_luigi.cfg"
export PYTHONPATH="${PYTHONPATH}:$HOME/vector-delirium/"
```

To do that edit .profile to add these three lines. The run it for changes to take effect:
`source .profile`

## Configuration Files: <a name="config"></a>
All of the configutation files are kept in config/ folder. There is a configuration file for each step of the pipeline.  Dataset configuration file is used by multiple other tasks to define where the data is coming from.
Parameter parsing for dataset configuration is being done datapipeline/config.py.

Note, specify your Gemini database username in config/gemini_data.cfg:
`username = <your-gemini-db-username>`

## Running as Pipeline: <a name="pipeline"></a>
Pipeline containing data extraction, prediction and analysis steps can be run by executing the following command:

To run the simulation, run:

`luigi --module pipeline_simulation Simulation --date-from 2018-01-01 --date-to 2020-06-01 --local-scheduler`

Simulation runs ML pipeline for each month in the specified interval (date-from to date-to)

To run pipeline once for specific time period, run:

`luigi --module pipeline Analysis --date-from 2018-08-01 --date-to 2018-10-01 --local-scheduler`

## Pipeline Components: <a name="components"></a>
In addition to Luigi pipeline, each of the components of the pipeline can be run on it's own from command line or from Jupyter notebook.

### Data Extraction:  <a name="data"></a>
To extract data from the database and save to csv (or change parameters in the config/gemini_data.cfg):

`python3 process_data.py -r -w --output <output-path>`

### Model Training:  <a name="training"></a>
To train a model:

`python3 train.py --dataset gemini`

### Prediction:  <a name="prediction"></a>
TODO

### Analysis:  <a name="analysis"></a>
To run dataset analysis, update config/gemini_analysis.cfg configuration with preferred options. Run:

`python3 analysis.py`

## Sample Notebooks:  <a name="notebooks"></a>
MlFlow - TODO
Analysis - TODO
Training - TODO




