# Vector-Delirium



## Environment Setup

To install prerequisites run:
`conda create --name <env> --file requirements.txt`

Environment Variables:
Add the following two environment variables to your profile in order to run full pipeline: 
```
export PGPASSWORD=<your-gemini-password>
export LUIGI_CONFIG_PATH="$HOME/vector-delirium/config/gemini_luigi.cfg"
export PYTHONPATH="${PYTHONPATH}:$HOME/vector-delirium/"
```

To do that edit .profile to add these three lines. The run it for changes to take effect:
`source .profile`

## Configuration Files:
All of the configutation files are kept in config/ folder. There is a configuration file for each step of the pipeline.  Dataset configuration file is used by multiple other tasks to define where the data is coming from.
Parameter parsing for dataset configuration is being done datapipeline/config.py.

## Luigi pipeline
Pipeline containing data extraction, prediction and analysis steps can be run by executing the following command:

To run the simulation, run:
```
luigi --module pipeline_simulation Simulation --date-from 2018-01-01 --date-to 2020-06-01 --local-scheduler
```
Simulation runs ML pipeline for each month in the specified interval (date-from to date-to)

To run pipeline once for specific time period, run:

`luigi --module pipeline Analysis --date-from 2018-08-01 --date-to 2018-10-01 --local-scheduler`

## Pipeline components:
In addition to Luigi pipeline, each of the components of the pipeline can be run on it's own from command line or from Jupyter notebook.

### Data Extraction:
To extract data from the database and save to csv (or change parameters in the datapipeline/delirium.conf):

`python3 process_data.py --user <db-user-name> -r -w --output <output-path>`

### Model Training: 
To train a model:
`python3 main.py --dataset gemini`

### Run prediction:
TODO

### Analysis
To run dataset analysis, update gemini_analysis.conf configuration with preferred options. Run:
`python3 analysis.py`

### Sample notebooks
MlFlow - TODO
Analysis - TODO
Training - TODO




