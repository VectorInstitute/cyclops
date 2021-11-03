# Vector-Delirium

- To create conda environment:

` conda create --name <env> --file requirements.txt`

- To extract data from the database and save to csv (or change parameters in the datapipeline/delirium.conf):

`python3 process_data.py --user <db-user-name> -r -w --output <output-path>`

- To run dataset analysis, update gemini_analysis.conf configuration with preferred options. Run:

`python3 analysis.py`

- To train a model:

`python3 main.py --dataset gemini`
