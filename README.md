# Vector-Delirium

- To create conda environment:

` conda create --name <env> --file requirements.txt`

- To extract data from the database and save to csv:

`python3 datapipeline/process_data.py --user <db-user-name> -w --output <output-path>`

- To run dataset analysis, update gemini_analysis.conf configuration with preferred options. Run:

`python3 analysis.py`

- To train a model:

`python3 main.py --dataset gemini`
