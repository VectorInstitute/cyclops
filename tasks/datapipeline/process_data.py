import pandas as pd
import numpy as np
import json
import os
import time

import tasks.datapipeline.extraction as ex

def save_data(data, config, format='csv'):
    if (format != 'csv'):
        print("Unsupported format {}".format(format))
        exit
    if config.output_full_path == None or len(config.output_full_path) == 0:
        t = time.localtime()
        date = time.strftime("%Y-%b-%d_%H-%M-%S", t)
        name = f'admin_data_{date}.csv'
        path = os.path.join(config.output_folder, name)
    else: 
        path = config.output_full_path
    data.to_csv(path)
    return path

def prune_columns(config_columns, data):
    columns = []

    #print(config_columns, data.columns)

    for c in config_columns:
        if c not in list(data.columns):
           data[c]=0
   
    return data

def get_splits (config, data):
    # drop columns not used in training
    target_list = config.target if isinstance(config.target, list) else [config.target]
    all_columns =  config.features + target_list
    data = prune_columns(all_columns, data)
   
 
    train = data.loc[data['train']==1, all_columns].dropna()
    test = data.loc[data['test'] == 1, all_columns].dropna()
    val = data.loc[data['val'] == 1, all_columns].dropna()

    return train, val, test

def pipeline (config):
    if not config.r:
        # read data from file
        try:
            with open(config.input) as f:
                data = pd.read_csv(config.input)
        except:
            print('Error: unable to read file {}'.format(config.input))
    else:
        # read data from database
        data = ex.extract(config)
        data = ex.transform(data)
        if len(config.split_column) > 0:
            data = ex.split(data, config)

    # persist processed data
    filepath = ''
    if config.w:
        filepath = save_data(data, config)

    return data, filepath


