import pandas as pd
import numpy as np
import json
import os
import time

import datapipeline.config as conf
import datapipeline.extraction as ex

def save_data(data, config, format='csv'):
    if (format != 'csv'):
        print("Unsupported format {}".format(format))
        exit
    t = time.localtime()
    date = time.strftime("%Y-%b-%d_%H-%M-%S", t)
    file_name = os.path.join(config.output, f'admin_data_{date}.csv')
    data.to_csv(file_name)
    return file_name

def prune_columns(config_columns, data):
    columns = []

    #print(config_columns, data.columns)

    for c in config_columns:
        if c in list(data.columns):
           columns.append(c)
   
    return columns 

def get_splits (config, data):
    # drop columns not used in training
    all_columns = prune_columns(config.features+config.target,data)
 
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
    
if __name__=="__main__":
    config = conf.read_config()
    conf.write_config(config)
    pipeline(config)


