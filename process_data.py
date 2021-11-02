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

def get_splits (config, data):
    # drop columns not used in training
    all_columns = config.features + config.target
   
    #TODO: do I need to further clean any NaN values?
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
        data = ex.split(data)

    # persist processed data
    if config.w:
        save_data(data, config)

    return data
    
if __name__=="__main__":
    config = conf.read_config()
    conf.write_config(config)
    pipeline(config)


