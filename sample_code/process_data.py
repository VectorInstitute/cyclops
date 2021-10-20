import pandas as pd
import numpy as np
import json
import os
import time

import config as conf
import extraction as ex

import plotly.offline as py #working offline
import plotly.graph_objs as go

from evidently.model_profile import Profile
from evidently.profile_sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab

#evaluate data drift with Evidently Profile
def eval_drift(label, reference, production, column_mapping, html=False):
    column_mapping['drift_conf_level'] = 0.95
    column_mapping['drift_features_share'] = 0.5
    data_drift_profile = Profile(sections=[DataDriftProfileSection])
    data_drift_profile.calculate(reference, production, column_mapping=column_mapping)
    report = data_drift_profile.json()
    json_report = json.loads(report)

    if html:
        dashboard = Dashboard(tabs=[DataDriftTab])
        dashboard.calculate(reference, production, column_mapping=column_mapping)
        dashboard.save("../data_drif_report.html") #TODO: filename should be a parameter

    drifts = [('label', label)]
    for feature in column_mapping['numerical_features'] + column_mapping['categorical_features']:
        drifts.append((feature, json_report['data_drift']['data']['metrics'][feature]['p_value'])) 
    return drifts

def analyze(data, config):
    column_mapping = {}
    column_mapping['numerical_features'] = ['age']
    column_mapping['categorical_features'] = ['sex', 'los']
    analysis_columns = column_mapping['numerical_features'] + column_mapping['categorical_features']
    
    #prepare data - select only numeric and categorical features
    #pick specific years to compare
    reference_years = config.ref
    years_to_evaluate = config.eval
    reference_data = data.loc[data['year'].isin(reference_years), analysis_columns]
    reference_data = reference_data.dropna()
    

    # generate report for each slice
    # TODO: add option to do analysis for all years except reference ones
    drifts = []
    for eval_year in years_to_evaluate:
        eval_data =  data.loc[data['year']==eval_year, analysis_columns]

        eval_data = eval_data.dropna()
    
        drifts.append(eval_drift (eval_year, reference_data, eval_data, column_mapping, html=False))

    return drifts   

def save_data(data, config, format='csv'):
    if (format != 'csv'):
        print("Unsupported format {}".format(format))
        exit
    t = time.localtime()
    date = time.strftime("%Y-%b-%d_%H-%M-%S", t)
    file_name = os.path.join(config.output, f'admin_data_{date}.csv')
    data.to_csv(file_name)

    
if __name__=="__main__":
    config = conf.read_config()
    conf.write_config(config)

    data = ex.extract(config)
    data = ex.transform(data)
    if config.w:
        save_data(data, config)

    if config.a:
        drifts = analyze(data, config)
        print(drifts)

