import pandas as pd
import argparse
import json
import time
import os

from evidently.model_profile import Profile
from evidently.profile_sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab, ClassificationPerformanceTab

import datapipeline.config as conf

def get_report_filname(config):
    t = time.localtime()
    date = time.strftime("%Y-%b-%d_%H-%M-%S", t)
    filename = os.path.join(config.report_path, f'{config.type}_report_{date}.html')
    return filename

def analyze_dataset_drift(data, config):
    column_mapping = {}
    column_mapping['numerical_features'] = config.numerical_features
    column_mapping['categorical_features'] = config.categorical_features
    analysis_columns = column_mapping['numerical_features'] + column_mapping['categorical_features']

    # prepare data - select only numeric and categorical features
    # pick specific slices to compare
    ref_slices = config.data_ref
    eval_slices = config.data_eval
    reference_data = data.loc[data[config.slice].isin(ref_slices), analysis_columns]
    reference_data = reference_data.dropna()

    eval_data = data.loc[data[config.slice].isin(eval_slices), analysis_columns]
    eval_data = eval_data.dropna()
    drift = eval_drift(reference_data, eval_data, column_mapping, html=True)

    return drift  

#evaluate data drift with Evidently Profile
def eval_drift(reference, production, column_mapping, html=False):
    column_mapping['drift_conf_level'] = 0.95
    column_mapping['drift_features_share'] = 0.5
    data_drift_profile = Profile(sections=[DataDriftProfileSection])
    data_drift_profile.calculate(reference, production, column_mapping=column_mapping)
    report = data_drift_profile.json()
    json_report = json.loads(report)

    if html:
        dashboard = Dashboard(tabs=[DataDriftTab])
        dashboard.calculate(reference, production, column_mapping=column_mapping)
        dashboard.save(get_report_filname(config)) #TODO: filename should be a parameter

    drifts = []
    for feature in column_mapping['numerical_features'] + column_mapping['categorical_features']:
        drifts.append((feature, json_report['data_drift']['data']['metrics'][feature]['p_value'])) 
    return drifts

# compare performance of the model on two sets of data
def analyze_model_drift(reference, test, config):
    column_mapping = {}

    column_mapping['target'] = config.target 
    column_mapping['prediction'] = config.prediction
    column_mapping['numerical_features'] = config.numerical_features
    column_mapping['categorical_features'] = config.categorical_features

    perfomance_dashboard = Dashboard(tabs=[ClassificationPerformanceTab])
    perfomance_dashboard.calculate(reference, test, column_mapping=column_mapping)

    perfomance_dashboard.save(get_report_filname(config))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config", type=str, default="datapipeline/delirium.config")
    parser.add_argument("--type", type=str, default="dataset")
    args = parser.parse_args()

    config = conf.read_config(args.dataset_config)
    config.type = args.type
    if args.type == "dataset":
        data = pd.read_csv(config.input)
        analyze_dataset_drift(data, config)
    else:
        reference = pd.read_csv(config.reference)
        test = pd.read_csv(config.test)
        analyze_model_drift(reference, test, config)
