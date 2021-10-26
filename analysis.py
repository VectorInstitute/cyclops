import pandas as pd
import argparse

from evidently.model_profile import Profile
from evidently.profile_sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab, ClassificationPerformanceTab

import datapipeline.config as conf

def analyze_dataset_drift(data, config):
    column_mapping = {}
    column_mapping['numerical_features'] = ['age']
    column_mapping['categorical_features'] = ['sex', 'los']
    analysis_columns = column_mapping['numerical_features'] + column_mapping['categorical_features']

    # prepare data - select only numeric and categorical features
    # pick specific years to compare
    reference_years = config.ref
    years_to_evaluate = config.eval
    reference_data = data.loc[data['year'].isin(reference_years), analysis_columns]
    reference_data = reference_data.dropna()

    # generate report for each slice
    # TODO: add option to do analysis for all years except reference ones
    drifts = []
    for eval_year in years_to_evaluate:
        eval_data = data.loc[data['year'] == eval_year, analysis_columns]

        eval_data = eval_data.dropna()

        drifts.append(eval_drift(eval_year, reference_data, eval_data, column_mapping, html=False))

    return drifts  

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

def analyze_model_drift(reference, test, config):
    column_mapping = {}

    column_mapping['target'] = 'los' #config.target TODO:fix
    column_mapping['prediction'] = 'prediction'
    column_mapping['numerical_features'] = config.numerical_features
    column_mapping['categorical_features'] = config.categorical_features

    perfomance_dashboard = Dashboard(tabs=[ClassificationPerformanceTab])
    perfomance_dashboard.calculate(reference, test, column_mapping=column_mapping)

    perfomance_dashboard.save("../performance_report.html")  # TODO: filename should be a parameter
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config", type=str, default="datapipeline/delirium.config")
    parser.add_argument("--type", type=str, default="dataset")
    args = parser.parse_args()

    config = conf.read_config(args.dataset_config)
    if args.type == "dataset":
        data = pd.read_csv(config.input)
        analyze_dataset_drift(data, config)
    else:
        reference = pd.read_csv(config.reference)
        test = pd.read_csv(config.test)
        analyze_model_drift(reference, test, config)
