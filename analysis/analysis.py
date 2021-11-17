import pandas as pd
import argparse
import json
import time
import os
import configargparse

from evidently.model_profile import Profile
from evidently.profile_sections import DataDriftProfileSection, ClassificationPerformanceProfileSection
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab, ClassificationPerformanceTab

import mlflow

def read_config(file = False):
    if not file:
        parser = configargparse.ArgumentParser()
    else:
        parser = configargparse.ArgumentParser(default_config_files=[file])

    parser.add('-c', '--config_file', is_config_file=True,  default='/mnt/nfs/home/koshkinam/vector-delirium/config/gemini_analysis.cfg', help='config file path')
    parser.add_argument("--type", type=str, default="dataset", help='Type of report to generate')

    # data-specific parameters
    parser.add('--input', type=str, default=None, required=False, help='Data file to read from instead of database')
    parser.add('--slice', default='year', type=str, required=False,
           help='What column to use to slice data for analysis?')
    parser.add('--data_ref', default=[], type=int, action='append', required=False,
           help='List of slices to take as reference data')
    parser.add('--data_eval', default=[], type=int, action='append', required=False,
           help='List of slices to evaluate on')
    parser.add('--numerical_features', default=[], type=str, action='append', required=False,
           help='List of numerical features (for analysis)')
    parser.add('--categorical_features', default=[], type=str, action='append', required=False,
           help='List of categorical features (for analysis)')
    parser.add('--report_path', default='../', type=str, required=False, help='Where to store html report?')
    parser.add('--report_full_path', default = '', type=str, required=False, help="Full path for the report (filename is generated if not provided)")
    parser.add('-html', action='store_true', help='Produce HTML report (otherwise save json report)')

                                                       
    parser.add('--target', default='target', type=str, required=False,
               help='Column we are trying to predict')
    parser.add('-target_num', action='store_true', required=False,
               help='Is target numerical (as opposed to categorical)')
    parser.add('--prediction', default='prediction', type=str, required=False, help='Name of the prediction column')

    # model performance parameters
    parser.add('--reference', type=str, required=False, help='Filename of features/prediction to use as reference')
    parser.add('--test', type=str, required=False,
           help='Filename of features/prediction to use as test (for model drift evaluation)')

    args, unknown = parser.parse_known_args()

    return args

def get_report_filename(config):
    if len(config.report_full_path) == 0:
        t = time.localtime()
        date = time.strftime("%Y-%b-%d_%H-%M-%S", t)
        ext = 'html' if config.html else 'json'
        filename = os.path.join(config.report_path, f'{config.type}_report_{date}.{ext}')
    else:
        filename = config.report_full_path
    return filename

def analyze_dataset_drift(ref_data, eval_data, config):
    column_mapping = {}
    column_mapping['numerical_features'] = config.numerical_features
    column_mapping['categorical_features'] = config.categorical_features
    if config.target_num:
        column_mapping['numerical_features'] += [config.target]
    else:
       column_mapping['categorical_features'] += [config.target]
    analysis_columns = column_mapping['numerical_features'] + column_mapping['categorical_features']

    # prepare data - select only numeric and categorical features
    # pick specific slices to compare
    reference_data = ref_data[analysis_columns].dropna()
    eval_data = eval_data[analysis_columns].dropna()
      
    drift = eval_drift(reference_data, eval_data, column_mapping,config,  html=config.html)

    return drift  

#evaluate data drift with Evidently Profile
def eval_drift(reference, production, column_mapping, config, html=False):
    column_mapping['drift_conf_level'] = 0.95
    column_mapping['drift_features_share'] = 0.5
    data_drift_profile = Profile(sections=[DataDriftProfileSection])
    data_drift_profile.calculate(reference, production, column_mapping=column_mapping)
    report = data_drift_profile.json()
    json_report = json.loads(report)

    report_filename = get_report_filename(config)
    if html:
        dashboard = Dashboard(tabs=[DataDriftTab])
        dashboard.calculate(reference, production, column_mapping=column_mapping)
        dashboard.save(report_filename) #TODO: filename should be a parameter
    else:
        with open(report_filename, 'w') as f:
            json.dump(json_report, f)

    metrics = {'drifts':[], 'report_filename':report_filename, 'results':{}}
    results = json_report['data_drift']['data']['metrics'] 
    for feature in column_mapping['numerical_features'] + column_mapping['categorical_features']:
        metrics['drifts'].append((feature, results[feature]['p_value'])) 
    metrics['timestamp'] = json_report['timestamp']
    metrics['results']['n_features'] = results['n_features']
    metrics['results']['dataset_drift'] = 1 if results['dataset_drift'] else 0
    metrics['results']['n_drifted_features'] = results['n_drifted_features']
    return metrics, report_filename

# compare performance of the model on two sets of data
def analyze_model_drift(reference, test, config):
    column_mapping = {}

    column_mapping['target'] = config.target 
    column_mapping['prediction'] = config.prediction
    column_mapping['numerical_features'] = config.numerical_features
    column_mapping['categorical_features'] = config.categorical_features

    perfomance_profile = Profile(sections=[ClassificationPerformanceProfileSection])
    perfomance_profile.calculate(reference, test, column_mapping=column_mapping)
    report = perfomance_profile.json()
    json_report = json.loads(report)

    report_filename = get_report_filename(config)
    if config.html:
        perfomance_dashboard = Dashboard(tabs=[ClassificationPerformanceTab])
        perfomance_dashboard.calculate(reference, test, column_mapping=column_mapping)
        perfomance_dashboard.save(report_filename)
    else:
       with open(report_filename, 'w') as f:
           json.dump(json_report, f)
    
    metrics = {'results':{}, 'report_filename':report_filename}
    results = json_report['classification_performance']['data']['metrics']
    metrics['timestamp'] = json_report['timestamp']
    metrics['results'] = {'ref_accuracy': results['reference']['accuracy'], 'ref_f1': results['reference']['f1'],
        'ref_precision': results['reference']['precision'], 'ref_recall': results['reference']['recall'],
        'test_accuracy': results['current']['accuracy'], 'test_f1':  results['current']['f1'],
        'test_precision': results['current']['precision'], 'test_recall': results['current']['recall']}
    return metrics, report_filename

def log_to_mlflow(config, metrics):
    exp_name = 'DatasetAnalysis' if config.type == 'dataset' else 'ModelComparison'
    exp = mlflow.get_experiment_by_name(exp_name)
    with mlflow.start_run(experiment_id=exp.experiment_id): 
        mlflow.log_dict(vars(config), 'config.json')
        mlflow.log_artifact(metrics['report_filename'])
        mlflow.log_metrics(metrics['results'])
        mlflow.log_params({'timestamp':metrics['timestamp']})

def main(config):
    if config.type == "dataset":
        if len(config.slice) > 0:
            data = pd.read_csv(config.input)
            eval_data = data.loc[data[config.slice].isin(config.data_eval)]
            ref_data = data.loc[data[config.slice].isin(config.data_ref)]
        else:
           ref_data = pd.read_csv(config.reference)
           eval_data = pd.read_csv(config.test)

        metrics, fn  = analyze_dataset_drift(ref_data, eval_data, config)
    else:
        reference = pd.read_csv(config.reference)
        test = pd.read_csv(config.test)
        metrics, fn = analyze_model_drift(reference, test, config)
    # log results of analysis to mlflow
    log_to_mlflow(config, metrics)
    return fn

if __name__ == "__main__":
    config = read_config()
    main(config)
