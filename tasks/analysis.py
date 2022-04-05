"""Analysis script."""

# pylint: disable-all

import json
import os
import time

import mlflow
import pandas as pd
from evidently.dashboard import Dashboard
from evidently.model_profile import Profile
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.profile_sections import (
    ClassificationPerformanceProfileSection,
    DataDriftProfileSection,
)
from evidently.tabs import ClassificationPerformanceTab, DataDriftTab

from cyclops.config import config_to_dict


def get_report_filename(config):
    """Get report filename."""
    if len(config.report_full_path) == 0:
        curr_time = time.localtime()
        date = time.strftime("%Y-%b-%d_%H-%M-%S", curr_time)
        ext = "html" if config.html else "json"
        filename = os.path.join(
            config.report_path, f"{config.type}_report_{date}.{ext}"
        )
    else:
        filename = config.report_full_path
    return filename


def analyze_dataset_drift(ref_data, eval_data, config):
    """Run drift analysis."""
    column_mapping = ColumnMapping()
    column_mapping.numerical_features = config.numerical_features
    column_mapping.categorical_features = config.categorical_features
    if config.target_num:
        column_mapping.numerical_features = config.numerical_features + [config.target]
    else:
        column_mapping.categorical_features = config.categorical_features + [
            config.target
        ]
    analysis_columns = (
        column_mapping.numerical_features + column_mapping.categorical_features
    )

    # prepare data - select only numeric and categorical features
    # pick specific slices to compare
    reference_data = ref_data[analysis_columns].dropna()
    eval_data = eval_data[analysis_columns].dropna()

    drift, func = eval_drift(
        reference_data, eval_data, column_mapping, config, html=config.html
    )

    return drift, func


# evaluate data drift with Evidently Profile
def eval_drift(reference, production, column_mapping, config, html=False):
    """Run eval drift."""
    column_mapping.drift_conf_level = 0.95
    column_mapping.drift_features_share = 0.5
    data_drift_profile = Profile(sections=[DataDriftProfileSection()])
    data_drift_profile.calculate(reference, production, column_mapping=column_mapping)
    report = data_drift_profile.json()
    json_report = json.loads(report)

    report_filename = get_report_filename(config)
    if html:
        dashboard = Dashboard(tabs=[DataDriftTab()])
        dashboard.calculate(reference, production, column_mapping=column_mapping)
        dashboard.save(report_filename)
    else:
        with open(report_filename, "w") as file_handle:
            json.dump(json_report, file_handle)

    metrics = {"drifts": [], "report_filename": report_filename, "results": {}}
    results = json_report["data_drift"]["data"]["metrics"]
    for feature in (
        column_mapping.numerical_features + column_mapping.categorical_features
    ):
        metrics["drifts"].append((feature, results[feature]["p_value"]))
    metrics["timestamp"] = json_report["timestamp"]
    metrics["results"]["n_features"] = results["n_features"]
    metrics["results"]["dataset_drift"] = 1 if results["dataset_drift"] else 0
    metrics["results"]["n_drifted_features"] = results["n_drifted_features"]
    return metrics, report_filename


# compare performance of the model on two sets of data
def analyze_model_drift(reference, test, config):
    """Analyze model drift."""
    column_mapping = ColumnMapping()

    column_mapping.target = config.target
    column_mapping.prediction = config.prediction_col
    column_mapping.numerical_features = config.numerical_features
    column_mapping.categorical_features = config.categorical_features

    perfomance_profile = Profile(sections=[ClassificationPerformanceProfileSection()])
    perfomance_profile.calculate(reference, test, column_mapping=column_mapping)
    report = perfomance_profile.json()
    json_report = json.loads(report)

    report_filename = get_report_filename(config)
    if config.html:
        perfomance_dashboard = Dashboard(tabs=[ClassificationPerformanceTab()])
        perfomance_dashboard.calculate(reference, test, column_mapping=column_mapping)
        perfomance_dashboard.save(report_filename)
    else:
        with open(report_filename, "w") as file_handle:
            json.dump(json_report, file_handle)

    metrics = {"results": {}, "report_filename": report_filename}
    results = json_report["classification_performance"]["data"]["metrics"]
    metrics["timestamp"] = json_report["timestamp"]
    metrics["results"] = {
        "ref_accuracy": results["reference"]["accuracy"],
        "ref_f1": results["reference"]["f1"],
        "ref_precision": results["reference"]["precision"],
        "ref_recall": results["reference"]["recall"],
        "test_accuracy": results["current"]["accuracy"],
        "test_f1": results["current"]["f1"],
        "test_precision": results["current"]["precision"],
        "test_recall": results["current"]["recall"],
    }
    return metrics, report_filename


def log_to_mlflow(config, metrics):
    """Log to mlflow."""
    exp_name = "DatasetAnalysis" if config.type == "dataset" else "ModelComparison"
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        mlflow.create_experiment(exp_name)
        exp = mlflow.get_experiment_by_name(exp_name)

    with mlflow.start_run(experiment_id=exp.experiment_id):
        mlflow.log_dict(config_to_dict(config), "config.json")
        mlflow.log_artifact(metrics["report_filename"])
        mlflow.log_metrics(metrics["results"])
        mlflow.log_params({"timestamp": metrics["timestamp"]})


def main(config):
    """Run analysis."""
    if config.type == "dataset":
        if config.slice is not None and len(config.slice) > 0:
            data = pd.read_csv(config.input)
            eval_data = data.loc[data[config.slice].isin(config.data_eval)]
            ref_data = data.loc[data[config.slice].isin(config.data_ref)]
        else:
            ref_data = pd.read_csv(config.reference)
            eval_data = pd.read_csv(config.test)

        metrics, func = analyze_dataset_drift(ref_data, eval_data, config)
    else:
        reference = pd.read_csv(config.reference)
        test = pd.read_csv(config.test)
        metrics, func = analyze_model_drift(reference, test, config)
    # log results of analysis to mlflow
    log_to_mlflow(config, metrics)

    return func
