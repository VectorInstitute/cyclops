"""A light weight server for evaluation."""

import logging
import shutil
from datetime import datetime

import pandas as pd
from datasets.arrow_dataset import Dataset
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing_extensions import Annotated

from cyclops.evaluate import evaluator
from cyclops.evaluate.metrics import create_metric
from cyclops.evaluate.metrics.experimental import MetricDict
from cyclops.report.plot.classification import ClassificationPlotter
from cyclops.report.report import ModelCardReport
from cyclops.report.utils import flatten_results_dict
from cyclops.utils.log import setup_logging


LOGGER = logging.getLogger(__name__)
setup_logging(print_level="WARN", logger=LOGGER)


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# This endpoint serves the UI
@app.get("/", response_class=HTMLResponse)
async def get_home():
    """Return home page for using evaluate API."""
    return templates.TemplateResponse("index.html", {"request": {"method": "POST"}})


@app.post("/evaluate")
async def evaluate_result(
    preds_prob: Annotated[str, Form()], target: Annotated[str, Form()]
):
    """Calculate metric and return result from request body."""
    preds_prob = [float(num.strip()) for num in preds_prob.split(",")]
    target = [float(num.strip()) for num in target.split(",")]

    # Evaluate and generate report file
    df = pd.DataFrame(data={"target": target, "preds_prob": preds_prob})
    _eval(df)
    LOGGER.info("Generated report.")
    return templates.TemplateResponse(
        "test_report.html", {"request": {"method": "GET"}}
    )


@app.get("/evaluate")
async def get_result():
    """Calculate metric and return result from request body."""
    return templates.TemplateResponse(
        "test_report.html", {"request": {"method": "GET"}}
    )


def _export(report: ModelCardReport):
    """Prepare and export report file."""
    report_path = report.export(
        output_filename="test_report.html",
        synthetic_timestamp=str(datetime.today()),
        last_n_evals=3,
    )
    shutil.copy(f"{report_path}", "./static")
    shutil.rmtree("./cyclops_report")


def _eval(df: pd.DataFrame):
    """Evaluate and return report."""
    report = ModelCardReport()
    data = Dataset.from_pandas(df)
    metric_names = [
        "binary_accuracy",
        "binary_precision",
        "binary_recall",
        "binary_f1_score",
    ]
    metrics = [
        create_metric(metric_name, experimental=True) for metric_name in metric_names
    ]
    metric_collection = MetricDict(metrics)
    result = evaluator.evaluate(
        dataset=data,
        metrics=metric_collection,  # type: ignore[list-item]
        target_columns="target",
        prediction_columns="preds_prob",
    )
    results_flat = flatten_results_dict(results=result)

    # Log into report
    for name, metric in results_flat["model_for_preds_prob"].items():
        split, name = name.split("/")  # noqa: PLW2901
        descriptions = {
            "BinaryPrecision": "The proportion of predicted positive instances that are correctly predicted.",
            "BinaryRecall": "The proportion of actual positive instances that are correctly predicted. Also known as recall or true positive rate.",
            "BinaryAccuracy": "The proportion of all instances that are correctly predicted.",
            "BinaryF1Score": "The harmonic mean of precision and recall.",
        }
        report.log_quantitative_analysis(
            "performance",
            name=name,
            value=metric.tolist(),
            description=descriptions[name],
            metric_slice=split,
            pass_fail_thresholds=0.6,
            pass_fail_threshold_fns=lambda x, threshold: bool(x >= threshold),
        )

    # Log plot in report
    plotter = ClassificationPlotter(task_type="binary", class_names=["0", "1"])
    plotter.set_template("plotly_white")

    # Extracting the overall classification metric values.
    overall_performance = {
        metric_name: metric_value
        for metric_name, metric_value in result["model_for_preds_prob"][
            "overall"
        ].items()
        if metric_name not in ["BinaryROC", "BinaryPrecisionRecallCurve"]
    }
    # Plotting the overall classification metric values.
    overall_performance_plot = plotter.metrics_value(
        overall_performance,
        title="Overall Performance",
    )
    report.log_plotly_figure(
        fig=overall_performance_plot,
        caption="Overall Performance",
        section_name="quantitative analysis",
    )
    report.log_from_dict(
        data={
            "name": "Heart Failure Prediction Model",
            "description": "The model was trained on the Kaggle Heart Failure \
        Prediction Dataset to predict risk of heart failure.",
        },
        section_name="model_details",
    )
    report.log_version(
        version_str="0.0.1",
        date=str(datetime.today().date()),
        description="Initial Release",
    )
    report.log_owner(
        name="CyclOps Team",
        contact="vectorinstitute.github.io/cyclops/",
        email="cyclops@vectorinstitute.ai",
    )
    report.log_license(identifier="Apache-2.0")
    report.log_reference(
        link="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html",  # noqa: E501
    )
    report.log_from_dict(
        data={
            "users": [
                {"description": "Hospitals"},
                {"description": "Clinicians"},
            ],
        },
        section_name="considerations",
    )
    report.log_user(description="ML Engineers")
    report.log_use_case(
        description="Predicting risk of heart failure.",
        kind="primary",
    )
    _export(report)
