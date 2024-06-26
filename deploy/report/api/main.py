"""A light weight server for evaluation."""

import logging
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
from datasets.arrow_dataset import Dataset
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, validator

from cyclops.data.slicer import SliceSpec
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
TEMPLATES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
templates = Jinja2Templates(directory=TEMPLATES_PATH)


class EvaluationInput(BaseModel):
    """Input data for evaluation."""

    preds_prob: List[float] = Field(..., min_items=1)
    target: List[float] = Field(..., min_items=1)
    metadata: Dict[str, List[Any]] = Field(default_factory=dict)

    @classmethod
    @validator("preds_prob", "target")
    def check_list_length(
        cls, v: List[float], values: Dict[str, List[Any]], **kwargs: Any
    ) -> List[float]:
        """Check if preds_prob and target have the same length.

        Parameters
        ----------
        v : List[float]
            List of values.
        values : Dict[str, List[Any]]
            Dictionary of values.

        Returns
        -------
        List[float]
            List of values.

        Raises
        ------
        ValueError
            If preds_prob and target have different lengths.

        """
        if "preds_prob" in values and len(v) != len(values["preds_prob"]):
            raise ValueError("preds_prob and target must have the same length")
        return v

    @classmethod
    @validator("metadata")
    def check_metadata_length(
        cls, v: Dict[str, List[Any]], values: Dict[str, List[Any]], **kwargs: Any
    ) -> Dict[str, List[Any]]:
        """Check if metadata columns have the same length as preds_prob and target.

        Parameters
        ----------
        v : Dict[str, List[Any]]
            Dictionary of values.
        values : Dict[str, List[Any]]
            Dictionary of values.

        Returns
        -------
        Dict[str, List[Any]]
            Dictionary of values.

        Raises
        ------
        ValueError
            If metadata columns have different lengths than preds_prob and target.

        """
        if "preds_prob" in values:
            for column in v.values():
                if len(column) != len(values["preds_prob"]):
                    raise ValueError(
                        "All metadata columns must have the same length as preds_prob and target"
                    )
        return v


# This endpoint serves the UI
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request) -> HTMLResponse:
    """Return home page for cyclops model report app.

    Parameters
    ----------
    request : Request
        Request object.

    Returns
    -------
    HTMLResponse
        HTML response for home page

    """
    return templates.TemplateResponse("test_report.html", {"request": request})


@app.post("/evaluate")
async def evaluate_result(data: EvaluationInput) -> None:
    """Calculate metric and return result from request body.

    Parameters
    ----------
    data : EvaluationInput
        Evaluation input data.

    Raises
    ------
    HTTPException
        If there is an internal server error.

    """
    try:
        # Create a dictionary with all data
        df_dict = {
            "target": data.target,
            "preds_prob": data.preds_prob,
            **data.metadata,
        }
        # Create DataFrame
        df = pd.DataFrame(df_dict)
        _eval(df)
        LOGGER.info("Generated report.")
    except Exception as e:
        LOGGER.error(f"Error during evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/evaluate", response_class=HTMLResponse)
async def get_report(request: Request) -> HTMLResponse:
    """Return latest updated model report.

    Parameters
    ----------
    request : Request
        Request object.

    Returns
    -------
    HTMLResponse
        HTML response for model report

    """
    return templates.TemplateResponse("test_report.html", {"request": request})


def _export(report: ModelCardReport) -> None:
    """Prepare and export report file.

    Parameters
    ----------
    report : ModelCardReport
        ModelCardReport object

    """
    if not os.path.exists("./cyclops_report"):
        LOGGER.info("Creating report for the first time!")
    report_path = report.export(
        output_filename="test_report.html",
        synthetic_timestamp=str(datetime.today()),
        last_n_evals=3,
    )
    shutil.copy(f"{report_path}", TEMPLATES_PATH)


def _eval(df: pd.DataFrame) -> None:
    """Evaluate and return report.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with target, preds_prob and metadata columns

    """
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
    metric_collection = MetricDict(metrics)  # type: ignore
    spec_list = [
        {
            "Age": {
                "min_value": 30,
                "max_value": 50,
                "min_inclusive": True,
                "max_inclusive": False,
            },
        },
        {
            "Age": {
                "min_value": 50,
                "max_value": 70,
                "min_inclusive": True,
                "max_inclusive": False,
            },
        },
    ]
    slice_spec = SliceSpec(spec_list)
    result = evaluator.evaluate(
        dataset=data,
        metrics=metric_collection,
        slice_spec=slice_spec,
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
            "description": "The model was trained on the Kaggle Heart Failure Prediction Dataset to predict risk of heart failure.",
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
