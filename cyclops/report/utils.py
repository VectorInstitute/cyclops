"""Utility functions for the `cyclops.report` module."""

import glob
import importlib
import inspect
import json
import os
from datetime import date as dt_date
from datetime import datetime as dt_datetime
from datetime import timedelta
from re import sub
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np

from cyclops.report.model_card import ModelCard  # type: ignore[attr-defined]
from cyclops.report.model_card.fields import (
    ComparativeMetrics,
    Graphic,
    PerformanceMetric,
    Test,
)


def str_to_snake_case(string: str) -> str:
    """Convert a string to snake_case.

    Parameters
    ----------
    string : str
        The string to convert.

    Returns
    -------
    str
        The converted string.

    Examples
    --------
    >>> str_to_snake_case("HelloWorld")
    'hello_world'
    >>> str_to_snake_case("Hello-World")
    'hello_world'
    >>> str_to_snake_case("Hello_World")
    'hello_world'
    >>> str_to_snake_case("Hello World")
    'hello_world'
    >>> str_to_snake_case("hello_world")
    'hello_world'

    """
    return "_".join(
        sub(
            "([A-Z][a-z]+)",
            r" \1",
            sub("([A-Z]+)", r" \1", string.replace("-", " ")),
        ).split(),
    ).lower()


def _raise_if_not_dict_with_str_keys(data: Any) -> None:
    """Raise an error if `data` is not a dictionary with string keys.

    Parameters
    ----------
    data : Any
        The data to check.

    Raises
    ------
    TypeError
        If `data` is not a dictionary with string keys.

    """
    if not (isinstance(data, Mapping) and all(isinstance(key, str) for key in data)):
        raise TypeError(f"Expected a dictionary with string keys. Got {data} instead.")


def _object_is_in_model_card_module(obj: object) -> bool:
    """Check if an object is defined in the same module as `ModelCard`.

    Parameters
    ----------
    obj : Any
        The object to check.

    Returns
    -------
    bool
        Whether or not the object is defined in the same module as `ModelCard`.

    """
    model_card_module = importlib.import_module(ModelCard.__module__)
    model_card_classes = inspect.getmembers(model_card_module, inspect.isclass)
    for name, model_card_class in model_card_classes:
        # match name or class
        if model_card_class.__module__ == ModelCard.__module__ and (
            obj.__class__.__name__ == name or obj.__class__ == model_card_class
        ):
            return True
    return False


def flatten_results_dict(  # noqa: PLR0912
    results: Dict[str, Dict[str, Dict[str, Any]]],
    remove_metrics: Optional[Union[str, List[str]]] = None,
    remove_slices: Optional[Union[str, List[str]]] = None,
    model_name: Optional[str] = None,
) -> Union[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Flatten a results dictionary. Needed for logging metrics and trends.

    Parameters
    ----------
    results : Dict[str, Dict[str, Dict[str, Any]]]
        The evaluation results dictionary to flatten.
    remove_metrics : Union[str,List[str]], optional
        Metric names to remove from the results, by default None
    remove_slices : Union[str,List[str]], optional
        Slice names to remove from the results, by default None
    model_name : str, optional
        Model name to filter results for, if None, return all models,
        by default None

    Returns
    -------
    Union[Dict[str, Dict[str, Any]], Dict[str, Any]]
        Dictionary of flattened results per model. \
            If `model_name` is not None,returns a dictionary \
            of flattened results for that model.

    """
    if isinstance(remove_metrics, str):
        remove_metrics = [remove_metrics]
    if remove_metrics is None:
        remove_metrics = []

    if isinstance(remove_slices, str):
        remove_slices = [remove_slices]
    if remove_slices is None:
        remove_slices = []

    results_flat = {}
    if model_name:
        assert model_name in results, f"Model name {model_name} not found in results."
        model_results = results[model_name]
        for slice_name, slice_results in model_results.items():
            for metric_name, metric_value in slice_results.items():
                if (
                    metric_name not in remove_metrics
                    and slice_name not in remove_slices
                ):
                    results_flat[f"{slice_name}/{metric_name}"] = metric_value

    else:
        for name, model_results in results.items():
            results_flat[name] = {}
            for slice_name, slice_results in model_results.items():
                for metric_name, metric_value in slice_results.items():
                    if (
                        metric_name not in remove_metrics
                        and slice_name not in remove_slices
                    ):
                        results_flat[name][f"{slice_name}/{metric_name}"] = metric_value

    return results_flat


def filter_results(
    results: List[Dict[str, Any]],
    slice_names: Optional[Union[str, List[str]]] = None,
    metric_names: Optional[Union[str, List[str]]] = None,
) -> List[Dict[str, Any]]:
    """Filter results by slice and metric names.

    Parameters
    ----------
    results : List[Dict[str, Any]]
        The results to filter as a list of dictionaries with keys \
        type, slice, and value.
    slice_names : Union[str, List[str]], optional
        Names of slices to filter by, if None, return all slices, \
        by default None
    metric_names : Union[str, List[str]], optional
        Names of metrics to filter by, if None, return all metrics, \
        by default None

    Returns
    -------
    List[Dict[str, Any]]
        List of filtered results.

    """
    if isinstance(slice_names, str):
        slice_names = [slice_names]

    if isinstance(metric_names, str):
        metric_names = [metric_names]

    return [
        d
        for d in results
        if (metric_names is None or d["type"] in metric_names)
        and (slice_names is None or d["slice"] in slice_names)
    ]


def extract_performance_metrics(
    root_directory: str,
    slice_names: Optional[Union[str, List[str]]] = None,
    metric_names: Optional[Union[str, List[str]]] = None,
    keep_timestamps: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract performance metrics from previous model cards.

    Parameters
    ----------
    root_directory : str
        Directory to search for model cards.
    slice_names : Union[str, List[str]], optional
        Name of slices to extract metrics for, if None, return all slices, \
        by default None
    metric_names : Union[str, List[str]], optional
        Name of metrics to extract, if None, return all metrics, \
        by default None
    keep_timestamps : bool, optional
        Whether or not to keep timestamps in the results keys, \
            by default False

    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        Dictionary of performance metrics per date with keys \
            of the form `YYYY-MM-DD` and values of lists of dictionaries \
            with keys type, slice, and value.

    """
    metrics_dict = {}
    json_files = glob.glob(f"{root_directory}/**/model_card.json", recursive=True)

    assert len(json_files) > 0, "No model cards found. Check the root directory."

    for file_path in sorted(json_files):
        time_string = os.path.basename(os.path.dirname(file_path))
        date_string = os.path.basename(os.path.dirname(os.path.dirname(file_path)))

        with open(file_path, "r", encoding="utf8") as file:
            data = json.load(file)

        quantitative_analysis = data.get("quantitative_analysis", {})
        performance_metrics = quantitative_analysis.get("performance_metrics", {})
        performance_metrics = filter_results(
            performance_metrics,
            slice_names=slice_names,
            metric_names=metric_names,
        )
        if len(performance_metrics) > 0:
            if keep_timestamps:
                metrics_dict[f"{date_string}: {time_string}"] = performance_metrics
            else:
                # If there are multiple model cards for the same date, \
                # only keep the most recent one
                metrics_dict[date_string] = performance_metrics
    return metrics_dict


def get_metrics_trends(
    report_directory: str,
    flat_results: Dict[str, Any],
    keep_timestamps: bool = False,
    slice_names: Optional[Union[str, List[str]]] = None,
    metric_names: Optional[Union[str, List[str]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Get the trends of the metrics over time to plot.

    Parameters
    ----------
    report_directory : str
        Directory to search for previous model cards.
    flat_results : Dict[str, Any]
        Dictionary of flattened results with keys of the form \
        slice_name/metric_name.
    slice_names : Union[str, List[str]], optional
        Names of slices to filter by, if None, return all slices, \
    by default None
    metric_names : Union[str, List[str]], optional
        Names of metrics to filter by, if None, return all metrics, \
        by default None
    keep_timestamps : bool, optional
        Whether or not to keep timestamps in the results keys, \
        by default False

    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        Dictionary of performance metrics per date with keys \
        date(or data and time) and values of lists of \
        dictionaries with keys slice, type, and value.

    """
    performance_history = extract_performance_metrics(
        report_directory,
        keep_timestamps=keep_timestamps,
        slice_names=slice_names,
        metric_names=metric_names,
    )

    assert (
        len(performance_history) > 0
    ), "No performance history found. Check slice and metric names."

    performance_recent = []

    for metric_name, metric_value in flat_results.items():
        name_split = metric_name.split("/")
        if len(name_split) == 1:
            slice_name = "overall"
            metric_name = name_split[0]  # noqa: PLW2901
        else:  # everything before the last slash is the slice name
            slice_name = "/".join(name_split[:-1])
            metric_name = name_split[-1]  # noqa: PLW2901

        data = {"type": metric_name, "value": metric_value, "slice": slice_name}
        performance_recent.append(data)

    performance_recent = filter_results(
        performance_recent,
        slice_names=slice_names,
        metric_names=metric_names,
    )
    assert (
        len(performance_recent) > 0
    ), "No performance metrics found. Check slice and metric names."
    today = dt_date.today().strftime("%Y-%m-%d")
    now = dt_datetime.now().strftime("%H-%M-%S")

    if keep_timestamps:
        performance_history[f"{today}: {now}"] = performance_recent
    else:
        performance_history[today] = performance_recent
    return performance_history


def sweep_tests(model_card: Any, tests: List[Any]) -> None:
    """Sweep model card to find all instances of Test.

    Parameters
    ----------
    model_card : Any
        The model card to sweep.
    tests : List[Any]
        The list to append all tests to.
    """
    for field in model_card:
        if isinstance(field, ComparativeMetrics):
            continue
        if isinstance(field, tuple):
            field = field[1]  # noqa: PLW2901
            if isinstance(field, ComparativeMetrics):
                continue
        if isinstance(field, Test):
            tests.append(field)
        if hasattr(field, "__fields__"):
            sweep_tests(field, tests)
        if isinstance(field, list) and len(field) != 0:
            for item in field:
                if isinstance(item, Test):
                    if len(field) == 1:
                        tests.append(field[0])
                    else:
                        tests.append(field)
                else:
                    sweep_tests(item, tests)


def sweep_metrics(model_card: Any, metrics: List[Any]) -> None:
    """Sweep model card to find all instances of PerformanceMetric.

    Parameters
    ----------
    model_card : Any
        The model card to sweep.
    tests : List[Any]
        The list to append all tests to.
    """
    for field in model_card:
        if isinstance(field, ComparativeMetrics):
            continue
        if isinstance(field, tuple):
            field = field[1]  # noqa: PLW2901
            if isinstance(field, ComparativeMetrics):
                continue
        if isinstance(field, PerformanceMetric):
            metrics.append(field)
        if hasattr(field, "__fields__"):
            sweep_metrics(field, metrics)
        if isinstance(field, list) and len(field) != 0:
            for item in field:
                if isinstance(item, PerformanceMetric):
                    if len(field) == 1:
                        metrics.append(field[0])
                    else:
                        metrics.append(field)
                else:
                    sweep_metrics(item, metrics)


def sweep_graphics(model_card: Any, graphics: list[Any], caption: str) -> None:
    """Sweep model card to find all instances of Graphic with a given caption.

    Parameters
    ----------
    model_card : Any
        The model card to sweep.
    graphics : List[Any]
        The list to append all graphics to.
    caption : str
        The caption to match.
    """
    for field in model_card:
        if isinstance(field, tuple):
            field = field[1]  # noqa: PLW2901
        if isinstance(field, Graphic) and field.name == caption:
            graphics.append(field)
        if hasattr(field, "__fields__"):
            sweep_graphics(field, graphics, caption)
        if isinstance(field, list) and len(field) != 0:
            for item in field:
                if isinstance(item, Graphic):
                    if item.name == caption:
                        graphics.append(item)
                else:
                    sweep_graphics(item, graphics, caption)


def compare_tests_metrics(  # noqa: PLR0912
    baseline_tests: List[Test],
    periodic_tests: List[Test],
    baseline_metrics: List[PerformanceMetric],
    periodic_metrics: List[PerformanceMetric],
    baseline_timestamp: str,
    periodic_timestamp: str,
    report_type: str,
) -> Dict[str, Any]:
    """Compare baseline and periodic tests and metrics."""
    baseline_passed = []
    periodic_passed = []
    for test in baseline_tests:
        baseline_passed.append(test.passed)
    for test in periodic_tests:
        periodic_passed.append(test.passed)
    baseline_pass_fail = np.array(baseline_passed)
    periodic_pass_fail = np.array(periodic_passed)
    baseline_fail_rate = 1 - (baseline_pass_fail.sum() / baseline_pass_fail.shape[0])
    fail_rate = 1 - (periodic_pass_fail.sum() / periodic_pass_fail.shape[0])
    fail_rate_change = fail_rate - baseline_fail_rate

    time_diff = dt_datetime.strptime(
        periodic_timestamp,
        "%Y-%m-%d",
    ) - dt_datetime.strptime(baseline_timestamp, "%Y-%m-%d")
    time_diff_string = get_time_diff_string(time_diff)

    new_tests_passed = []
    new_tests_failed = []
    for ptest in periodic_tests:
        for btest in baseline_tests:
            if ptest.name == btest.name:
                if btest.passed and not ptest.passed:
                    new_tests_failed.append(ptest)
                elif not btest.passed and ptest.passed:
                    new_tests_passed.append(ptest)

    # get all metrics failed for periodic report
    all_metrics_failed = []
    for metric in periodic_metrics:
        if metric.tests and not metric.tests[0].passed:
            all_metrics_failed.append(metric)
    # get new metrics failed for periodic report
    new_metrics_failed_periodic = []
    new_metrics_failed_baseline = []
    new_metrics_passed = []
    for pmetric, bmetric in zip(periodic_metrics, baseline_metrics):
        if pmetric.type == bmetric.type:
            if bmetric.tests[0].passed and not pmetric.tests[0].passed:  # type: ignore[index]
                new_metrics_failed_periodic.append(pmetric)
                new_metrics_failed_baseline.append(bmetric)
            elif not bmetric.tests[0].passed and pmetric.tests[0].passed:  # type: ignore[index]
                new_metrics_passed.append(pmetric)
    return {
        "report_type": report_type,
        "fail_rate": str(round(fail_rate * 100)) + "%",
        "fail_rate_change": str(round(fail_rate_change * 100, 1)) + "%",
        "time_diff_string": time_diff_string,
        "new_tests_failed": new_tests_failed,
        "new_tests_passed": new_tests_passed,
        "all_metrics_failed": all_metrics_failed,
        "new_metrics_failed_periodic": new_metrics_failed_periodic,
        "new_metrics_failed_baseline": new_metrics_failed_baseline,
        "new_metrics_passed": new_metrics_passed,
    }


def get_time_diff_string(time_diff: timedelta) -> str:
    """Get a time difference string from a timedelta object."""
    if time_diff.days == 0:
        if time_diff.total_seconds() < 60:
            result = f"past {time_diff.seconds} seconds"
        elif time_diff.total_seconds() < 3600:
            if time_diff.seconds % 60 == 1:
                result = "past minute"
            else:
                result = f"past {time_diff.total_seconds() // 60} minutes"
        elif time_diff.seconds % 3600 == 1:
            result = "past hour"
        else:
            result = f"past {time_diff.total_seconds() // 3600} hours"

    elif time_diff.days % 7 == 0:
        if time_diff.days == 7:
            result = "past week"
        else:
            result = f"past {time_diff.days // 7} weeks"
    else:
        result = f"past {time_diff.days} days"

    return result
