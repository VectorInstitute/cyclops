"""Utility functions for the `cyclops.report` module."""

import glob
import importlib
import inspect
import json
import os
from datetime import date as dt_date
from datetime import datetime as dt_datetime
from re import sub
from typing import Any, Dict, List, Mapping, Optional, Union

from cyclops.report.model_card import ModelCard  # type: ignore[attr-defined]


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
