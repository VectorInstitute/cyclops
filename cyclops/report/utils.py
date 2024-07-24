"""Utility functions for the `cyclops.report` module."""

import glob
import importlib
import inspect
import json
import os
from datetime import date as dt_date
from datetime import datetime as dt_datetime
from re import findall, sub
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go

from cyclops.report.model_card import ModelCard  # type: ignore[attr-defined]
from cyclops.report.model_card.fields import (
    Graphic,
    GraphicsCollection,
    MetricCard,
    PerformanceMetric,
    Test,
)


_METRIC_NAMES_DISPLAY_MAP = {
    "PositivePredictiveValue": "Positive Predictive Value (PPV)",
    "NegativePredictiveValue": "Negative Predictive Value (NPV)",
    "FalsePositiveRate": "False Positive Rate (FPR)",
    "FalseNegativeRate": "False Negative Rate (FNR)",
    "F1Score": "F1 Score",
}


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
    'hello__world'
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
        if isinstance(field, tuple):
            field = field[1]  # noqa: PLW2901
        if isinstance(field, Test):
            tests.append(field)
        if hasattr(field, "__fields__"):
            sweep_tests(field, tests)
        if (
            isinstance(field, list)
            and len(field) != 0
            and not isinstance(field[0], str)
            and not isinstance(field[0], int)
            and not isinstance(field[0], float)
        ):
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
        if isinstance(field, tuple):
            field = field[1]  # noqa: PLW2901
        if isinstance(field, PerformanceMetric):
            metrics.append(field)
        if hasattr(field, "__fields__"):
            sweep_metrics(field, metrics)
        if (
            isinstance(field, list)
            and len(field) != 0
            and not isinstance(field[0], str)
            and not isinstance(field[0], int)
            and not isinstance(field[0], float)
        ):
            for item in field:
                if isinstance(item, PerformanceMetric):
                    if len(field) == 1:
                        metrics.append(field[0])
                    else:
                        metrics.append(field)
                else:
                    sweep_metrics(item, metrics)


def sweep_metric_cards(model_card: Any, metric_cards: List[Any]) -> None:
    """Sweep model card to find all instances of MetricCard.

    Parameters
    ----------
    model_card : Any
        The model card to sweep.
    metric_cards : List[Any]
        The list to append all metric cards to.
    """
    for field in model_card:
        if isinstance(field, tuple):
            field = field[1]  # noqa: PLW2901
        if isinstance(field, MetricCard):
            metric_cards.append(field)
        if hasattr(field, "__fields__"):
            sweep_metric_cards(field, metric_cards)
        if (
            isinstance(field, list)
            and len(field) != 0
            and not isinstance(field[0], str)
            and not isinstance(field[0], int)
            and not isinstance(field[0], float)
        ):
            for item in field:
                if isinstance(item, MetricCard):
                    if len(field) == 1:
                        metric_cards.append(field[0])
                    else:
                        metric_cards.append(field)
                else:
                    sweep_metric_cards(item, metric_cards)


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
        if (
            isinstance(field, list)
            and len(field) != 0
            and not isinstance(field[0], str)
            and not isinstance(field[0], int)
            and not isinstance(field[0], float)
        ):
            for item in field:
                if isinstance(item, Graphic):
                    if item.name == caption:
                        graphics.append(item)
                else:
                    sweep_graphics(item, graphics, caption)


def get_slices(model_card: ModelCard) -> str:
    """Get all slices from a model card."""
    names = {}
    if (
        (model_card.overview is None)
        or (model_card.overview.metric_cards is None)
        or (model_card.overview.metric_cards.slices is None)
        or (model_card.overview.metric_cards.collection is None)
    ):
        pass
    else:
        all_slices = model_card.overview.metric_cards.slices
        for itr, metric_card in enumerate(model_card.overview.metric_cards.collection):
            name = (
                ["metric:" + metric_card.name] if metric_card.name else ["metric:none"]
            )
            card_slice = metric_card.slice
            if card_slice is not None:
                if card_slice == "overall":
                    card_slice_list = [
                        f"{slices}:overall_{slices}" for slices in all_slices
                    ]
                else:
                    card_slice_list = card_slice.split("&")
                    card_slice_list_split = [
                        card_slice.split(":")[0] for card_slice in card_slice_list
                    ]

                    for slices in all_slices:
                        card_slice_list_split = [
                            card_slice.split(":")[0] for card_slice in card_slice_list
                        ]
                        if slices not in card_slice_list_split:
                            card_slice_list.append(f"{slices}:overall_{slices}")
                name.extend(card_slice_list)
            names[itr] = name
    return json.dumps(names)


def get_thresholds(model_card: ModelCard) -> str:
    """Get all thresholds from a model card."""
    thresholds: Dict[int, Optional[str]] = {}
    if (
        (model_card.overview is None)
        or (model_card.overview.metric_cards is None)
        or (model_card.overview.metric_cards.collection is None)
    ):
        pass
    else:
        for itr, metric_card in enumerate(model_card.overview.metric_cards.collection):
            thresholds[itr] = str(metric_card.threshold)
    return json.dumps(thresholds)


def get_passed(model_card: ModelCard) -> str:
    """Get all passed from a model card."""
    passed: Dict[int, Optional[bool]] = {}
    if (
        (model_card.overview is None)
        or (model_card.overview.metric_cards is None)
        or (model_card.overview.metric_cards.collection is None)
    ):
        pass
    else:
        for itr, metric_card in enumerate(model_card.overview.metric_cards.collection):
            passed[itr] = metric_card.passed
    return json.dumps(passed)


def get_names(model_card: ModelCard) -> str:
    """Get all names from a model card."""
    names = {}
    if (
        (model_card.overview is None)
        or (model_card.overview.metric_cards is None)
        or (model_card.overview.metric_cards.collection is None)
    ):
        pass
    else:
        for itr, metric_card in enumerate(model_card.overview.metric_cards.collection):
            names[itr] = metric_card.name
    return json.dumps(names)


def get_histories(model_card: ModelCard) -> str:
    """Get all plots from a model card."""
    plots: Dict[int, Optional[List[str]]] = {}
    if (
        (model_card.overview is None)
        or (model_card.overview.metric_cards is None)
        or (model_card.overview.metric_cards.collection is None)
    ):
        pass
    else:
        for itr, metric_card in enumerate(model_card.overview.metric_cards.collection):
            plots[itr] = [str(history) for history in metric_card.history]
    return json.dumps(plots)


def get_timestamps(model_card: ModelCard) -> str:
    """Get all timestamps from a model card."""
    timestamps = {}
    if (
        (model_card.overview is None)
        or (model_card.overview.metric_cards is None)
        or (model_card.overview.metric_cards.collection is None)
    ):
        pass
    else:
        for itr, metric_card in enumerate(model_card.overview.metric_cards.collection):
            timestamps[itr] = metric_card.timestamps
    return json.dumps(timestamps)


def get_sample_sizes(model_card: ModelCard) -> str:
    """Get all sample sizes from a model card."""
    sample_sizes = {}
    if (
        (model_card.overview is None)
        or (model_card.overview.metric_cards is None)
        or (model_card.overview.metric_cards.collection is None)
    ):
        pass
    else:
        for itr, metric_card in enumerate(model_card.overview.metric_cards.collection):
            sample_sizes[itr] = (
                [str(sample_size) for sample_size in metric_card.sample_sizes]
                if metric_card.sample_sizes is not None
                else None
            )
    return json.dumps(sample_sizes)


def _extract_slices_and_values(
    current_metrics: List[PerformanceMetric],
) -> Tuple[List[str], List[List[str]]]:
    """Extract slice and value names from a list of performance metrics.

    Parameters
    ----------
    current_metrics : List[PerformanceMetric]
        The list of performance metrics to extract slice and value names from.

    Returns
    -------
    Tuple[List[str], List[List[str]]]
        A tuple of lists of slice and value names.

    """
    slices_values = []
    for current_metric in current_metrics:
        if current_metric.slice is not None:
            for slice_val in current_metric.slice.split("&"):
                if slice_val not in slices_values and slice_val != "overall":
                    slices_values.append(slice_val)
    slices = [
        slice_val.split(":")[0]
        for slice_val in slices_values
        if slice_val.split(":")[0] != "overall"
    ]
    slices = list(dict.fromkeys(slices))
    values_all = [
        slice_val.split(":")[1]
        for slice_val in slices_values
        if slice_val.split(":")[0] != "overall"
    ]
    values: List[List[str]] = [[] for _ in range(len(slices))]
    for i, slice_name in enumerate(slices):
        for j, slice_val in enumerate(slices_values):
            if slice_val.startswith(slice_name):
                values[i].append(values_all[j])

    return slices, values


def _gather_metrics(
    current_metrics: List[PerformanceMetric],
    last_metric_cards: Optional[List[MetricCard]] = None,
) -> List[Dict[str, Any]]:
    """Gather all metrics from current metrics and last metric cards.

    Parameters
    ----------
    current_metrics : List[PerformanceMetric]
        The current performance metrics.
    last_metric_cards : Optional[List[MetricCard]], optional
        The last metric cards, by default None.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries with keys type, slice, current_metric,
        and last_metric_card.

    """
    all_metrics = []
    for current_metric in current_metrics:
        last_metric_card_match = next(
            (
                last_metric_card
                for last_metric_card in last_metric_cards or []
                if current_metric.type == last_metric_card.type
                and current_metric.slice == last_metric_card.slice
            ),
            None,
        )
        all_metrics.append(
            {
                "type": current_metric.type,
                "slice": current_metric.slice,
                "current_metric": current_metric,
                "last_metric_card": last_metric_card_match,
            }
        )
    if last_metric_cards is not None:
        for last_metric_card in last_metric_cards:
            current_metric_match = next(
                (
                    current_metric
                    for current_metric in current_metrics
                    if current_metric.type == last_metric_card.type
                    and current_metric.slice == last_metric_card.slice
                ),
                None,
            )
            if current_metric_match is None:
                all_metrics.append(
                    {
                        "type": last_metric_card.type,
                        "slice": last_metric_card.slice,
                        "current_metric": None,
                        "last_metric_card": last_metric_card,
                    }
                )

    return all_metrics


def _process_metric_name(
    metric: Dict[str, Any],
) -> str:
    """Process a metric name.

    Parameters
    ----------
    metric : Dict[str, Any]
        Metric dictionary.

    Returns
    -------
    str
        The processed metric name.

    """
    if isinstance(metric["type"], str):
        # Check if name has prefix "Binary", "Multiclass", or "Multilabel"
        if metric["type"].startswith("Binary"):
            name = metric["type"][6:]
        elif metric["type"].startswith("Multiclass") or metric["type"].startswith(
            "Multilabel",
        ):
            name = metric["type"][10:]
        for key, value in _METRIC_NAMES_DISPLAY_MAP.items():
            name = name.replace(key, value)
    else:
        raise ValueError(f"Invalid metric type: {metric['type']}")

    return name


def _create_metric_card(
    metric: Dict[str, Any],
    name: str,
    history: List[float],
    timestamps: List[str],
    sample_sizes: List[int],
    threshold: Union[float, None],
    passed: Union[bool, None],
) -> MetricCard:
    """Create a metric card.

    Parameters
    ----------
    metric : Dict[str, Any]
        Metric dictionary.
    name : str
        The name for the metric card.
    history : List[float]
        The history for the metric card.
    timestamps : List[str]
        The timestamps for the metric card.
    threshold : Union[float, None]
        The threshold for the metric card.
    sample_sizes : List[int]
        The sample sizes for the metric card.
    passed : Union[bool, None]
        Whether or not the metric card passed.

    Returns
    -------
    MetricCard
        The created metric card.

    """
    return MetricCard(
        name=name,
        type=metric["current_metric"].type
        if isinstance(metric["current_metric"], PerformanceMetric)
        else None,
        slice=metric["current_metric"].slice
        if isinstance(metric["current_metric"], PerformanceMetric)
        else None,
        tooltip=metric["current_metric"].description
        if isinstance(metric["current_metric"], PerformanceMetric)
        else None,
        value=metric["current_metric"].value
        if isinstance(metric["current_metric"], PerformanceMetric)
        and isinstance(metric["current_metric"].value, float)
        else None,
        threshold=threshold,
        passed=passed,
        history=history,
        timestamps=timestamps,
        sample_sizes=sample_sizes,
    )


def _get_metric_card(
    metric: Dict[str, Any],
    name: str,
    timestamp: str,
) -> MetricCard:
    """Get a metric card.

    Parameters
    ----------
    metric : Dict[str, Any]
        Metric dictionary.
    name : str
        The name for the metric card.
    timestamp : str
        The timestamp for the current metric card.

    Returns
    -------
    Tuple[List[float], List[str], MetricCard]
        The history, timestamps, and metric card.

    """
    metric_card = None
    if (
        metric["current_metric"] is None
        and metric["last_metric_card"]
        and isinstance(
            metric["last_metric_card"],
            MetricCard,
        )
    ):
        history = metric["last_metric_card"].history
        history.append(np.nan)
        timestamps = metric["last_metric_card"].timestamps
        if timestamps is not None:
            timestamps.append(timestamp)
        sample_sizes = metric["last_metric_card"].sample_sizes
        if sample_sizes is not None:
            sample_sizes.append(0)  # Append 0 for missing data
        metric["last_metric_card"].timestamps = timestamps
        metric["last_metric_card"].sample_sizes = sample_sizes
        metric_card = metric["last_metric_card"]
    elif (
        metric["current_metric"] is not None
        and metric["last_metric_card"]
        and isinstance(
            metric["last_metric_card"],
            MetricCard,
        )
    ):
        history = metric["last_metric_card"].history
        if (isinstance(metric["current_metric"], PerformanceMetric)) and (
            isinstance(metric["current_metric"].value, float)
        ):
            history.append(metric["current_metric"].value)
        timestamps = metric["last_metric_card"].timestamps
        if timestamps is not None:
            timestamps.append(timestamp)
        sample_sizes = metric["last_metric_card"].sample_sizes
        if sample_sizes is not None:
            sample_sizes.append(metric["current_metric"].sample_size)
    else:
        history = [
            metric["current_metric"].value
            if isinstance(
                metric["current_metric"],
                PerformanceMetric,
            )
            and isinstance(metric["current_metric"].value, float)
            else 0,
        ]
        timestamps = [timestamp]
        sample_sizes = (
            [metric["current_metric"].sample_size]
            if isinstance(metric["current_metric"], PerformanceMetric)
            else [0]
        )
    if metric_card is None:
        metric_card = _create_metric_card(
            metric,
            name,
            history,
            timestamps,
            sample_sizes,
            _get_threshold(metric),
            _get_passed(metric),
        )

    return metric_card


def _get_threshold(metric: Dict[str, Any]) -> Union[float, None]:
    """Get the threshold for a metric card.

    Parameters
    ----------
    metric : Dict[str, Any]
        Metric dictionary.

    Returns
    -------
    Union[float, None]
        The threshold for the metric card.

    """
    return (
        metric["current_metric"].tests[0].threshold
        if (
            isinstance(metric["current_metric"], PerformanceMetric)
            and (metric["current_metric"].tests is not None)
            and (isinstance(metric["current_metric"].tests[0], Test))
            and (metric["current_metric"].tests[0].threshold is not None)
        )
        else None
    )


def _get_passed(metric: Dict[str, Any]) -> Union[bool, None]:
    """Get whether or not a metric card test passed.

    Parameters
    ----------
    metric : Dict[str, Any]
        Metric dictionary.

    Returns
    -------
    Union[bool, None]
        Whether or not the metric card passed.

    """
    return (
        metric["current_metric"].tests[0].passed
        if (
            isinstance(metric["current_metric"], PerformanceMetric)
            and (metric["current_metric"].tests is not None)
            and (isinstance(metric["current_metric"].tests[0], Test))
            and (metric["current_metric"].tests[0].passed is not None)
        )
        else None
    )


def create_metric_cards(
    current_metrics: List[PerformanceMetric],
    timestamp: str,
    last_metric_cards: Optional[List[MetricCard]] = None,
) -> Tuple[
    List[str],
    List[Optional[str]],
    List[str],
    List[List[str]],
    List[MetricCard],
]:
    """Create metric cards for each metric.

    Parameters
    ----------
    current_metrics : List[PerformanceMetric]
        The current performance metrics.
    timestamp : str
        The timestamp for the current metric card.
    last_metric_cards : Optional[List[MetricCard]], optional
        The last metric cards, by default None.

    Returns
    -------
    Tuple[
        List[str],
        List[Optional[str]],
        List[str],
        List[List[str]],
        List[MetricCard]
    ]
        A tuple of lists of metrics, tooltips, slices, values, and metric cards.

    """
    slices, values = _extract_slices_and_values(current_metrics)
    all_metrics = _gather_metrics(current_metrics, last_metric_cards)
    # Create dict to populate metrics cards
    metric_cards = []
    metrics = []
    tooltips = []
    for metric in all_metrics:
        name = _process_metric_name(metric)
        metrics.append(name)
        if isinstance(metric["current_metric"], PerformanceMetric):
            tooltips.append(metric["current_metric"].description)
        metric_card = _get_metric_card(metric, name, timestamp)
        metric_cards.append(metric_card)
    metrics = list(dict.fromkeys(metrics))
    tooltips = list(dict.fromkeys(tooltips))

    return metrics, tooltips, slices, values, metric_cards


def create_metric_card_plot(
    history: List[float],
    threshold: float,
) -> GraphicsCollection:
    """Create a plot for a metric card."""
    fig = go.Figure(
        data=[
            go.Scatter(
                y=history,
                mode="lines+markers",
                marker={"color": "rgb(31,111,235)"},
                line={"color": "rgb(31,111,235)"},
                showlegend=False,
            ),
            # dotted black threshold line
            go.Scatter(
                y=[threshold for _ in range(len(history))],
                mode="lines",
                line={"color": "black", "dash": "dot"},
                showlegend=False,
            ),
        ],
        layout=go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            # hide x-axis
            xaxis={
                "zeroline": False,
                "showticklabels": False,
                "showgrid": False,
            },
            yaxis={
                "gridcolor": "#ffffff",  # add this line to change the grid color
            },
        ),
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=125,
        width=250,
        # ),
    )
    data = {
        "name": None,
        "image": fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={"displayModeBar": False},
        ),
    }
    graphic = Graphic.parse_obj(data)
    return GraphicsCollection(description="plot", collection=[graphic])


def regex_replace(string: str, find: str, replace: str) -> str:
    """Replace a regex pattern with a string."""
    return sub(find, replace, string)


def regex_search(string: str, find: str) -> List[Any]:
    """Search a regex pattern in a string and return the match."""
    return findall(r"\((.*?)\)", string)


def empty(x: Optional[List[Any]]) -> bool:
    """Check if a variable is empty."""
    empty = True
    if x is not None:
        for _, obj in x:
            if isinstance(obj, list):
                if len(obj) > 0:
                    empty = False
            elif isinstance(obj, GraphicsCollection):
                if len(obj.collection) > 0:  # type: ignore[arg-type]
                    empty = False
            elif obj is not None:
                empty = False
    return empty
