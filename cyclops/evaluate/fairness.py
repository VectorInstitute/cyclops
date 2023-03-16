"""Fairness metrics."""
import itertools
import logging
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Union
import warnings

import numpy as np
import pandas as pd

from datasets import Dataset, config
from dataset.features import Features

from cyclops.datasets.slice import SliceSpec
from cyclops.datasets.utils import (
    check_required_columns,
    feature_is_numeric,
    feature_is_datetime,
)
from cyclops.evaluate.metrics.metric import Metric, MetricCollection, create_metric
from cyclops.evaluate.metrics.utils import _check_thresholds  # noqa: F401
from cyclops.utils.log import setup_logging

LOGGER = logging.getLogger(__name__)
setup_logging(print_level="WARN", logger=LOGGER)


def fairness_metric(  # pylint: disable=too-many-arguments
    metrics: Union[str, Callable[..., Any], Metric, MetricCollection],
    dataset: Dataset,
    groups: Union[str, List[str]],
    target_columns: Union[str, List[str]],
    prediction_columns: Union[str, List[str]] = "predictions",
    group_bins: Optional[Mapping[str, Union[int, List[Any]]]] = None,
    group_base_values: Optional[Mapping[str, Any]] = None,
    thresholds: Optional[Union[int, List[float]]] = None,
    batch_size: int = config.DEFAULT_MAX_BATCH_SIZE,
    compute_optimal_threshold: bool = False,  # expensive operation
    metric_name: Optional[str] = None,
    metric_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Dict[str, Any]]]]:
    """Compute fairness metrics.

    Parameters
    ----------
    metrics : Union[str, Callable[..., Any], Metric, MetricCollection]
        The metric or metrics to compute. If a string, it should be the name of a
        metric provided by Cyclops. If a callable, it should be a function that
        takes target, prediction, and optionally threshold/thresholds as arguments
        and returns a dictionary of metric names and values.
    dataset : Dataset
        The dataset to compute the metrics on.
    groups : Union[str, List[str]]
        The group or groups to evaluate fairness on. If a string, it should be the
        name of a column in the dataset. If a list, it should be a list of column
        names in the dataset. Lists allow for evaluating fairness at the intersection
        of multiple groups.
    target_columns : Union[str, List[str]]
        The target or targets columns used to compute metrics. If a string, it should
        be the name of a column in the dataset. If a list, it should be a list of
        column names in the dataset. Lists will be treated as multilabel targets.
    prediction_columns : Union[str, List[str]], default="predictions"
        The names of the prediction columns used to compute metrics. If a string, it
        should be the name of a column in the dataset. If a list, it should be a list
        of column names in the dataset. Lists allow for evaluating multiple models
        on the same dataset.
    group_bins : Mapping[str, Union[int, List[Any]]], optional, default=None
        Bins to use for groups with continuous values. If int, an equal number of
        bins will be created for the group. If list, the bins will be created from
        the values in the list. If None, the bins will be created from the unique
        values in the group, which may be very slow for large groups.
    group_base_values : Mapping[str, Any], optional, default=None
        The base values to use for groups. This is used in the denominator when
        computing parity across groups. If None, the base value will be the overall
        metric value.
    thresholds : Optional[Union[int, List[float]]], optional, default=None
        The thresholds to use when computing metrics. If int, thresholds will be
        created using np.linspace(0, 1, thresholds). If list, the values must be
        between 0 and 1, and monotonic. If None, the default threshold value for the
        metric will be used.
    batch_size : int, optional, default=1000
        The batch size to use when computing metrics. This is used to control memory
        usage when computing metrics on large datasets. For image datasets, this
        value should be relatively small (e.g. 32) to avoid memory issues.
    compute_optimal_threshold : bool, optional, default=False
        Whether to compute the optimal threshold for each metric. This is an
        expensive operation, and should only be used when necessary.
    metric_name : Optional[str], optional, default=None
        The name of the metric. If None, the name of the metric will be used.
    metric_kwargs : Optional[Dict[str, Any]], optional, default=None
        Keyword arguments to use when creating the metric. Only used if metrics is a
        string.

    Returns
    -------
    Union[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Dict[str, Any]]]]
        A nested dictionary containing the metric values. The first level of the
        dictionary is keyed by the prediction columns. The second level of the
        dictionary is keyed by the metric names. The third level of the dictionary
        is keyed by the slice names. If there is only one prediction column, the
        first level of the dictionary will be omitted.

    Raises
    ------
    ValueError
        If the dataset does not contain the required columns.
    TypeError
        If the batch size is not an integer.
    RuntimeError
        If an empty slice is encountered.

    """

    metrics_: Union[Callable[..., Any], MetricCollection] = _format_metrics(
        metrics, metric_name, **metric_kwargs
    )

    _format_dataset(dataset)

    groups, target_columns, prediction_columns = _format_column_names(
        groups, target_columns, prediction_columns
    )
    check_required_columns(
        dataset.column_names,
        groups,
        target_columns,
        prediction_columns,
        list(group_base_values.keys()) if group_base_values is not None else None,
        list(group_bins.keys()) if group_bins is not None else None,
    )

    unique_values = {group: dataset.unique(group) for group in groups}
    if group_bins is None:
        warn_too_many_unique_values(unqiue_values=unique_values)

    if group_base_values is not None:
        _validate_base_values(
            base_values=group_base_values, groups=groups, unique_values=unique_values
        )

    if group_bins is not None:
        _validate_group_bins(
            group_bins=group_bins, groups=groups, unique_values=unique_values
        )

        bins = _create_bins(
            group_bins=group_bins,
            dataset_features=dataset.features,
            unique_values=unique_values,
        )
        unique_values.update(bins)  # update unique values with bins

        if group_base_values is not None:  # update the base values with bins
            group_base_values = _update_base_values_with_bins(
                base_values=group_base_values, bins=bins, unique_values=unique_values
            )

    _check_thresholds(thresholds)
    if isinstance(thresholds, int):
        thresholds = np.linspace(0, 1, thresholds).tolist()

    if not isinstance(batch_size, int):
        raise TypeError(
            f"Expected `batch_size` to be of type `int`, but got {type(batch_size)}."
        )

    slice_def = _get_slice_spec(
        groups=groups,
        unique_values=unique_values,
        column_names=dataset.column_names,
    )

    if group_base_values is not None:  # since we have base values, remove overall slice
        slice_def._slice_function_registry.pop("no_filter")

    results = {}

    for slice_name, slice_fn in slice_spec.get_slices().items():
        sliced_dataset = dataset.filter(slice_fn, batched=True)

        if len(sliced_dataset) == 0:
            raise RuntimeError(
                f"Slice {slice_name} is empty. Please check your slice "
                f"configuration or the data."
            )

        for prediction_column in prediction_columns:
            results[prediction_column] = {
                "Group Size": {slice_name: sliced_dataset.num_rows}
            }

            results[prediction_column].update(
                _get_metric_results_for_prediction_and_slice(
                    metrics=metrics_,
                    dataset=sliced_dataset,
                    target_columns=target_columns,
                    prediction_column=prediction_column,
                    slice_name=slice_name,
                    batch_size=batch_size,
                    metric_name=metric_name,
                    thresholds=thresholds,
                )
            )

            if compute_optimal_threshold:
                # IDEA: generate a comprehensive list of thresholds and compute
                # the metric for each threshold. Next compute the parity metrics
                # for each threshold and select the threshold that leads to
                # the least disparity across all slices for each metric.
                raise NotImplementedError(
                    "Computing optimal threshold is not yet implemented."
                )

    # compute parity metrics
    if group_base_values is not None:
        base_slice_name = _construct_base_slice_name(base_values=group_base_values)
        parity_results = _compute_parity_metrics(
            results=results, base_slice_name=base_slice_name
        )
    else:
        parity_results = _compute_parity_metrics(
            results=results, base_slice_name="no_filter"
        )

    # add parity metrics to the results
    for prediction_column, parity_metrics in parity_results.items():
        results[prediction_column].update(parity_metrics[prediction_column])

    if len(prediction_columns) == 1:
        return results[prediction_columns[0]]

    return results


def warn_too_many_unique_values(
    unqiue_values: Union[List[Any], Mapping[str, List[Any]]],
    max_unique_values: int = 50,
) -> None:
    """Warns if the number of unique values is greater than `max_unique_values`.

    Parameters
    ----------
    unique_values : Union[List[Any], Mapping[str, List[Any]]]
        A list of unique values or a mapping from group names to lists of unique
        values.
    max_unique_values : int, default=50
        The maximum number of unique values to allow before warning.

    Raises
    ------
    TypeError
        If `unique_values` is not a list or a mapping.

    Warnings
    --------
    If the number of unique values in any group is greater than `max_unique_values`.

    """
    msg = (
        "The number of unique values for the group is greater than "
        "%s. This may take a long time to compute. "
        "Consider binning the values into fewer groups.",
        max_unique_values,
    )
    if isinstance(unqiue_values, list):
        if len(unqiue_values) > max_unique_values:
            LOGGER.warning(msg)
    elif isinstance(unqiue_values, dict) and any(
        len(values) > max_unique_values for values in unique_values.values()
    ):
        LOGGER.warning(msg)
    else:
        raise TypeError(
            f"`unique_values` must be a list or a mapping. Got {type(unqiue_values)}."
        )


def _format_metrics(
    metrics: Union[str, Callable[..., Any], Metric, MetricCollection],
    metric_name: Optional[str] = None,
    **metric_kwargs: Any,
) -> Union[Callable[..., Any], Metric, MetricCollection]:
    """Formats the metrics argument.

    Parameters
    ----------
    metrics : Union[str, Callable[..., Any], Metric, MetricCollection]
        The metrics to use for computing the metric results.
    metric_name : str, optional, default=None
        The name of the metric. This is only used if `metrics` is a callable.
    **metric_kwargs : Any
        Additional keyword arguments to pass when creating the metric. Only used
        if `metrics` is a string.

    Returns
    -------
    Union[Callable[..., Any], Metric, MetricCollection]
        The formatted metrics.

    Raises
    ------
    TypeError
        If `metrics` is not of type `str`, `Callable`, `Metric`, or `MetricCollection`.

    """
    if isinstance(metrics, str):
        metrics = create_metric(metric_name=metrics, **metric_kwargs)
    if isinstance(metrics, Metric):
        return MetricCollection(metrics)
    if isinstance(metrics, MetricCollection):
        return metrics
    if callable(metrics):
        if metric_name is None:
            LOGGER.warning(
                "No metric name was specified. The metric name will be set to "
                "the function name or 'Unnammed Metric' if the function does not "
                "have a name."
            )
        return metrics

    raise TypeError(
        f"Expected `metrics` to be of type `str`, `Metric`, `MetricCollection`, or "
        f"`Callable`, but got {type(metrics)}."
    )


def _format_dataset(dataset: Dataset, format_str: str = "numpy") -> Dataset:
    """Set the output format of the dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset to format.
    format_str : str, default="numpy"
        The format to set the dataset to. This can be any of the formats
        supported by `Dataset.set_format`.

    Raises
    ------
    TypeError
        If `dataset` is not of type `Dataset`.

    """
    if not isinstance(dataset, Dataset):
        raise TypeError(
            "Expected `dataset` to be of type `Dataset`, but got " f"{type(dataset)}."
        )

    dataset.set_format(format_str)


def _format_column_names(*column_names: Union[str, List[str]]) -> Tuple[List[str]]:
    """Formats the column names to lists of strings.

    Parameters
    ----------
    *column_names : Union[str, List[str]]
        The column names to format.

    Returns
    -------
    Tuple[List[str]]
        The formatted column names.

    Raises
    ------
    TypeError
        If any of the column names are not strings or lists of strings.

    """
    ret = []
    for column_name in column_names:
        if isinstance(column_name, str):
            ret.append([column_name])
        elif isinstance(column_name, list):
            ret.append(column_name)
        else:
            raise TypeError(
                f"Expected column name {column_name} to be a string or "
                f"list of strings, but got {type(column_name)}."
            )

    return tuple(ret)


def _validate_base_values(
    base_values: Mapping[str, Any],
    groups: List[str],
    unique_values: Mapping[str, List[Any]],
) -> None:
    """Checks that the base values are valid.

    Parameters
    ----------
    base_values : Mapping[str, Any]
        The base values for each group.
    groups : List[str]
        The groups to use for computing the metric results.
    unique_values : Mapping[str, List[Any]]
        The unique values for each group.

    Raises
    ------
    ValueError
        If the base values are not defined for all groups or if the base values
        are not part of the unique values for the group.

    """
    base_group_names = set(base_values.keys())
    group_names = set(groups)
    if not base_group_names == group_names:
        raise ValueError(
            f"The base values must be defined for all groups. Got {base_group_names} "
            f"but expected {group_names}."
        )

    # base values for each group must be part of the unique values
    for group, base_value in base_values.items():
        if base_value not in unique_values[group]:
            raise ValueError(
                f"The base value for {group} must be one of the unique "
                f"values for the group. Got {base_value} but expected one of "
                f"{unique_values[group]}."
            )


def _validate_group_bins(
    group_bins: Mapping[str, Union[int, List[Any]]],
    groups: List[str],
    unique_values: Mapping[str, List[Any]],
) -> None:
    """Checks that the group bins are valid.

    Parameters
    ----------
    group_bins : Mapping[str, Union[int, List[Any]]]
        The bins for each group.
    groups : List[str]
        The groups to use for accessing fairness.
    unique_values : Mapping[str, List[Any]]
        The unique values for each group.

    Raises
    ------
    ValueError
        If extra groups are defined in `group_bins` that are not in `groups` or
        if the number of bins is less than 2 or greater than the number of unique
        values for the group.
    TypeError
        If the bins for a group are not a list or an integer.

    """
    group_bin_names = set(group_bins.keys())
    group_names = set(groups)
    if not group_bin_names.issubset(group_names):
        raise ValueError(
            "All groups defined in `group_bins` must be in `groups`. "
            f"Found {group_bin_names - group_names} in `group_bins` but not in "
            f"`groups`."
        )

    for group, bins in group_bins.items():
        if not (isinstance(bins, list) or isinstance(bins, int)):
            raise TypeError(
                f"The bins for {group} must be a list or an integer. "
                f"Got {type(bins)}."
            )

        if isinstance(bins, int) and not 2 <= bins < len(unique_values[group]):
            raise ValueError(
                f"The number of bins must be greater than or equal to 2 "
                f"and less than the number of unique values for {group}. "
                f"Got {bins} bins and {len(unique_values[group])} unique values."
            )

        if isinstance(bins, list) and len(bins) < 2:
            raise ValueError(
                f"The number of bin values must be greater than or equal to 2. "
                f"Got {len(bins)}."
            )


def _create_bins(
    group_bins: Mapping[str, Union[int, List[Any]]],
    dataset_features: Features,
    unique_values: Mapping[str, List[Any]],
) -> Dict[str, pd.IntervalIndex]:
    """Creates the bins for numeric and datetime features.

    Parameters
    ----------
    group_bins : Mapping[str, Union[int, List[Any]]]
        The user-defined bins for each group.
    dataset_features : Features
        The features of the dataset.
    unique_values : Mapping[str, List[Any]]
        The unique values for each group.

    Returns
    -------
    Dict[str, pandas.IntervalIndex]
        The bins for each group.

    Raises
    ------
    ValueError
        If the feature for any group is not numeric or datetime.

    """
    breaks = {}
    for group, bins in group_bins.items():
        group_feature = dataset_features[group]
        if not (
            feature_is_numeric(group_feature) or feature_is_datetime(group_feature)
        ):
            raise ValueError(
                f"Column {group} in the must have a numeric or datetime dtype. "
                f"Got {group_feature.dtype}."
            )

        if isinstance(bins, list):
            # make sure it is monotonic
            if not all(bins[i] <= bins[i + 1] for i in range(len(bins) - 1)):
                bins = sorted(bins)

        out = pd.cut(unique_values[group], bins, duplicates="drop")

        intervals = out.categories

        # add -inf and inf to the left and right ends
        left_end = pd.Interval(left=-np.inf, right=intervals[0].left)
        right_end = pd.Interval(left=intervals[-1].right, right=np.inf)

        lefts = (
            [left_end.left]
            + [interval.left for interval in intervals]
            + [right_end.left]
        )
        rights = (
            [left_end.right]
            + [interval.right for interval in intervals]
            + [right_end.right]
        )

        breaks[group] = pd.IntervalIndex.from_arrays(lefts, rights)

    return breaks


def _update_base_values_with_bins(
    base_values: Dict[str, Any],
    bins: Dict[str, pd.IntervalIndex],
    unique_values: Dict[str, Any],
) -> Dict[str, Any]:
    """Updates the base values with the corresponding interval.

    Parameters
    ----------
    base_values : Dict[str, Any]
        The base values for each group.
    bins : Dict[str, pandas.IntervalIndex]
        The bins for each group.
    unique_values : Dict[str, Any]
        The unique values for each group.

    Returns
    -------
    Dict[str, Any]
        The base values with the corresponding interval for datetime and numeric
        columns.

    """
    for group, bin_values in bins.items():
        base_value = base_values[group]
        base_value_idx = np.where(unique_values[group] == base_value)[0][0]

        # use that index to get the corresponding interval
        base_values[group] = bin_values[base_value_idx]

    return base_values


def _get_slice_spec(
    groups: List[str], unique_values: Mapping[str, Any], column_names: List[str]
) -> SliceSpec:
    """Create the slice specifications for computing the metrics.

    Parameters
    ----------
    groups : List[str]
        The groups (columns) to slice on.
    unique_values : Mapping[str, Any]
        The unique values for each group.
    column_names : List[str]
        The names of the columns in the dataset.

    Returns
    -------
    SliceSpec
        The slice specifications for computing the metrics.

    """

    slices = []

    group_combinations = list(itertools.product(*unique_values.values()))

    for combination in group_combinations:
        slice_dict = {}
        for group, value in zip(groups, combination):
            if isinstance(value, pd.Interval):
                slice_dict[group] = {
                    "min_value": value.left,
                    "max_value": value.right,
                    "min_inclusive": value.closed_left,
                    "max_inclusive": value.closed_right,
                    "keep_nulls": False,
                }
            else:
                slice_dict[group] = {"value": value, "keep_nulls": False}
        slices.append(slice_dict)

    return SliceSpec(feature_values=slices, column_names=column_names)


def _compute_metrics(
    metrics: Union[Callable, MetricCollection],
    dataset: Dataset,
    target_columns: List[str],
    prediction_column: str,
    threshold: Optional[float] = None,
    batch_size: int = config.DEFAULT_MAX_BATCH_SIZE,
    metric_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute the metrics for the dataset.

    Parameters
    ----------
    metrics : Union[Callable, MetricCollection]
        The metrics to compute.
    dataset : Dataset
        The dataset to compute the metrics on.
    target_columns : Union[str, List[str]]
        The target columns.
    prediction_column : str
        The prediction column.
    threshold : Optional[float]
        The threshold to use for the metrics.
    batch_size : int
        The batch size to use for the computation.
    metric_name : Optional[str]
        The name of the metric to compute.

    Returns
    -------
    Dict[str, Any]
        The computed metrics.

    """
    if isinstance(metrics, MetricCollection):
        if threshold is not None:
            # set the threshold for each metric in the collection
            for metric in metrics._metrics:  # pylint: disable=protected-access
                if hasattr(metric, "threshold"):
                    metric.threshold = threshold
                elif hasattr(metric, "thresholds"):
                    metric.thresholds = [threshold]
                else:
                    LOGGER.warning(
                        "Metric %s does not have a threshold attribute. "
                        "Skipping setting the threshold.",
                        metric.name,
                    )

        for batch in dataset.iter(batch_size=batch_size):
            targets = np.stack(
                [batch[target_column] for target_column in target_columns],
                axis=1,
            ).squeeze()
            predictions = batch[prediction_column]

            metrics.update_state(targets, predictions)

        results: Dict[str, Any] = metrics.compute()
        metrics.reset_state()
    elif callable(metrics):
        targets = np.stack(
            [dataset[target_column] for target_column in target_columns],
            axis=1,
        ).squeeze()
        predictions = dataset[prediction_column]

        # check if the callable can take thresholds as an argument
        if threshold is not None:
            if "threshold" in inspect.signature(metrics).parameters:
                output = metrics(targets, predictions, threshold=threshold)
            elif "thresholds" in inspect.signature(metrics).parameters:
                output = metrics(targets, predictions, thresholds=[threshold])
            else:
                LOGGER.warning(
                    "The `metrics` argument is a callable that does not take a "
                    "`threshold` or `thresholds` argument. The `threshold` argument "
                    "will be ignored."
                )
                output = metrics(targets, predictions)
        else:
            output = metrics(targets, predictions)

        if metric_name is None:
            metric_name = getattr(metrics, "__name__", "Unnamed Metric")

        results = {metric_name.title(): output}
    else:
        raise TypeError(
            "The `metrics` argument must be a string, a Metric, a MetricCollection, "
            f"or a callable. Got {type(metrics)}."
        )

    return results


def _get_metric_results_for_prediction_and_slice(
    metrics: Union[Callable, MetricCollection],
    dataset: Dataset,
    target_columns: List[str],
    prediction_column: str,
    slice_name: str,
    batch_size: int,
    metric_name: Optional[str] = None,
    thresholds: Optional[List[float]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute metrics for a slice of a dataset.

    Parameters
    ----------
    metrics : Union[Callable, MetricCollection]
        The metrics to compute.
    dataset : Dataset
        The dataset to compute the metrics on.
    target_columns : Union[str, List[str]]
        The target columns.
    prediction_column : str
        The prediction column.
    slice_name : str
        The name of the slice.
    batch_size : int
        The batch size to use for the computation.
    metric_name : Optional[str]
        The name of the metric to compute.
    thresholds : Optional[List[float]]
        The thresholds to use for the metrics.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        The computed metrics.

    """

    if thresholds is None:
        metric_output = _compute_metrics(
            metrics=metrics,
            dataset=dataset,
            target_columns=target_columns,
            prediction_column=prediction_column,
            batch_size=batch_size,
            metric_name=metric_name,
        )

        # results format: {metric_name: {slice_name: metric_value}}
        return {key: {slice_name: value} for key, value in metric_output.items()}
    else:
        results = {}
        for threshold in thresholds:
            metric_output = _compute_metrics(
                metrics=metrics,
                dataset=dataset,
                target_columns=target_columns,
                prediction_column=prediction_column,
                batch_size=batch_size,
                threshold=threshold,
                metric_name=metric_name,
            )

            # results format: {metric_name@threshold: {slice_name: metric_value}}
            for key, value in metric_output.items():
                results[f"{key}@{threshold}"] = {slice_name: value}

        return results


def _construct_base_slice_name(base_values: Dict[str, Any]) -> str:
    """Construct the slice name for the base group.

    Parameters
    ----------
    base_values : Dict[str, Any]
        A dictionary mapping the group name to the base value.

    Returns
    -------
    base_slice_name : str
        A string representing the slice name for the base group.
    """
    base_slice_name = ""
    for group, base_value in base_values.items():
        if isinstance(base_value, pd.Interval):
            base_slice_name += f"{group}:{base_value.left} - {base_value.right}+"
        else:
            base_slice_name += f"{group}:{base_value}+"
    base_slice_name = base_slice_name[:-1]

    return base_slice_name


def _compute_parity_metrics(
    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    base_slice_name: str,
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Compute the parity metrics for each group and threshold if specified.

    Parameters
    ----------
    results : Dict[str, Dict[str, Dict[str, Dict[str, float]]]]
        A dictionary mapping the prediction column to the metrics dictionary.
    group_base_values : Dict[str, Any]
        A dictionary mapping the group name to the base value.
    group_names : List[str]
        A list of group names.
    thresholds : Optional[List[float]]
        A list of thresholds.

    Returns
    -------
    Dict[str, Dict[str, Dict[str, Dict[str, float]]]]
        A dictionary mapping the prediction column to the metrics dictionary.
    """

    parity_results = {}

    for key, prediction_result in results.items():
        parity_results[key] = {}
        for metric_name, slice_result in prediction_result.items():
            if metric_name == "Group Size":
                continue

            for slice_name, metric_value in slice_result.items():
                # add 'Parity' to the metric name before @threshold, if specified
                metric_name_parts = metric_name.split("@")
                parity_metric_name = f"{metric_name_parts[0]} Parity"
                if len(metric_name_parts) > 1:
                    parity_metric_name += f"@{metric_name_parts[1]}"

                numerator = metric_value
                denominator = slice_result[base_slice_name]
                parity_metric_value = denominator and numerator / denominator or 0

                # add the parity metric to the results
                parity_results[key] = {
                    parity_metric_name: {slice_name: parity_metric_value}
                }

    return parity_results
