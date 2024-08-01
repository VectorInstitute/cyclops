"""Fairness evaluator."""

import inspect
import itertools
import logging
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from datasets import Dataset, config
from datasets.features import Features

from cyclops.data.slicer import SliceSpec, is_datetime
from cyclops.data.utils import (
    feature_is_datetime,
    feature_is_numeric,
    set_decode,
)
from cyclops.evaluate.metrics.experimental.functional.precision_recall_curve import (
    _format_thresholds,
    _validate_thresholds,
)
from cyclops.evaluate.metrics.experimental.metric import Metric, OperatorMetric
from cyclops.evaluate.metrics.experimental.metric_dict import MetricDict
from cyclops.evaluate.metrics.experimental.utils.types import Array
from cyclops.evaluate.metrics.factory import create_metric
from cyclops.evaluate.utils import (
    _SUPPORTED_ARRAY_LIBS,
    _format_column_names,
    check_required_columns,
    get_columns_as_array,
)
from cyclops.utils.log import setup_logging


LOGGER = logging.getLogger(__name__)
setup_logging(print_level="WARN", logger=LOGGER)


def evaluate_fairness(  # noqa: PLR0912
    metrics: Union[str, Callable[..., Any], Metric, MetricDict],
    dataset: Dataset,
    groups: Union[str, List[str]],
    target_columns: Union[str, List[str]],
    prediction_columns: Union[str, List[str]],
    group_values: Optional[Dict[str, Any]] = None,
    group_bins: Optional[Dict[str, Union[int, List[Any]]]] = None,
    group_base_values: Optional[Dict[str, Any]] = None,
    thresholds: Optional[Union[int, List[float]]] = None,
    compute_optimal_threshold: bool = False,  # expensive operation
    remove_columns: Optional[Union[str, List[str]]] = None,
    batch_size: Optional[int] = config.DEFAULT_MAX_BATCH_SIZE,
    raise_on_empty_slice: bool = False,
    metric_name: Optional[str] = None,
    metric_kwargs: Optional[Dict[str, Any]] = None,
    array_lib: Literal["torch", "numpy", "cupy"] = "numpy",
) -> Union[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Dict[str, Any]]]]:
    """Compute fairness indicators.

    This function computes fairness indicators for a dataset that includes
    predictions and targets.

    Parameters
    ----------
    metrics : Union[str, Callable[..., Any], Metric, MetricDict]
        The metric or metrics to compute. If a string, it should be the name of a
        metric provided by CyclOps. If a callable, it should be a function that
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
    prediction_columns : Union[str, List[str]]
        The names of the prediction columns used to compute metrics. If a string, it
        should be the name of a column in the dataset. If a list, it should be a list
        of column names in the dataset. Lists allow for evaluating multiple models
        on the same dataset.
    group_values : Dict[str, Any], optional, default=None
        The values to use for groups. If None, the values will be the unique values
        in the group. This can be used to limit the number of groups that are
        evaluated.
    group_bins : Dict[str, Union[int, List[Any]]], optional, default=None
        Bins to use for groups with continuous values. If int, an equal number of
        bins will be created for the group. If list, the bins will be created from
        the values in the list. If None, the bins will be created from the unique
        values in the group, which may be very slow for large groups.
    group_base_values : Dict[str, Any], optional, default=None
        The base values to use for groups. This is used in the denominator when
        computing parity across groups. If None, the base value will be the overall
        metric value.
    thresholds : Optional[Union[int, List[float]]], optional, default=None
        The thresholds to use when computing metrics. If int, thresholds will be
        created using np.linspace(0, 1, thresholds). If list, the values must be
        between 0 and 1, and monotonic. If None, the default threshold value for the
        metric will be used.
    compute_optimal_threshold : bool, optional, default=False
        Whether to compute the optimal threshold for each metric. This is an
        expensive operation, and should only be used when necessary.
    remove_columns : Union[str, List[str]], optional, default=None
        The name of the column(s) to remove from the dataset before filtering
        and computing metrics. This is useful if the dataset contains columns
        that are not needed for computing metrics but may be expensive to
        keep in memory (e.g. image columns).
    batch_size : int, optional, default=1000
        The batch size to use when computing metrics. This is used to control memory
        usage when computing metrics on large datasets. For image datasets, this
        value should be relatively small (e.g. 32) to avoid memory issues.
    raise_on_empty_slice : bool, optional, default=False
        Whether to raise an error if an empty slice is encountered when computing
        metrics. If False, the metric values for the slice will be set to `NaN`.
    metric_name : Optional[str], optional, default=None
        The name of the metric. If None, the name of the metric will be used.
    metric_kwargs : Optional[Dict[str, Any]], optional, default=None
        Keyword arguments to use when creating the metric. Only used if metrics is a
        string.
    array_lib : {"numpy", "torch", "cupy"}, default="numpy"
        The array library to use for the metric computation. The metric results
        will be returned in the format of `array_lib`.

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
        If the dataset is not a HuggingFace Dataset object or if the batch size is
        not an integer.
    RuntimeError
        If an empty slice is encountered when computing metrics.

    """
    # input validation and formatting
    if not isinstance(dataset, Dataset):
        raise TypeError(
            "Expected `dataset` to be of type `Dataset`, but got " f"{type(dataset)}.",
        )
    if array_lib not in _SUPPORTED_ARRAY_LIBS:
        raise NotImplementedError(f"The array library `{array_lib}` is not supported.")
    _validate_thresholds(thresholds)

    metrics_: Union[Callable[..., Any], MetricDict] = _format_metrics(
        metrics,
        metric_name,
        **(metric_kwargs or {}),
    )
    fmt_thresholds = _format_thresholds(thresholds, xp=_SUPPORTED_ARRAY_LIBS[array_lib])
    fmt_groups: List[str] = _format_column_names(groups)
    fmt_target_columns: List[str] = _format_column_names(target_columns)
    fmt_prediction_columns: List[str] = _format_column_names(prediction_columns)

    check_required_columns(
        dataset.column_names,
        fmt_groups,
        fmt_target_columns,
        fmt_prediction_columns,
        list(group_base_values.keys()) if group_base_values is not None else None,
        list(group_bins.keys()) if group_bins is not None else None,
        list(group_values.keys()) if group_values is not None else None,
    )

    set_decode(
        dataset,
        decode=False,
        exclude=fmt_target_columns + fmt_prediction_columns,
    )  # don't decode columns that we don't need; pass dataset by reference

    with dataset.formatted_as(
        "arrow", columns=fmt_groups + fmt_target_columns + fmt_prediction_columns
    ):
        unique_values: Dict[str, List[Any]] = _get_unique_values(
            dataset=dataset,
            groups=fmt_groups,
            group_values=group_values,
        )

        if group_base_values is not None:
            _validate_base_values(
                base_values=group_base_values,
                groups=fmt_groups,
                unique_values=unique_values,
            )
            # reorder keys to match order in `groups`
            group_base_values = {
                group: group_base_values[group]
                for group in fmt_groups
                if group in group_base_values
            }

        if group_bins is None:
            warn_too_many_unique_values(unique_values=unique_values)
        else:
            _validate_group_bins(
                group_bins=group_bins,
                groups=fmt_groups,
                unique_values=unique_values,
            )

            group_bins = {
                group: group_bins[group] for group in fmt_groups if group in group_bins
            }  # reorder keys to match order given in `groups`

            bins = _create_bins(
                group_bins=group_bins,
                dataset_features=dataset.features,
                unique_values=unique_values,
            )

            if group_base_values is not None:  # update the base values with bins
                group_base_values = _update_base_values_with_bins(
                    base_values=group_base_values,
                    bins=bins,
                )

            unique_values.update(bins)  # update unique values with bins

        slice_spec = _get_slice_spec(
            groups=fmt_groups,
            unique_values=unique_values,
            column_names=dataset.column_names,
        )

        if group_base_values is not None:
            # since we have base values, remove overall slice
            slice_spec._registry.pop("overall", None)

        results: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for slice_name, slice_fn in slice_spec.slices():
            sliced_dataset = dataset.remove_columns(remove_columns or []).filter(
                slice_fn,
                batched=True,
                batch_size=batch_size,
                desc=f"Filter -> {slice_name}",
            )

            if len(sliced_dataset) == 0 and raise_on_empty_slice:
                raise RuntimeError(
                    f"Slice {slice_name} is empty. Please check your slice "
                    f"configuration or the data.",
                )

            for prediction_column in fmt_prediction_columns:
                results.setdefault(prediction_column, {})
                results[prediction_column].setdefault(slice_name, {}).update(
                    {"sample_size": len(sliced_dataset)},
                )

                pred_result = _get_metric_results_for_prediction_and_slice(
                    metrics=metrics_,
                    dataset=sliced_dataset,
                    target_columns=fmt_target_columns,
                    prediction_column=prediction_column,
                    slice_name=slice_name,
                    batch_size=batch_size,
                    metric_name=metric_name,
                    thresholds=fmt_thresholds,
                    array_lib=array_lib,
                )
                # if metric_name does not exist, add it to the dictionary
                # otherwise, update the dictionary for the metric_name
                for key, slice_result in pred_result.items():
                    results[prediction_column].setdefault(key, {}).update(slice_result)

                if compute_optimal_threshold:
                    # TODO: generate a comprehensive list of thresholds and compute
                    # the metric for each threshold. Next compute the parity metrics
                    # for each threshold and select the threshold that leads to
                    # the least disparity across all slices for each metric.
                    raise NotImplementedError(
                        "Computing optimal threshold is not yet implemented.",
                    )

    set_decode(dataset, decode=True)  # reset decode

    # compute parity metrics
    if group_base_values is not None:
        base_slice_name = _construct_base_slice_name(base_values=group_base_values)
        parity_results = _compute_parity_metrics(
            results=results, base_slice_name=base_slice_name, array_lib=array_lib
        )
    else:
        parity_results = _compute_parity_metrics(
            results=results, base_slice_name="overall", array_lib=array_lib
        )

    # add parity metrics to the results
    for pred_column, pred_results in parity_results.items():
        for slice_name, slice_results in pred_results.items():
            results[pred_column][slice_name].update(slice_results)

    if len(fmt_prediction_columns) == 1:
        return results[fmt_prediction_columns[0]]

    return results


def warn_too_many_unique_values(
    unique_values: Union[List[Any], Dict[str, List[Any]]],
    max_unique_values: int = 50,
) -> None:
    """Warns if the number of unique values is greater than `max_unique_values`.

    Parameters
    ----------
    unique_values : Union[List[Any], Dict[str, List[Any]]]
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
    if not (isinstance(max_unique_values, int) and max_unique_values > 0):
        raise TypeError(
            "`max_unique_values` must be a positive integer. Got "
            f"{type(max_unique_values)}.",
        )

    msg = (
        "The number of unique values for the group is greater than "
        "%s. This may take a long time to compute. "
        "Consider binning the values into fewer groups."
    )
    if isinstance(unique_values, list):
        if len(unique_values) > max_unique_values:
            LOGGER.warning(msg, max_unique_values)
        return
    if isinstance(unique_values, dict):
        if any(len(values) > max_unique_values for values in unique_values.values()):
            LOGGER.warning(msg, max_unique_values)
        return
    raise TypeError(
        f"`unique_values` must be a list or a mapping. Got {type(unique_values)}.",
    )


def _format_metrics(
    metrics: Union[str, Callable[..., Any], Metric, MetricDict],
    metric_name: Optional[str] = None,
    **metric_kwargs: Any,
) -> Union[Callable[..., Any], Metric, MetricDict]:
    """Format the metrics argument.

    Parameters
    ----------
    metrics : Union[str, Callable[..., Any], Metric, MetricDict]
        The metrics to use for computing the metric results.
    metric_name : str, optional, default=None
        The name of the metric. This is only used if `metrics` is a callable.
    **metric_kwargs : Any
        Additional keyword arguments to pass when creating the metric. Only used
        if `metrics` is a string.

    Returns
    -------
    Union[Callable[..., Any], Metric, MetricDict]
        The formatted metrics.

    Raises
    ------
    TypeError
        If `metrics` is not of type `str`, `Callable`, `Metric`, or `MetricDict`.

    """
    if isinstance(metrics, str):
        metrics = create_metric(metric_name=metrics, experimental=True, **metric_kwargs)
    if isinstance(metrics, Metric):
        if metric_name is not None and isinstance(metrics, OperatorMetric):
            # single metric created from arithmetic operation, with given name
            return MetricDict({metric_name: metrics})
        return MetricDict(metrics)
    if isinstance(metrics, MetricDict):
        return metrics
    if callable(metrics):
        if metric_name is None:
            LOGGER.warning(
                "No metric name was specified. The metric name will be set to "
                "the function name or 'Unnammed Metric' if the function does not "
                "have a name.",
            )
        return metrics

    raise TypeError(
        f"Expected `metrics` to be of type `str`, `Metric`, `MetricDict`, or "
        f"`Callable`, but got {type(metrics)}.",
    )


def _get_unique_values(
    dataset: Dataset,
    groups: List[str],
    group_values: Optional[Dict[str, Any]],
) -> Dict[str, List[Any]]:
    """Get the unique values for a group."""
    unique_values = {}
    for group in groups:
        column_unique_values = dataset.unique(group)
        if group_values is not None and group in group_values:
            udv = group_values[group]  # user defined values
            if not isinstance(udv, list):
                udv = [udv]

            # check that the user defined values are in the unique values
            if not set(udv).issubset(set(column_unique_values)):
                raise ValueError(
                    f"User defined values {udv} for group {group} are not a subset of "
                    f"the unique values {column_unique_values}.",
                )
            unique_values[group] = udv
        else:
            unique_values[group] = column_unique_values
    return unique_values


def _validate_base_values(
    base_values: Dict[str, Any],
    groups: List[str],
    unique_values: Dict[str, List[Any]],
) -> None:
    """Check that the base values are valid.

    Parameters
    ----------
    base_values : Dict[str, Any]
        The base values for each group.
    groups : List[str]
        The groups to use for computing the metric results.
    unique_values : Dict[str, List[Any]]
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
            f"but expected {group_names}.",
        )

    # base values for each group must be part of the unique values
    # unless it a numeric or datetime type, then it can be any value
    # in the range of the unique values
    for group, base_value in base_values.items():
        if isinstance(base_value, (int, float, datetime)) or is_datetime(base_value):
            continue
        if base_value not in unique_values[group]:
            raise ValueError(
                f"The base value {base_value} for group {group} is not part of the "
                f"unique values for the group. Got {unique_values[group]}.",
            )


def _validate_group_bins(
    group_bins: Dict[str, Union[int, List[Any]]],
    groups: List[str],
    unique_values: Dict[str, List[Any]],
) -> None:
    """Check that the group bins are valid.

    Parameters
    ----------
    group_bins : Dict[str, Union[int, List[Any]]]
        The bins for each group.
    groups : List[str]
        The groups to use for accessing fairness.
    unique_values : Dict[str, List[Any]]
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
            f"`groups`.",
        )

    for group, bins in group_bins.items():
        if not isinstance(bins, (list, int)):
            raise TypeError(
                f"The bins for {group} must be a list or an integer. "
                f"Got {type(bins)}.",
            )

        if isinstance(bins, int) and not 2 <= bins < len(unique_values[group]):
            raise ValueError(
                f"The number of bins must be greater than or equal to 2 "
                f"and less than the number of unique values for {group}. "
                f"Got {bins} bins and {len(unique_values[group])} unique values.",
            )

        if isinstance(bins, list) and len(bins) < 2:
            raise ValueError(
                f"The number of bin values must be greater than or equal to 2. "
                f"Got {len(bins)}.",
            )


def _create_bins(
    group_bins: Dict[str, Union[int, List[Any]]],
    dataset_features: Features,
    unique_values: Dict[str, List[Any]],
) -> Dict[str, pd.IntervalIndex]:
    """Create the bins for numeric and datetime features.

    Parameters
    ----------
    group_bins : Dict[str, Union[int, List[Any]]]
        The user-defined bins for each group.
    dataset_features : Features
        The features of the dataset.
    unique_values : Dict[str, List[Any]]
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
        column_is_datetime = feature_is_datetime(group_feature)
        if not (feature_is_numeric(group_feature) or column_is_datetime):
            raise ValueError(
                f"Column {group} in the must have a numeric or datetime dtype. "
                f"Got {group_feature.dtype}.",
            )

        if isinstance(bins, list):
            # make sure it is monotonic
            if not all(bins[i] <= bins[i + 1] for i in range(len(bins) - 1)):
                bins = sorted(bins)  # noqa: PLW2901

            # convert timestring values to datetime
            if column_is_datetime:
                bins = pd.to_datetime(bins).values  # noqa: PLW2901

        cut_data = pd.Series(
            unique_values[group],
            dtype="datetime64[ns]" if column_is_datetime else None,
        ).to_numpy()
        out = pd.cut(cut_data, bins, duplicates="drop")

        intervals = out.categories

        # add -inf and inf to the left and right ends
        left_end = pd.Interval(
            left=pd.Timestamp.min if column_is_datetime else -np.inf,
            right=intervals[0].left,
        )
        right_end = pd.Interval(
            left=intervals[-1].right,
            right=pd.Timestamp.max if column_is_datetime else np.inf,
        )

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
) -> Dict[str, Any]:
    """Update the base values with the corresponding interval.

    Parameters
    ----------
    base_values : Dict[str, Any]
        The base values for each group.
    bins : Dict[str, pandas.IntervalIndex]
        The bins for each group.

    Returns
    -------
    Dict[str, Any]
        The base values with the corresponding interval for datetime and numeric
        columns.

    """
    for group, bin_values in bins.items():
        base_value = base_values[group]

        # find the interval that contains the base value
        for interval in bin_values:
            if isinstance(interval.left, pd.Timestamp):
                base_value = pd.to_datetime(base_value)
            if interval.left <= base_value <= interval.right:
                base_values[group] = interval
                break

    return base_values


def _get_slice_spec(
    groups: List[str],
    unique_values: Dict[str, List[Any]],
    column_names: List[str],
) -> SliceSpec:
    """Create the slice specifications for computing the metrics.

    Parameters
    ----------
    groups : List[str]
        The groups (columns) to slice on.
    unique_values : Dict[str, List[Any]]
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
                    "min_value": -np.inf
                    if value.left == pd.Timestamp.min
                    else value.left,
                    "max_value": np.inf
                    if value.right == pd.Timestamp.max
                    else value.right,
                    "min_inclusive": value.closed_left,
                    "max_inclusive": value.closed_right,
                    "keep_nulls": False,
                }
            else:
                slice_dict[group] = {"value": value, "keep_nulls": False}
        slices.append(slice_dict)

    return SliceSpec(slices, validate=True, column_names=column_names)


def _compute_metrics(  # noqa: C901, PLR0912
    metrics: Union[Callable[..., Any], MetricDict],
    dataset: Dataset,
    target_columns: List[str],
    prediction_column: str,
    threshold: Optional[float] = None,
    batch_size: Optional[int] = config.DEFAULT_MAX_BATCH_SIZE,
    metric_name: Optional[str] = None,
    array_lib: Literal["torch", "numpy", "cupy"] = "numpy",
) -> Dict[str, Any]:
    """Compute the metrics for the dataset.

    Parameters
    ----------
    metrics : Union[Callable, MetricDict]
        The metrics to compute.
    dataset : Dataset
        The dataset to compute the metrics on.
    target_columns : Union[str, List[str]]
        The target columns.
    prediction_column : str
        The prediction column.
    threshold : Union[float, List[float]], optional, default=None
        The threshold to use for the metrics.
    batch_size : int
        The batch size to use for the computation.
    metric_name : Optional[str]
        The name of the metric to compute.
    array_lib : {"torch", "numpy, "cupy"}, default="numpy"
        The array library to use for the metric computation. The metric results
        will be returned in the format of `array_lib`.

    Returns
    -------
    Dict[str, Any]
        The computed metrics.

    """
    empty_dataset_msg = (
        "Encountered empty dataset while computing metrics. "
        "The metric values will be set to `None`."
    )
    if isinstance(metrics, MetricDict):
        if threshold is not None:
            # set the threshold for each metric in the collection
            for name, metric in metrics.items():
                if isinstance(metric, Metric) and hasattr(metric, "threshold"):
                    metric.threshold = threshold
                elif isinstance(metric, OperatorMetric):
                    if hasattr(metric.metric_a, "threshold") and hasattr(
                        metric.metric_b,
                        "threshold",
                    ):
                        metric.metric_a.threshold = threshold
                        metric.metric_b.threshold = threshold  # type: ignore[union-attr]
                else:
                    LOGGER.warning(
                        "Metric %s does not have a threshold attribute. "
                        "Skipping setting the threshold.",
                        name,
                    )

        if len(dataset) == 0:
            warnings.warn(empty_dataset_msg, RuntimeWarning, stacklevel=1)
            results: Dict[str, Any] = {
                metric_name: float("NaN")
                for metric_name in metrics  # type: ignore[attr-defined]
            }
        elif (
            batch_size is None or batch_size <= 0
        ):  # dataset.iter does not support getting all rows
            targets = get_columns_as_array(
                dataset=dataset, columns=target_columns, array_lib=array_lib
            )
            predictions = get_columns_as_array(
                dataset=dataset, columns=prediction_column, array_lib=array_lib
            )
            results = metrics(targets, predictions)
        else:
            for batch in dataset.iter(batch_size=batch_size):
                targets = get_columns_as_array(
                    dataset=batch, columns=target_columns, array_lib=array_lib
                )
                predictions = get_columns_as_array(
                    dataset=batch, columns=prediction_column, array_lib=array_lib
                )

                metrics.update(targets, predictions)

            results = metrics.compute()

        metrics.reset()

        return results
    if callable(metrics):
        if metric_name is None:
            metric_name = getattr(metrics, "__name__", "Unnamed Metric")

        if len(dataset) == 0:
            warnings.warn(empty_dataset_msg, RuntimeWarning, stacklevel=1)
            return {metric_name.title(): float("NaN")}

        targets = get_columns_as_array(
            dataset=dataset, columns=target_columns, array_lib=array_lib
        )
        predictions = get_columns_as_array(
            dataset=dataset, columns=prediction_column, array_lib=array_lib
        )

        # check if the callable can take thresholds as an argument
        if threshold is not None:
            if "threshold" in inspect.signature(metrics).parameters:
                output = metrics(targets, predictions, threshold=threshold)
            else:
                LOGGER.warning(
                    "The `metrics` argument is a callable that does not take a "
                    "`threshold` or `thresholds` argument. The `threshold` argument "
                    "will be ignored.",
                )
                output = metrics(targets, predictions)
        else:
            output = metrics(targets, predictions)

        return {metric_name.title(): output}

    raise TypeError(
        "The `metrics` argument must be a string, a Metric, a MetricDict, "
        f"or a callable. Got {type(metrics)}.",
    )


def _get_metric_results_for_prediction_and_slice(
    metrics: Union[Callable[..., Any], MetricDict],
    dataset: Dataset,
    target_columns: List[str],
    prediction_column: str,
    slice_name: str,
    batch_size: Optional[int] = config.DEFAULT_MAX_BATCH_SIZE,
    metric_name: Optional[str] = None,
    thresholds: Optional[Array] = None,
    array_lib: Literal["torch", "numpy", "cupy"] = "numpy",
) -> Dict[str, Dict[str, Any]]:
    """Compute metrics for a slice of a dataset.

    Parameters
    ----------
    metrics : Union[Callable, MetricDict]
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
    thresholds : Optional[Array]
        The thresholds to use for the metrics.
    array_lib : {"numpy", "torch", "cupy"}, default="numpy"
        The array library to use for the metric computation. The metric results
        will be returned in the format of `array_lib`.

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
            array_lib=array_lib,
        )

        # result format -> {slice_name: {metric_name: metric_value}}
        return {slice_name: metric_output}

    results: Dict[str, Dict[str, Any]] = {}
    for threshold in thresholds:  # type: ignore[attr-defined]
        metric_output = _compute_metrics(
            metrics=metrics,
            dataset=dataset,
            target_columns=target_columns,
            prediction_column=prediction_column,
            batch_size=batch_size,
            threshold=threshold,
            metric_name=metric_name,
            array_lib=array_lib,
        )

        # result format -> {slice_name: {metric_name@threshold: metric_value}}
        for key, value in metric_output.items():
            results.setdefault(slice_name, {}).update({f"{key}@{threshold}": value})
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
            min_value = (
                -np.inf if base_value.left == pd.Timestamp.min else base_value.left
            )
            max_value = (
                np.inf if base_value.right == pd.Timestamp.max else base_value.right
            )
            min_end = "[" if base_value.closed_left else "("
            max_end = "]" if base_value.closed_right else ")"
            base_slice_name += f"{group}:{min_end}{min_value} - {max_value}{max_end}&"
        else:
            base_slice_name += f"{group}:{base_value}&"
    return base_slice_name[:-1]


def _compute_parity_metrics(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    base_slice_name: str,
    array_lib: Literal["numpy", "torch", "cupy"],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Compute the parity metrics for each group and threshold if specified.

    Parameters
    ----------
    results : Dict[str, Dict[str, Dict[str, Any]]]
        A dictionary mapping the prediction column to the metrics dictionary.
    base_slice_name : str
        The name of the base slice.
    array_lib : {"numpy", "torch", "cupy"}, default="numpy"
        The array library to use for the metric computation. The metric results
        will be returned in the format of `array_lib`.

    Returns
    -------
    Dict[str, Dict[str, Dict[str, Any]]]
        A dictionary mapping the prediction column to the metrics dictionary.

    """
    xp = _SUPPORTED_ARRAY_LIBS[array_lib]
    parity_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for key, prediction_result in results.items():
        parity_results[key] = {}
        for slice_name, slice_result in prediction_result.items():
            for metric_name, metric_value in slice_result.items():
                if metric_name == "sample_size":
                    continue

                # add 'Parity' to the metric name before @threshold, if specified
                metric_name_parts = metric_name.split("@")
                parity_metric_name = f"{metric_name_parts[0]} Parity"
                if len(metric_name_parts) > 1:
                    parity_metric_name += f"@{metric_name_parts[1]}"

                numerator = metric_value
                denominator = prediction_result[base_slice_name][metric_name]
                # value is NaN if either the numerator or denominator is NaN
                if xp.all(xp.isnan(numerator)) or xp.all(xp.isnan(denominator)):
                    parity_metric_value = xp.nan
                else:
                    parity_metric_value = xp.divide(numerator, denominator)

                parity_results[key].setdefault(slice_name, {}).update(
                    {parity_metric_name: parity_metric_value},
                )

    return parity_results
