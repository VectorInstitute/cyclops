"""Evaluate one or more models on a dataset."""

import logging
import warnings
from dataclasses import asdict
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from datasets import Dataset, DatasetDict, config, load_dataset
from datasets.splits import Split

from cyclops.data.slicer import SliceSpec
from cyclops.data.utils import set_decode
from cyclops.evaluate.fairness.config import FairnessConfig
from cyclops.evaluate.fairness.evaluator import evaluate_fairness
from cyclops.evaluate.metrics.experimental.metric import Metric
from cyclops.evaluate.metrics.experimental.metric_dict import MetricDict
from cyclops.evaluate.metrics.experimental.utils.types import Array
from cyclops.evaluate.utils import (
    _format_column_names,
    check_required_columns,
    choose_split,
    get_columns_as_array,
)
from cyclops.utils.log import setup_logging


LOGGER = logging.getLogger(__name__)
setup_logging(print_level="WARN", logger=LOGGER)


def evaluate(
    dataset: Union[str, Dataset, DatasetDict],
    metrics: Union[Metric, Sequence[Metric], Dict[str, Metric], MetricDict],
    target_columns: Union[str, List[str]],
    prediction_columns: Union[str, List[str]],
    ignore_columns: Optional[Union[str, List[str]]] = None,
    slice_spec: Optional[SliceSpec] = None,
    split: Optional[Union[str, Split]] = None,
    batch_size: Optional[int] = config.DEFAULT_MAX_BATCH_SIZE,
    raise_on_empty_slice: bool = False,
    fairness_config: Optional[FairnessConfig] = None,
    override_fairness_metrics: bool = True,
    load_dataset_kwargs: Optional[Dict[str, Any]] = None,
    array_lib: Literal["numpy", "torch", "cupy"] = "numpy",
) -> Dict[str, Any]:
    """Evaluate one or more models on a dataset using one or more metrics.

    Parameters
    ----------
    dataset : Union[str, Dataset, DatasetDict]
        The dataset to evaluate on. If a string, the dataset will be loaded
        using `datasets.load_dataset`. If `DatasetDict`, the `split` argument
        must be specified.
    metrics : Union[Metric, Sequence[Metric], Dict[str, Metric], MetricDict]
        The metrics to compute.
    target_columns : Union[str, List[str]]
        The name of the column(s) containing the target values. A string value
        indicates a single column. A list of strings indicates a multi-label
        task - the target values will be the union of the columns.
    prediction_columns : Union[str, List[str]]
        The names of the prediction columns used to compute metrics. If a string, it
        should be the name of a column in the dataset. If a list, it should be a list
        of column names in the dataset. Lists allow for evaluating multiple models
        on the same dataset.
    ignore_columns : Union[str, List[str]], optional
        The name of the column(s) to ignore while filtering the dataset and computing
        metrics. This is useful if the dataset contains columns that are not needed
        for computing metrics but may be expensive to keep in memory
        (e.g. image columns).
    slice_spec : SliceSpec, optional
        The slice specification to use for computing metrics. If None, no slices
        will be computed - the metrics will be computed on the entire dataset.
        Note that this is not used for computing fairness metrics.
    split : Union[str, Split], optional
        The split of the dataset to use. If None and `dataset` is a string, a
        split will be chosen based on the available splits in the dataset. The
        first split that matches one of the following in order will be chosen:
        ("test", "testing", "eval", "evaluation", "validation", "val", "valid",
        "dev", "train", "training")
        If `dataset` is a `DatasetDict`, this must be specified.
    batch_size : int, optional
        The batch size to use when computing metrics. If None or a negative
        integer, the entire dataset will be loaded into memory and metrics
        will be computed in one batch.
    raise_on_empty_slice : bool, default=False
        Whether to raise an error if a slice is empty. If False, a warning will
        be logged and the metric values will be set to `NaN`.
    fairness_config : Optional[FairnessConfig], optional
        The configuration for computing fairness metrics. If None, no fairness
        metrics will be computed. Before computing fairness metrics, the following
        arguments in the configuration will be overridden by the arguments provided
        to this function: `dataset`, `target_columns`, `prediction_columns`,
        `remove_columns`, and `batch_size`. If `override_fairness_metrics` is True,
        the metrics in the configuration will be overridden by the metrics provided
        to this function.
    override_fairness_metrics : bool, optional, default=True
        If True, the `metrics` argument in fairness_config will be overridden by
        the `metrics` argument provided to this function.
    load_dataset_kwargs : Dict[str, Any], optional
        Keyword arguments to pass to `datasets.load_dataset`. Only used if
        `dataset` is a string.
    array_lib : {"numpy", "torch", "cupy"}, default="numpy"
        The array library to use for the metric computation. The metric results
        will be returned in the format of `array_lib`.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the results of the evaluation.

    Raises
    ------
    ValueError
        - If `dataset` is a `DatasetDict` and `split` is None.

    """
    dataset = _load_data(dataset, split, **(load_dataset_kwargs or {}))
    metrics = _prepare_metrics(metrics)

    check_required_columns(
        dataset.column_names,
        target_columns,
        prediction_columns,
        ignore_columns,
    )

    if slice_spec is None:
        slice_spec = SliceSpec()

    metric_results = _compute_metrics(
        dataset=dataset,
        metrics=metrics,
        slice_spec=slice_spec,
        target_columns=target_columns,
        prediction_columns=prediction_columns,
        ignore_columns=ignore_columns,
        batch_size=batch_size,
        raise_on_empty_slice=raise_on_empty_slice,
        array_lib=array_lib,
    )

    results = {}
    results.update(metric_results)

    if fairness_config is not None:
        if override_fairness_metrics:
            fairness_config.metrics = metrics

        fairness_config.dataset = dataset
        fairness_config.target_columns = target_columns
        fairness_config.prediction_columns = prediction_columns
        fairness_config.batch_size = batch_size
        fairness_config.remove_columns = ignore_columns

        fairness_results = evaluate_fairness(
            **asdict(fairness_config), array_lib=array_lib
        )
        results["fairness"] = fairness_results

    return results


def _load_data(
    dataset: Union[str, Dataset, DatasetDict],
    split: Optional[Union[str, Split]] = None,
    **load_dataset_kwargs: Any,
) -> Dataset:
    """Load data for evaluation."""
    if isinstance(dataset, str):
        if split is None:
            split = choose_split(dataset, **load_dataset_kwargs)
            LOGGER.warning(
                "Got `split=None` but `dataset` is a string. "
                "Using `split=%s` instead.",
                split,
            )

        if load_dataset_kwargs is None:
            load_dataset_kwargs = {}
        # remove `split` from `load_dataset_kwargs` if it's there
        load_dataset_kwargs.pop("split", None)

        dataset_ = load_dataset(dataset, split=split, **load_dataset_kwargs)
        assert isinstance(
            dataset_,
            Dataset,
        ), f"Expected a `Dataset` but got {type(dataset_)}."
        return dataset_
    if isinstance(dataset, DatasetDict):
        if split is None:
            split = choose_split(dataset)
            LOGGER.warning(
                "Got `split=None` but `dataset` is a DatasetDict or "
                "IterableDatasetDict. Using `split=%s` instead.",
                split,
            )

        if split == Split.ALL:
            raise ValueError(
                "Got `split=Split.ALL` but `dataset` is a DatasetDict. "
                "Please specify a split name.",
            )

        return dataset[split]
    if isinstance(dataset, Dataset):
        return dataset

    raise TypeError(
        f"Invalid type for `dataset`: {type(dataset)}. Expected one of: "
        "string, Dataset, DatasetDict.",
    )


def _prepare_metrics(
    metrics: Union[Metric, Sequence[Metric], Dict[str, Metric], MetricDict],
) -> MetricDict:
    """Prepare metrics for evaluation."""
    # TODO [fcogidi]: wrap in BootstrappedMetric if computing confidence intervals
    if isinstance(metrics, (Metric, Sequence, Dict)) and not isinstance(
        metrics,
        MetricDict,
    ):
        return MetricDict(metrics)  # type: ignore[arg-type]
    if isinstance(metrics, MetricDict):
        return metrics

    raise TypeError(
        f"Invalid type for `metrics`: {type(metrics)}. "
        "Expected one of: Metric, Sequence[Metric], Dict[str, Metric], "
        "MetricDict.",
    )


def _compute_metrics(
    dataset: Dataset,
    metrics: MetricDict,
    slice_spec: SliceSpec,
    target_columns: Union[str, List[str]],
    prediction_columns: Union[str, List[str]],
    ignore_columns: Optional[Union[str, List[str]]] = None,
    batch_size: Optional[int] = config.DEFAULT_MAX_BATCH_SIZE,
    raise_on_empty_slice: bool = False,
    array_lib: Literal["numpy", "torch", "cupy"] = "numpy",
) -> Dict[str, Dict[str, Any]]:
    """Compute metrics for a dataset."""
    target_columns = _format_column_names(target_columns)
    prediction_columns = _format_column_names(prediction_columns)

    # temporarily stop decoding features to save memory
    set_decode(dataset, False, exclude=target_columns + prediction_columns)

    with dataset.formatted_as("arrow", columns=target_columns + prediction_columns):
        results: Dict[str, Dict[str, Any]] = {}
        for slice_name, slice_fn in slice_spec.slices():
            sliced_dataset = dataset.remove_columns(ignore_columns or []).filter(
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

            for prediction_column in prediction_columns:
                if len(sliced_dataset) == 0:
                    warnings.warn(
                        "Got an empty dataset after applying the slice "
                        "%s. Metric values will be set to `None`." % slice_name,
                        RuntimeWarning,
                        stacklevel=1,
                    )
                    metric_output: Dict[str, Array] = {
                        metric_name: float("NaN")  # type: ignore
                        for metric_name in metrics  # type: ignore
                    }
                elif (
                    batch_size is None or batch_size < 0
                ):  # dataset.iter does not support getting all batches at once
                    targets = get_columns_as_array(
                        dataset=sliced_dataset,
                        columns=target_columns,
                        array_lib=array_lib,
                    )
                    predictions = get_columns_as_array(
                        dataset=sliced_dataset,
                        columns=prediction_column,
                        array_lib=array_lib,
                    )
                    metric_output = metrics(targets, predictions)
                else:
                    for batch in sliced_dataset.iter(batch_size=batch_size):
                        targets = get_columns_as_array(
                            dataset=batch, columns=target_columns, array_lib=array_lib
                        )
                        predictions = get_columns_as_array(
                            dataset=batch,
                            columns=prediction_column,
                            array_lib=array_lib,
                        )

                        # update the metric state
                        metrics.update(targets, predictions)

                    metric_output = metrics.compute()
                metrics.reset()

                model_name: str = "model_for_%s" % prediction_column
                results.setdefault(model_name, {})
                results[model_name][slice_name] = metric_output
                results[model_name][slice_name]["sample_size"] = len(sliced_dataset)

        set_decode(dataset, True)  # restore decoding features

        return results
