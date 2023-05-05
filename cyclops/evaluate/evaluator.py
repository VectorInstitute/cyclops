"""Evaluate one or more models on a dataset."""
import logging
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, get_args

from datasets import Dataset, DatasetDict, config, load_dataset
from datasets.splits import Split
from sklearn.compose import ColumnTransformer

from cyclops.data.slicer import SliceSpec
from cyclops.data.utils import (
    check_required_columns,
    get_columns_as_numpy_array,
    set_decode,
)
from cyclops.evaluate.fairness.config import FairnessConfig
from cyclops.evaluate.fairness.evaluator import evaluate_fairness
from cyclops.evaluate.metrics.metric import Metric, MetricCollection
from cyclops.evaluate.utils import choose_split
from cyclops.models.wrappers import WrappedModel
from cyclops.utils.log import setup_logging

LOGGER = logging.getLogger(__name__)
setup_logging(print_level="WARN", logger=LOGGER)

# pylint: disable=fixme # TODOs


def evaluate(  # pylint: disable=too-many-function-args
    dataset: Union[str, Dataset, DatasetDict],
    metrics: Union[Metric, Sequence[Metric], Dict[str, Metric], MetricCollection],
    target_columns: Union[str, List[str]],
    feature_columns: Optional[Union[str, List[str]]] = None,
    prediction_column_prefix: str = "predictions",
    remove_columns: Optional[Union[str, List[str]]] = None,
    models: Optional[
        Union[WrappedModel, Sequence[WrappedModel], Dict[str, WrappedModel]]
    ] = None,
    transforms: Optional[Union[Callable[..., Any], ColumnTransformer]] = None,
    slice_spec: Optional[SliceSpec] = None,
    split: Optional[Union[str, Split]] = None,
    batch_size: Optional[int] = config.DEFAULT_MAX_BATCH_SIZE,
    fairness_config: Optional[FairnessConfig] = None,
    override_fairness_metrics: bool = True,
    load_dataset_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate one or more models on a dataset using one or more metrics.

    Parameters
    ----------
    dataset : Union[str, Dataset, DatasetDict]
        The dataset to evaluate on. If a string, the dataset will be loaded
        using `datasets.load_dataset`. If `DatasetDict`, the `split` argument
        must be specified.
    metrics : Union[Metric, Sequence[Metric], Dict[str, Metric], MetricCollection]
        The metrics to compute.
    target_columns : Union[str, List[str]]
        The name of the column(s) containing the target values.
    feature_columns : Union[str, List[str]], optional
        The name of the column(s) containing the feature values. This must be provided
        if `models` is not None.
    prediction_column_prefix : str, optional
        The prefix of the column(s) containing the predictions. If `models` is not
        None, the predictions will be added to the dataset and the column names will
        be `{prediction_column_prefix}.{model_name}`. If `models` is None, the
        predictions will be read from the dataset and the column names must start
        with `prediction_column_prefix`.
    remove_columns : Union[str, List[str]], optional
        The name of the column(s) to remove from the dataset before filtering
        and computing metrics. This is useful if the dataset contains columns
        that are not needed for computing metrics but may be expensive to
        keep in memory (e.g. image columns).
    models : Union[WrappedModel, Sequence[WrappedModel], Dict[str, WrappedModel]]
        The model(s) to evaluate. If a `Sequence` of `WrappedModel`, each model will
        be evaluated on the entire dataset and the model class name will be used as
        the model name. If a `Dict` of `WrappedModel`, each model will be evaluated
        on the entire dataset and the keys will be used as the model names.
    transforms : Callable, optional
        A function that transforms the dataset before doing inference. This is
        useful if the dataset needs to be transformed before being passed to
        the model.
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

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the results of the evaluation.

    Raises
    ------
    ValueError
        - If `dataset` is a `DatasetDict` and `split` is None.
        - If `models` is None and `dataset` does not have a column that starts
          with `prediction_column_prefix`.
        - If `models` is not None and `feature_columns` is None.
        - If multiple models are provided and only one set of results is found
          after computing metrics.

    """
    dataset = _load_data(dataset, split, **(load_dataset_kwargs or {}))

    column_names: List[str] = dataset.column_names
    check_required_columns(
        column_names, target_columns, feature_columns, remove_columns
    )

    metrics = _prepare_metrics(metrics)

    if models is None and not any(
        col.startswith(prediction_column_prefix) for col in column_names
    ):
        raise ValueError(
            "Got `model=None` but `dataset` does not have a column that "
            f"starts with `{prediction_column_prefix}`. Please specify a "
            f"model or add a column that starts with `{prediction_column_prefix}` "
            "to the dataset."
        )

    if models is not None:
        if feature_columns is None:
            raise ValueError(
                "Got `models` but `feature_columns` is None. Please specify "
                "`feature_columns` argument."
            )
        models = _prepare_models(models)
        for model_name, model in models.items():
            dataset = model.predict(
                dataset,
                feature_columns,
                prediction_column_prefix=prediction_column_prefix,
                model_name=model_name,
                transforms=transforms,
                only_predictions=False,
            )

    # compute metrics for each model
    results = {}

    if slice_spec is None:
        slice_spec = SliceSpec()

    metric_results = _compute_metrics(
        dataset,
        metrics,
        slice_spec,
        target_columns=target_columns,
        prediction_column_prefix=prediction_column_prefix,
        remove_columns=remove_columns,
        batch_size=batch_size,
    )
    if "default" in metric_results:
        if models is not None and len(models) > 1:
            raise ValueError(
                "Got multiple models but only one set of predictions. "
                "Please make sure that the predictions for each model "
                f"starts with `{prediction_column_prefix}` followed by "
                "the model name. For example, if the model name is "
                "`my_model`, the predictions should be in a column "
                f"called `{prediction_column_prefix}.my_model`."
            )
        if models is not None:  # only one model; replace "default" with model name
            model_name = list(models.keys())[0]
            metric_results[model_name] = metric_results.pop("default")
        else:  # no models; don't name the results
            metric_results = metric_results.pop("default")

    results.update(metric_results)

    if fairness_config is not None:
        if override_fairness_metrics:
            fairness_config.metrics = metrics

        fairness_config.dataset = dataset
        fairness_config.target_columns = target_columns
        fairness_config.prediction_columns = [
            col
            for col in dataset.column_names
            if col.startswith(prediction_column_prefix)
        ]
        fairness_config.batch_size = batch_size
        fairness_config.remove_columns = remove_columns

        fairness_results = evaluate_fairness(**asdict(fairness_config))
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
            dataset_, Dataset
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
                "Please specify a split name."
            )

        return dataset[split]
    if isinstance(dataset, Dataset):
        return dataset

    raise TypeError(
        f"Invalid type for `dataset`: {type(dataset)}. Expected one of: "
        "string, Dataset, DatasetDict."
    )


def _prepare_metrics(
    metrics: Union[Metric, Sequence[Metric], Dict[str, Metric], MetricCollection],
) -> MetricCollection:
    """Prepare metrics for evaluation."""
    # TODO: wrap in BootstrappedMetric if computing confidence intervals
    if isinstance(metrics, (Metric, Sequence, Dict)) and not isinstance(
        metrics, MetricCollection
    ):
        return MetricCollection(metrics)
    if isinstance(metrics, MetricCollection):
        return metrics

    raise TypeError(
        f"Invalid type for `metrics`: {type(metrics)}. "
        "Expected one of: Metric, Sequence[Metric], Dict[str, Metric], "
        "MetricCollection."
    )


def _prepare_models(
    model: Union[WrappedModel, Sequence[WrappedModel], Dict[str, WrappedModel]],
) -> Dict[str, WrappedModel]:
    """Prepare models for evaluation."""
    if isinstance(model, get_args(WrappedModel)):
        model_name: str = model.model_.__class__.__name__  # type: ignore
        return {model_name: model}  # type: ignore[dict-item]
    if isinstance(model, (list, tuple)):
        assert all(isinstance(m, get_args(WrappedModel)) for m in model)
        return {m.getattr("model_").__class__.__name__: m for m in model}
    if isinstance(model, dict):
        assert all(isinstance(m, get_args(WrappedModel)) for m in model.values())
        return model

    raise TypeError(
        f"Invalid type for `model`: {type(model)}. "
        "Expected one of: WrappedModel, Sequence[WrappedModel], "
        "Dict[str, WrappedModel]."
    )


def _compute_metrics(
    dataset: Dataset,
    metrics: MetricCollection,
    slice_spec: SliceSpec,
    target_columns: Union[str, List[str]],
    prediction_column_prefix: str = "predictions",
    remove_columns: Optional[Union[str, List[str]]] = None,
    batch_size: Optional[int] = config.DEFAULT_MAX_BATCH_SIZE,
) -> Dict[str, Dict[str, Any]]:
    """Compute metrics for a dataset."""
    if isinstance(target_columns, str):
        target_columns = [target_columns]

    # get the predictions (there could be multiple)
    # any column starting with `prediction_column_prefix` is considered a
    # prediction column, for a single model
    prediction_columns = [
        col for col in dataset.column_names if col.startswith(prediction_column_prefix)
    ]

    # temporarily stop decoding features to save memory
    set_decode(dataset, False, exclude=target_columns + prediction_columns)

    with dataset.formatted_as(
        "numpy", columns=target_columns + prediction_columns, output_all_columns=True
    ):
        results: Dict[str, Dict[str, Any]] = {}

        for slice_name, slice_fn in slice_spec.slices():
            sliced_dataset = dataset.remove_columns(remove_columns or []).filter(
                slice_fn,
                batched=True,
                batch_size=batch_size,
                desc=f"Filter -> {slice_name}",
            )

            if len(sliced_dataset) == 0:
                raise RuntimeError(
                    f"Slice {slice_name} is empty. Please check your slice "
                    f"configuration or the data."
                )

            for prediction_column in prediction_columns:
                if (
                    batch_size is None or batch_size < 0
                ):  # dataset.iter does not support getting all batches at once
                    targets = get_columns_as_numpy_array(
                        dataset=sliced_dataset, columns=target_columns
                    )
                    predictions = get_columns_as_numpy_array(
                        dataset=sliced_dataset, columns=prediction_column
                    )
                    metric_output = metrics(targets, predictions)
                else:
                    for batch in sliced_dataset.iter(batch_size=batch_size):
                        targets = get_columns_as_numpy_array(
                            dataset=batch, columns=target_columns
                        )
                        predictions = get_columns_as_numpy_array(
                            dataset=batch, columns=prediction_column
                        )

                        # update the metric state
                        metrics.update_state(targets, predictions)

                    metric_output = metrics.compute()
                    metrics.reset_state()

                # get the model name from the prediction column name
                # model name is everything after the first `prediction_column_prefix.`
                model_name: str = "default"
                pred_col_split = prediction_column.split(".", 1)
                if len(pred_col_split) == 2:
                    model_name = pred_col_split[1]

                results.setdefault(model_name, {})
                results[model_name][slice_name] = metric_output

        set_decode(dataset, True)  # restore decoding features

        return results
