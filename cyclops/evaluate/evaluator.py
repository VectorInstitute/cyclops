"""Evaluator class."""
import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union, get_args

import numpy as np
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    config,
    load_dataset,
)
from datasets.splits import Split

from cyclops.datasets.slice import SliceSpec
from cyclops.datasets.utils import check_required_columns
from cyclops.evaluate.metrics.metric import Metric, MetricCollection
from cyclops.evaluate.utils import choose_split
from cyclops.models.wrappers import WrappedModel
from cyclops.utils.log import setup_logging

LOGGER = logging.getLogger(__name__)
setup_logging(print_level="WARN", logger=LOGGER)


class Evaluator:
    """Evaluator class."""

    def load_data(
        self,
        dataset: Union[str, Dataset, DatasetDict, IterableDataset, IterableDatasetDict],
        split: Optional[Union[str, Split]] = None,
        **load_dataset_kwargs: Mapping[str, Any],
    ) -> Union[Dataset, IterableDataset]:
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

            return load_dataset(dataset, split=split, **load_dataset_kwargs)
        if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
            if split is None:
                split = choose_split(dataset)
                LOGGER.warning(
                    "Got `split=None` but `dataset` is a DatasetDict or "
                    "IterableDatasetDict. Using `split=%s` instead.",
                    split,
                )

            if split == Split.ALL:
                raise ValueError(
                    "Got `split=Split.ALL` but `dataset` is a DatasetDict or "
                    "IterableDatasetDict. Please specify a split name."
                )

            return dataset[split]
        if isinstance(dataset, (Dataset, IterableDataset)):
            return dataset

        raise TypeError(
            f"Invalid type for `dataset`: {type(dataset)}. "
            "Expected one of: str, Dataset, DatasetDict, IterableDataset, "
            "IterableDatasetDict."
        )

    def prepare_metrics(
        self,
        metrics: Union[Metric, Sequence[Metric], Dict[str, Metric], MetricCollection],
    ) -> MetricCollection:
        """Prepare metrics for evaluation."""
        # TODO: wrap in BootstrappedMetric if computing confidence intervals
        if isinstance(metrics, (Metric, Sequence, Dict)):
            return MetricCollection(metrics)
        if isinstance(metrics, MetricCollection):
            return metrics

        raise TypeError(
            f"Invalid type for `metrics`: {type(metrics)}. "
            "Expected one of: Metric, Sequence[Metric], Dict[str, Metric], "
            "MetricCollection."
        )

    def prepare_models(
        self,
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

    def compute(  # pylint: disable=too-many-function-args
        self,
        dataset: Union[str, Dataset, DatasetDict, IterableDataset, IterableDatasetDict],
        metrics: Union[Metric, Sequence[Metric], Dict[str, Metric], MetricCollection],
        target_columns: Union[str, List[str]],
        feature_columns: Union[str, List[str]],
        models: Optional[
            Union[WrappedModel, Sequence[WrappedModel], Dict[str, WrappedModel]]
        ] = None,
        slice_spec: Optional[SliceSpec] = None,
        split: Optional[Union[str, Split]] = None,
        load_dataset_kwargs: Optional[Mapping[str, Any]] = None,
        prediction_column_prefix: str = "predictions",
        remove_columns: Optional[Union[str, List[str]]] = None,
        batch_size: int = config.DEFAULT_MAX_BATCH_SIZE,
    ) -> Dict[str, Any]:
        """Compute metrics for a dataset and/or models."""
        dataset = self.load_data(dataset, split, load_dataset_kwargs)  # type: ignore

        column_names: List[str] = list(dataset.features.keys())
        check_required_columns(
            column_names, target_columns, feature_columns, remove_columns
        )

        metrics = self.prepare_metrics(metrics)

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
            models = self.prepare_models(models)

            # TODO: compute predictions for each model and add them to the dataset
            # with name `prediction_column_prefix.model_name`

        # compute metrics for each model
        results = {}

        if slice_spec is None:
            slice_spec = SliceSpec()

        metric_results = self.compute_metrics(
            dataset,
            metrics,
            slice_spec,
            target_columns=target_columns,  # TODO: get target columns from task
            prediction_column_prefix=prediction_column_prefix,
            batch_size=batch_size,
            remove_columns=remove_columns,
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

        # TODO: compute fairness metrics

        return results

    def compute_metrics(
        self,
        dataset: Union[Dataset, IterableDataset],
        metrics: MetricCollection,
        slice_spec: SliceSpec,
        target_columns: Union[str, List[str]],
        prediction_column_prefix: str = "predictions",
        batch_size: int = config.DEFAULT_MAX_BATCH_SIZE,
        remove_columns: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Compute metrics for a dataset."""
        if isinstance(target_columns, str):
            target_columns = [target_columns]

        # drop columns that are not needed for slicing or computing metrics
        if remove_columns is not None:
            dataset = dataset.remove_columns(remove_columns)

        # get the predictions (there could be multiple)
        # any column starting with `prediction_column_prefix` is considered a
        # prediction column, for a single model
        prediction_columns = [
            col
            for col in dataset.features.keys()
            if col.startswith(prediction_column_prefix)
        ]

        dataset.set_format(type="numpy")

        results: Dict[str, Dict[str, Any]] = {}

        for slice_name, slice_fn in slice_spec.get_slices().items():
            sliced_dataset = dataset.filter(
                slice_fn,
                batched=True,
                batch_size=batch_size,
                desc=f"Dataset -> {slice_name}",
            )

            for prediction_column in prediction_columns:
                for batch in sliced_dataset.iter(batch_size=batch_size):
                    # get the targets and predictions
                    targets = np.stack(
                        [batch[target_column] for target_column in target_columns],
                        axis=1,
                    ).squeeze()
                    predictions = batch[prediction_column]

                    # update the metric state
                    metrics.update_state(targets, predictions)

                # get the model name from the prediction column name
                # model name is everything after the first `prediction_column_prefix.`
                model_name: str = "default"
                pred_col_split = prediction_column.split(".", 1)
                if len(pred_col_split) == 2:
                    model_name = pred_col_split[1]

                results.setdefault(model_name, {})
                results[model_name][slice_name] = metrics.compute()
                metrics.reset_state()

        return results
