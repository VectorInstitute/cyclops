"""Classification tasks."""

import logging
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, config
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from cyclops.data.slicer import SliceSpec
from cyclops.evaluate.evaluator import evaluate
from cyclops.evaluate.fairness.config import FairnessConfig
from cyclops.evaluate.metrics.experimental.metric_dict import MetricDict
from cyclops.evaluate.metrics.factory import create_metric
from cyclops.models.catalog import (
    _img_model_keys,
    _model_names_mapping,
    _static_model_keys,
)
from cyclops.models.utils import get_split
from cyclops.models.wrappers import WrappedModel
from cyclops.models.wrappers.sk_model import SKModel
from cyclops.models.wrappers.utils import to_numpy
from cyclops.tasks.base import BaseTask
from cyclops.tasks.utils import apply_image_transforms
from cyclops.utils.log import setup_logging
from cyclops.utils.optional import import_optional_module


if TYPE_CHECKING:
    from torchvision.transforms import Compose
else:
    Compose = import_optional_module(
        "torchvision.transforms",
        attribute="Compose",
        error="warn",
    )


LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


class BinaryTabularClassificationTask(BaseTask):
    """Binary tabular classification task."""

    @property
    def task_type(self) -> str:
        """The classification task type.

        Returns
        -------
        str
            Classification task type.

        """
        return "binary"

    @property
    def data_type(self) -> str:
        """The data type.

        Returns
        -------
        str
            The data type.

        """
        return "tabular"

    def _validate_models(self) -> None:
        """Validate the models for the task data type."""
        assert all(
            _model_names_mapping.get(model.model.__name__) in _static_model_keys  # type: ignore
            for model in self.models.values()
        ), "All models must be static type model."

    def train(
        self,
        X: Union[np.typing.NDArray[Any], pd.DataFrame, Dataset, DatasetDict],
        y: Optional[Union[np.typing.NDArray[Any], pd.Series]] = None,
        model_name: Optional[str] = None,
        transforms: Optional[Union[ColumnTransformer, Pipeline]] = None,
        best_model_params: Optional[Dict[str, Any]] = None,
        splits_mapping: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> WrappedModel:
        """Fit a model on tabular data.

        Parameters
        ----------
        X_train : Union[np.ndarray, pd.DataFrame, Dataset, DatasetDict]
            Data features.
        y_train : Optional[Union[np.ndarray, pd.Series]]
            Data labels, required when the input data is not a Hugging Face dataset, \
                by default None
        model_name : Optional[str], optional
            Model name, required if more than one model exists, \
                by default None
        transforms : Optional[Union[ColumnTransformer, Pipeline]], optional
            Transformations to be applied to the data before \
                fitting the model, default None
        splits_mapping: Optional[dict], optional
            Mapping from 'train', 'validation' and 'test' to dataset splits names \
                used when input is a dataset dictionary, \
                by default {"train": "train", "validation": "validation"}
        best_model_params : Optional[Dict[str, Any]], optional
            Parameters for finding the best model, by default None
        **kwargs: Any, optional
            Additional parameters for the model.

        Returns
        -------
        WrappedModel
            The trained model.

        """
        if splits_mapping is None:
            splits_mapping = {"train": "train", "validation": "validation"}
        model_name, model = self.get_model(model_name)
        if isinstance(X, (Dataset, DatasetDict)):
            if best_model_params:
                metric = best_model_params.pop("metric", None)
                method = best_model_params.pop("method", "grid")
                model.find_best(
                    best_model_params,
                    X,
                    feature_columns=self.task_features,
                    target_columns=self.task_target,
                    transforms=transforms,
                    metric=metric,
                    method=method,
                    splits_mapping=splits_mapping,
                    **kwargs,
                )
            else:
                model.fit(
                    X,
                    feature_columns=self.task_features,
                    target_columns=self.task_target,
                    transforms=transforms,
                    splits_mapping=splits_mapping,
                    **kwargs,
                )
        else:
            if y is None:
                raise ValueError(
                    "Missing data labels 'y'. Please provide the labels for \
                    the training data when not using a Hugging Face dataset \
                    as the input.",
                )
            X = to_numpy(X)
            if transforms is not None:
                try:
                    X = transforms.transform(X)
                except NotFittedError:
                    X = transforms.fit_transform(X)
            y = to_numpy(y)
            assert len(X) == len(y)
            if best_model_params:
                metric = best_model_params.pop("metric", None)
                method = best_model_params.pop("method", "grid")
                model.find_best(
                    best_model_params,
                    X,
                    y=y,  # type: ignore
                    metric=metric,
                    method=method,
                    **kwargs,
                )
            else:
                model.fit(X, y, **kwargs)  # type: ignore
        self.trained_models.append(model_name)

        return model

    def predict(
        self,
        dataset: Union[np.typing.NDArray[Any], pd.DataFrame, Dataset, DatasetDict],
        model_name: Optional[str] = None,
        transforms: Optional[ColumnTransformer] = None,
        proba: bool = True,
        splits_mapping: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Union[np.typing.NDArray[Any], Dataset]:
        """Predict mortality on the given dataset.

        Parameters
        ----------
        dataset : Union[np.ndarray, pd.DataFrame, Dataset, DatasetDict]
            Data features.
        model_name : Optional[str], optional
            Model name, required if more than one model exists, by default None
        transforms : Optional[ColumnTransformer], optional
            Transformations to be applied to the data before \
                prediction. This is used when the input is a \
                Hugging Face Dataset, by default None, by default None
        proba: bool
            Predict probabilities, default True
        splits_mapping: Optional[dict], optional
            Mapping from 'train', 'validation' and 'test' to dataset splits names, \
            used when input is a dataset dictionary, by default {"test": "test"}
        **kwargs: Any, optional
            Additional parameters for the prediction.

        Returns
        -------
        Union[np.ndarray, Dataset]
            Predicted labels or the Hugging Face dataset with predicted labels.

        Raises
        ------
        NotFittedError
            If the model is not fitted or not loaded with a pretrained estimator.

        """
        if splits_mapping is None:
            splits_mapping = {"test": "test"}
        model_name, model = self.get_model(model_name)
        if model_name not in self.pretrained_models + self.trained_models:
            raise NotFittedError(
                f"It seems you have neither trained the {model_name} model nor \
                loaded a pretrained model.",
            )
        if isinstance(dataset, (Dataset, DatasetDict)):
            if proba and isinstance(model, SKModel):
                return model.predict_proba(
                    dataset,
                    feature_columns=self.task_features,
                    transforms=transforms,
                    model_name=model_name,
                    splits_mapping=splits_mapping,
                    **kwargs,
                )
            return model.predict(
                dataset,
                feature_columns=self.task_features,
                transforms=transforms,
                model_name=model_name,
                splits_mapping=splits_mapping,
                **kwargs,
            )
        dataset = to_numpy(dataset)
        if transforms is not None:
            try:
                dataset = transforms.transform(dataset)
            except NotFittedError:
                LOGGER.warning("Fitting preprocessor on evaluation dataset.")
                dataset = transforms.fit_transform(dataset)
        if proba and isinstance(model, SKModel):
            predictions = model.predict_proba(dataset, **kwargs)
        else:
            predictions = model.predict(dataset, **kwargs)

        return predictions

    def evaluate(
        self,
        dataset: Union[Dataset, DatasetDict],
        metrics: Union[List[str], MetricDict],
        model_names: Optional[Union[str, List[str]]] = None,
        transforms: Optional[ColumnTransformer] = None,
        prediction_column_prefix: str = "predictions",
        splits_mapping: Optional[Dict[str, str]] = None,
        slice_spec: Optional[SliceSpec] = None,
        batch_size: int = config.DEFAULT_MAX_BATCH_SIZE,
        remove_columns: Optional[Union[str, List[str]]] = None,
        fairness_config: Optional[FairnessConfig] = None,
        override_fairness_metrics: bool = False,
        array_lib: Literal["numpy", "torch", "cupy"] = "numpy",
    ) -> Tuple[Dict[str, Any], Dataset]:
        """Evaluate model(s) on a HuggingFace dataset.

        Parameters
        ----------
        dataset : Union[Dataset, DatasetDict]
            HuggingFace dataset.
        metrics : Union[List[str], MetricDict]
            Metrics to be evaluated.
        model_names : Union[str, List[str]], optional
            Model names to be evaluated, if not specified all fitted models \
                will be used for evaluation, by default None
        transforms : Optional[ColumnTransformer], optional
            Transformations to be applied to the data before prediction, \
                by default None
        prediction_column_prefix : str, optional
            Name of the prediction column to be added to \
                the dataset, by default "predictions"
        splits_mapping: Optional[dict], optional
            Mapping from 'train', 'validation' and 'test' to dataset splits names \
                used when input is a dataset dictionary, by default {"test": "test"}
        slice_spec : Optional[SlicingConfig], optional
            Specifications for creating a slices of a dataset, by default None
        batch_size : int, optional
            Batch size for batched prediction and evaluation, \
                by default config.DEFAULT_MAX_BATCH_SIZE
        remove_columns : Optional[Union[str, List[str]]], optional
            Unnecessary columns to be removed from the dataset, by default None
        fairness_config : Optional[FairnessConfig], optional
            The configuration for computing fairness metrics. If None, no fairness \
            metrics will be computed, by default None
        override_fairness_metrics : bool, optional
            If True, the `metrics` argument in fairness_config will be overridden by \
            the `metrics`, by default False
        array_lib : {"numpy", "torch", "cupy"}, default="numpy"
            The array library to use for the metric computation. The metric results
            will be returned in the format of `array_lib`.

        Returns
        -------
        Dict[str, Any]
            Dictionary with evaluation results.

        """
        if splits_mapping is None:
            splits_mapping = {"test": "test"}
        if isinstance(metrics, list) and len(metrics):
            metrics_collection = MetricDict(
                [
                    create_metric(  # type: ignore[misc]
                        m,
                        task=self.task_type,
                        num_labels=len(self.task_features),
                    )
                    for m in metrics
                ],
            )
        elif isinstance(metrics, MetricDict):
            metrics_collection = metrics
        if isinstance(model_names, str):
            model_names = [model_names]
        elif not model_names:
            model_names = self.pretrained_models + self.trained_models
        for model_name in model_names:
            if model_name not in self.pretrained_models + self.trained_models:
                LOGGER.warning(
                    "It seems you have neither trained the model nor \
                    loaded a pretrained model.",
                )
            dataset = self.predict(
                dataset,
                model_name=model_name,
                transforms=transforms,
                prediction_column_prefix=prediction_column_prefix,
                only_predictions=False,
                splits_mapping=splits_mapping,
            )

            # select the probability scores of the positive class since metrics
            # expect a single column of probabilities
            dataset = dataset.map(  # type: ignore[union-attr]
                lambda examples: {
                    f"{prediction_column_prefix}.{model_name}": np.array(  # noqa: B023
                        examples,
                    )[
                        :,
                        1,
                    ].tolist(),
                },
                batched=True,
                batch_size=batch_size,
                input_columns=f"{prediction_column_prefix}.{model_name}",
            )
        results = evaluate(
            dataset=dataset,
            metrics=metrics_collection,
            target_columns=self.task_target,
            slice_spec=slice_spec,
            prediction_columns=[
                f"{prediction_column_prefix}.{model_name}" for model_name in model_names
            ],
            ignore_columns=remove_columns,
            split=splits_mapping["test"],
            batch_size=batch_size,
            fairness_config=fairness_config,
            override_fairness_metrics=override_fairness_metrics,
            array_lib=array_lib,
        )
        return results, dataset


class MultilabelImageClassificationTask(BaseTask):
    """Binary tabular classification task."""

    @property
    def task_type(self) -> str:
        """The classification task type.

        Returns
        -------
        str
            Classification task type.

        """
        return "multilabel"

    @property
    def data_type(self) -> str:
        """The data type.

        Returns
        -------
        str
            The data type.

        """
        return "image"

    def _validate_models(self) -> None:
        """Validate the models for the task data type."""
        assert all(
            _model_names_mapping.get(model.model.__name__) in _img_model_keys  # type: ignore
            for model in self.models.values()
        ), "All models must be image type model."
        for model in self.models.values():
            model.initialize()

    def predict(
        self,
        dataset: Union[np.typing.NDArray[Any], Dataset, DatasetDict],
        model_name: Optional[str] = None,
        transforms: Optional[Compose] = None,
        splits_mapping: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Union[np.typing.NDArray[Any], Dataset]:
        """Predict the pathologies on the given dataset.

        Parameters
        ----------
        dataset : Union[np.ndarray, Dataset, DatasetDict]
            Image representation as a numpy array or a Hugging Face dataset.
        model_name : Optional[str], optional
             Model name, required if more than one model exists, by default None
        transforms : Optional[Compose], optional
            Transforms to be applied to the data, by default None
        splits_mapping: Optional[dict], optional
            Mapping from 'train', 'validation' and 'test' to dataset splits names, \
                used when input is a dataset dictionary, by default {"test": "test"}
        **kwargs: Any, optional
            Additional parameters for the prediction.

        Returns
        -------
        Union[np.typing.NDArray[Any], Dataset]
            Predicted labels or the Hugging Face dataset with predicted labels.

        """
        if splits_mapping is None:
            splits_mapping = {"test": "test"}
        model_name, model = self.get_model(model_name)
        if transforms:
            transforms = partial(apply_image_transforms, transforms=transforms)
        if isinstance(dataset, (Dataset, DatasetDict)):
            return model.predict(
                dataset,
                feature_columns=self.task_features,
                transforms=transforms,
                model_name=model_name,
                splits_mapping=splits_mapping,
                **kwargs,
            )

        return model.predict(dataset, **kwargs)

    def evaluate(
        self,
        dataset: Union[Dataset, DatasetDict],
        metrics: Union[List[str], MetricDict],
        model_names: Optional[Union[str, List[str]]] = None,
        transforms: Optional[Compose] = None,
        prediction_column_prefix: str = "predictions",
        splits_mapping: Optional[Dict[str, str]] = None,
        slice_spec: Optional[SliceSpec] = None,
        batch_size: int = 64,
        remove_columns: Optional[Union[str, List[str]]] = None,
        fairness_config: Optional[FairnessConfig] = None,
        override_fairness_metrics: bool = False,
        array_lib: Literal["numpy", "torch", "cupy"] = "numpy",
    ) -> Tuple[Dict[str, Any], Dataset]:
        """Evaluate model(s) on a HuggingFace dataset.

        Parameters
        ----------
        dataset : Union[Dataset, DatasetDict]
            HuggingFace dataset.
        metrics : Union[List[str], MetricDict]
            Metrics to be evaluated.
        model_names : Union[str, List[str]], optional
            Model names to be evaluated, required if more than one model exists, \
                by default Nonee
        transforms : Optional[Compose], optional
            Transforms to be applied to the data, by default None
        prediction_column_prefix : str, optional
            Name of the prediction column to be added to the dataset, \
                by default "predictions"
        splits_mapping: Optional[dict], optional
            Mapping from 'train', 'validation' and 'test' to dataset splits names \
                used when input is a dataset dictionary, by default {"test": "test"}
        slice_spec : Optional[SliceSpec], optional
            Specifications for creating a slices of a dataset, by default None
        batch_size : int, optional
            Batch size for batched evaluation, by default 64
        remove_columns : Optional[Union[str, List[str]]], optional
            Unnecessary columns to be removed from the dataset, by default None
        fairness_config : Optional[FairnessConfig], optional
            The configuration for computing fairness metrics. If None, no fairness \
            metrics will be computed, by default None
        override_fairness_metrics : bool, optional
            If True, the `metrics` argument in fairness_config will be overridden by \
            the `metrics`, by default False
        array_lib : {"numpy", "torch", "cupy"}, default="numpy"
            The array library to use for the metric computation. The metric results
            will be returned in the format of `array_lib`.

        Returns
        -------
        Dict[str, Any]
            Dictionary with evaluation results.

        """
        if splits_mapping is None:
            splits_mapping = {"test": "test"}
        if isinstance(dataset, DatasetDict):
            split = get_split(dataset, "test", splits_mapping=splits_mapping)
            dataset = dataset[split]

        missing_labels = [
            label for label in self.task_target if label not in dataset.column_names
        ]
        if len(missing_labels):

            def add_missing_labels(examples: Dict[str, Any]) -> Dict[str, Any]:
                for label in missing_labels:
                    examples[label] = 0.0
                return examples

            dataset = dataset.map(add_missing_labels)
        if isinstance(metrics, list) and len(metrics):
            metrics_collection = MetricDict(
                [
                    create_metric(  # type: ignore[misc]
                        m,
                        task=self.task_type,
                        num_labels=len(self.task_target),
                    )
                    for m in metrics
                ],
            )
        elif isinstance(metrics, MetricDict):
            metrics_collection = metrics
        if isinstance(model_names, str):
            model_names = [model_names]
        elif model_names is None:
            model_names = self.list_models()
        for model_name in model_names:
            dataset = self.predict(
                dataset,
                model_name=model_name,
                transforms=transforms,
                prediction_column_prefix=prediction_column_prefix,
                only_predictions=False,
                splits_mapping=splits_mapping,
            )
        results = evaluate(
            dataset=dataset,
            metrics=metrics_collection,
            slice_spec=slice_spec,
            target_columns=self.task_target,
            prediction_columns=[
                f"{prediction_column_prefix}.{model_name}" for model_name in model_names
            ],
            ignore_columns=remove_columns,
            split=splits_mapping["test"],
            batch_size=batch_size,
            fairness_config=fairness_config,
            override_fairness_metrics=override_fairness_metrics,
            array_lib=array_lib,
        )

        return results, dataset
