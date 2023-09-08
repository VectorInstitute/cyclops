"""Mortality Prediction Task."""

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, config
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError

from cyclops.data.slicer import SliceSpec
from cyclops.evaluate.evaluator import evaluate
from cyclops.evaluate.fairness.config import FairnessConfig
from cyclops.evaluate.metrics.factory import create_metric
from cyclops.evaluate.metrics.metric import MetricCollection
from cyclops.models.catalog import _model_names_mapping, _static_model_keys
from cyclops.models.wrappers import WrappedModel
from cyclops.models.wrappers.sk_model import SKModel
from cyclops.models.wrappers.utils import to_numpy
from cyclops.tasks.base import BaseTask
from cyclops.utils.log import setup_logging


LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


class MortalityPredictionTask(BaseTask):
    """Mortality prediction task for tabular data as binary classification."""

    def __init__(
        self,
        models: Union[
            str,
            WrappedModel,
            Sequence[Union[str, WrappedModel]],
            Dict[str, WrappedModel],
        ],
        task_features: Union[str, List[str]] = None,
        task_target: Union[str, List[str]] = None,
    ) -> None:
        """Mortality prediction task for tabular data.

        Parameters
        ----------
        models
            The model(s) to be used for training, prediction, and evaluation.
        task_features : List[str]
            List of feature names.
        task_target : str
            List of target names.

        """
        if task_target is None:
            task_target = ["outcome_death"]
        if task_features is None:
            task_features = ["age", "sex", "admission_type", "admission_location"]
        super().__init__(models, task_features, task_target)

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

    def _validate_models(self):
        """Validate the models for the task data type."""
        assert all(
            _model_names_mapping.get(model.model.__name__) in _static_model_keys
            for model in self.models.values()
        ), "All models must be static type model."

    def list_models_params(self) -> Dict[str, Any]:
        """List the parameters of the models in the task.

        Returns
        -------
        Dict[str, Any]
            Dictionary of model parameters.

        """
        return {n: m.get_params() for n, m in self.models.items()}

    def train(
        self,
        X: Union[np.ndarray, pd.DataFrame, Dataset, DatasetDict],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        model_name: Optional[str] = None,
        transforms: Optional[ColumnTransformer] = None,
        best_model_params: Optional[dict] = None,
        splits_mapping: dict = None,
        **kwargs,
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
        transforms : Optional[ColumnTransformer], optional
            Transformations to be applied to the data before \
                fitting the model, by default Noney default None
        splits_mapping: Optional[dict], optional
            Mapping from 'train', 'validation' and 'test' to dataset splits names \
                used when input is a dataset dictionary, \
                by default {"train": "train", "validation": "validation"}

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
                    y=y,
                    metric=metric,
                    method=method,
                    **kwargs,
                )
            else:
                model.fit(X, y, **kwargs)

        self.trained_models.append(model_name)
        return model

    def predict(
        self,
        dataset: Union[np.ndarray, pd.DataFrame, Dataset, DatasetDict],
        model_name: Optional[str] = None,
        transforms: Optional[ColumnTransformer] = None,
        proba: bool = True,
        splits_mapping: dict = None,
        **kwargs,
    ) -> Union[np.ndarray, Dataset]:
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
        metrics: Union[List[str], MetricCollection],
        model_names: Optional[Union[str, List[str]]] = None,
        transforms: Optional[ColumnTransformer] = None,
        prediction_column_prefix: str = "predictions",
        splits_mapping: dict = None,
        slice_spec: Optional[SliceSpec] = None,
        batch_size: int = config.DEFAULT_MAX_BATCH_SIZE,
        remove_columns: Optional[Union[str, List[str]]] = None,
        fairness_config: Optional[FairnessConfig] = None,
        override_fairness_metrics: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate model(s) on a HuggingFace dataset.

        Parameters
        ----------
        dataset : Union[Dataset, DatasetDict]
            HuggingFace dataset.
        metrics : Union[List[str], MetricCollection]
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

        Returns
        -------
        Dict[str, Any]
            Dictionary with evaluation results.

        """
        if splits_mapping is None:
            splits_mapping = {"test": "test"}
        if isinstance(metrics, list) and len(metrics):
            metrics = [
                create_metric(
                    m,
                    task=self.task_type,
                    num_labels=len(self.task_features),
                )
                for m in metrics
            ]
            metrics = MetricCollection(metrics)

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

        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            target_columns=self.task_target,
            slice_spec=slice_spec,
            prediction_column_prefix=prediction_column_prefix,
            remove_columns=remove_columns,
            split=splits_mapping["test"],
            batch_size=batch_size,
            fairness_config=fairness_config,
            override_fairness_metrics=override_fairness_metrics,
        )
        return results, dataset

    def save_model(
        self,
        filepath: Union[str, Dict[str, str]],
        model_names: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> None:
        """Save the model to a specified filepath.

        Parameters
        ----------
        filepath : Union[str, Dict[str, str]]
            The destination path(s) where the model(s) will be saved.
            Can be a dictionary of model names and their corresponding paths
            or a single parent dirctory.
        model_name : Optional[Union[str, List[str]]], optional
            Model name, required if more than one model exists, by default None.
        **kwargs : Any
            Additional keyword arguments to be passed to the model's save method.

        Returns
        -------
        None

        """
        if isinstance(model_names, str):
            model_names = [model_names]
        elif not model_names:
            model_names = self.trained_models

        if isinstance(filepath, Dict):
            assert len(filepath) == len(model_names), (
                "Number of filepaths must match number of models"
                "if a dictionary is given."
            )
        if isinstance(filepath, str) and len(model_names) > 1:
            assert len(os.path.basename(filepath).split(".")) == 1, (
                "Filepath must be a directory if a single string is given"
                "for multiple models."
            )

        for model_name in model_names:
            if model_name not in self.trained_models:
                LOGGER.warning(
                    "It seems you have not trained the %s model.",
                    model_name,
                )
            model_name, model = self.get_model(model_name)  # noqa: PLW2901
            model_path = (
                filepath[model_name] if isinstance(filepath, Dict) else filepath
            )
            model.save_model(model_path, **kwargs)

    def load_model(
        self,
        filepath: str,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> WrappedModel:
        """Load a pretrained model.

        Parameters
        ----------
        filepath : str
            Path to the save model.
        model_name : Optional[str], optional
            Model name, required if more than one model exists, by default Nonee

        Returns
        -------
        WrappedModel
            The loaded model.

        """
        model_name, model = self.get_model(model_name)
        model.load_model(filepath, **kwargs)
        self.pretrained_models.append(model_name)
        return model
