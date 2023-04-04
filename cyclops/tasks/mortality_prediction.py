"""Mortality Prediction Task."""
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset, config
from multipledispatch import dispatch
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError

from cyclops.datasets.slicer import SliceSpec
from cyclops.evaluate.evaluator import evaluate
from cyclops.evaluate.metrics.factory import create_metric
from cyclops.evaluate.metrics.metric import MetricCollection
from cyclops.models.catalog import _static_model_keys
from cyclops.models.wrappers import WrappedModel
from cyclops.tasks.utils import prepare_models, to_numpy
from cyclops.utils.log import setup_logging

LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)

# pylint: disable=function-redefined, dangerous-default-value
# noqa: F811


class MortalityPrediction:
    """Mortality prediction task for tabular data as binary classification."""

    def __init__(
        self,
        models: Union[
            str,
            WrappedModel,
            Sequence[Union[str, WrappedModel]],
            Dict[str, WrappedModel],
        ],
        models_config_path: Union[str, Dict[str, str]] = None,
        task_features: List[str] = [
            "age",
            "sex",
            "admission_type",
            "admission_location",
        ],
        task_target: List[str] = ["outcome_death"],
    ):
        """Mortality prediction task for tabular data.

        Parameters
        ----------
        models : Union[
            str, WrappedModel,
            Sequence[Union[str, WrappedModel]],
            Dict[str, WrappedModel],
        ],
            The model(s) to be used for training, prediction, and evaluation.
        models_config_path : Union[str, Dict[str, str]], optional
            Path to the configuration file(s) for the model(s), by default None
        task_features : List[str]
            List of feature names.
        task_target : str
            List of target names.

        """
        self.models = prepare_models(models, models_config_path)
        self._validate_models()
        self.task_features = task_features
        self.task_target = task_target
        self.trained_models = []
        self.pretrained_models = []

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
            m.model.__name__ in _static_model_keys for m in self.models.values()
        ), "All models must be image classification model."

    @property
    def models_count(self) -> int:
        """Number of models in the task.

        Returns
        -------
        int
            Number of models.

        """
        return len(self.models)

    def list_models(self) -> List[str]:
        """List the names of the models in the task.

        Returns
        -------
        List[str]
            List of model names.

        """
        return list(self.models.keys())

    def list_models_params(self) -> Dict[str, Any]:
        """List the parameters of the models in the task.

        Returns
        -------
        Dict[str, Any]
            Dictionary of model parameters.

        """
        return {n: m.get_params() for n, m in self.models.items()}

    def add_model(
        self,
        model: Union[str, WrappedModel, Dict[str, WrappedModel]],
        model_config_path: Optional[str] = None,
    ):
        """Add a model to the task.

        Parameters
        ----------
        model : Union[str, WrappedModel, Dict[str, WrappedModel]]
            Model to be added.
        model_config_path : Optional[str], optional
            Path to the configuration file for the model.

        """
        model_dict = prepare_models(model, model_config_path)
        if set(model_dict.keys()).issubset(self.list_models()):
            LOGGER.error(
                "Failed to add the model. A model with same name already exists."
            )
        else:
            self.models.update(model_dict)
            LOGGER.info("%s is added to task models.", ", ".join(model_dict.keys()))

    def get_model(self, model_name: Optional[str] = None) -> Tuple[str, WrappedModel]:
        """Get a model. If more than one model exists, the name should be specified.

        Parameters
        ----------
        model_name : Optional[str], optional
            Model name, required if more than one model exists, by default None

        Returns
        -------
        Tuple[str, WrappedModel]
            The model name and the model object.

        Raises
        ------
        ValueError
            If more than one model exists and no name is specified.
        ValueError
            If no model exists with the specified name.

        """
        if self.models_count > 1 and not model_name:
            raise ValueError(f"Please specify a model from {self.list_models()}")
        if model_name and model_name not in self.list_models():
            raise ValueError(
                f"The model {model_name} does not exist. "
                "You can add the model using Task.add_model()"
            )

        model_name = model_name if self.models_count > 1 else self.list_models()[0]
        model = self.models[model_name]

        return model_name, model

    @dispatch(Dataset)
    def train(
        self,
        dataset: Dataset,
        model_name: Optional[str] = None,
        preprocessor: Optional[ColumnTransformer] = None,
        batch_size: int = config.DEFAULT_MAX_BATCH_SIZE,
        **kwargs,
    ) -> WrappedModel:
        """Train a model on a HuggingFace dataset.

        Parameters
        ----------
        dataset : Dataset
            HuggingFace dataset.
        model_name : Optional[str], optional
            Model name, required if more than one model exists, by default None
        preprocessor : Optional[ColumnTransformer], optional
            Transformations to be applied to the dataset before \
                fitting the model, by default None
        batch_size : int, optional
            Size of the batch, by default config.DEFAULT_MAX_BATCH_SIZE

        Returns
        -------
        WrappedModel
            The trained model.

        """
        model_name, model = self.get_model(model_name)
        model.fit(
            dataset,
            self.task_features,
            self.task_target,
            preprocessor=preprocessor,
            batch_size=batch_size,
            **kwargs,
        )

        self.trained_models.append(model_name)
        return model

    @dispatch((np.ndarray, pd.DataFrame), (np.ndarray, pd.Series))
    def train(  # noqa: F811
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        model_name: Optional[str] = None,
        preprocessor: Optional[ColumnTransformer] = None,
        best_model_params: Optional[Dict] = None,
        **kwargs,
    ) -> WrappedModel:
        """Fit a model on tabular data.

        Parameters
        ----------
        X_train : Union[np.ndarray, pd.DataFrame]
            Data features.
        y_train : Union[np.ndarray, pd.Series]
            Data labels.
        model_name : Optional[str], optional
            Model name, required if more than one model exists, \
                by default None
        preprocessor : Optional[ColumnTransformer], optional
            Transformations to be applied to the data before \
                fitting the model, by default Noney default None
        best_model_params : Optional[Dict], optional
            Parameters for finding the best model from hyperparameter search, \
                by default None

        Returns
        -------
        WrappedModel
            The trained model.

        """
        # TODO: add support for evaluation metrics # pylint: disable=fixme

        X_train = to_numpy(X_train)

        if preprocessor is not None:
            try:
                X_train = preprocessor.transform(X_train)
            except NotFittedError:
                X_train = preprocessor.fit_transform(X_train)

        y_train = to_numpy(y_train)

        assert len(X_train) == len(y_train)

        model_name, model = self.get_model(model_name)

        if best_model_params:
            try:
                metric = (
                    best_model_params.pop("metric")
                    if best_model_params.get("metric")
                    else "auc-roc"
                )
                method = (
                    best_model_params.pop("method")
                    if best_model_params.get("method")
                    else "grid"
                )
                model.find_best(
                    X_train,
                    y_train,
                    parameters=best_model_params,
                    metric=metric,
                    method=method,
                    **kwargs,
                )
            except AttributeError:
                LOGGER.warning(
                    "Model %s does not have a `find_best` method.", model_name
                )
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        self.trained_models.append(model_name)
        return model

    def load_pretrained_model(
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

    def predict(
        self,
        dataset: Union[np.ndarray, pd.DataFrame, Dataset],
        model_name: Optional[str] = None,
        preprocessor: Optional[ColumnTransformer] = None,
        proba: bool = True,
        **kwargs,
    ) -> Union[np.ndarray, Dataset]:
        """Predict mortality on the given dataset.

        Parameters
        ----------
        dataset : Union[np.ndarray, pd.DataFrame, Dataset]
            Data features.
        model_name : Optional[str], optional
            Model name, required if more than one model exists, by default None
        preprocessor : Optional[ColumnTransformer], optional
            Transformations to be applied to the data before \
                fitting the model, by default None
        proba : bool, optional
            Whether to output the prediction probabilities rather than \
                predicted classes, by default True

        Returns
        -------
        Union[np.ndarray, Dataset]
            Predicted labels or the Hugging Face dataset with predicted labels.

        Raises
        ------
        NotFittedError
            If the model is not fitted or not loaded with a pretrained estimator.

        """
        model_name, model = self.get_model(model_name)
        if model_name not in self.pretrained_models + self.trained_models:
            raise NotFittedError(
                "It seems you have neither trained the model nor \
                loaded a pretrained model."
            )

        if isinstance(dataset, Dataset):
            return model.predict(
                dataset,
                self.task_features,
                preprocessor=preprocessor,
                proba=proba,
                model_name=model_name,
                **kwargs,
            )

        dataset = to_numpy(dataset)

        if preprocessor is not None:
            try:
                dataset = preprocessor.transform(dataset)
            except NotFittedError:
                LOGGER.warning("Fitting preprocessor on evaluation dataset.")
                dataset = preprocessor.fit_transform(dataset)

        if proba and hasattr(model, "predict_proba"):
            predictions = model.predict_proba(dataset)
        else:
            predictions = model.predict(dataset)

        return predictions

    def evaluate(
        self,
        dataset: Dataset,
        metrics: Union[List[str], MetricCollection],
        model_names: Union[str, List[str]] = None,
        preprocessor: Optional[ColumnTransformer] = None,
        prediction_column_prefix: str = "predictions",
        slice_spec: Optional[SliceSpec] = None,
        batch_size: int = config.DEFAULT_MAX_BATCH_SIZE,
        remove_columns: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Evaluate model(s) on a HuggingFace dataset.

        Parameters
        ----------
        dataset : Dataset
            HuggingFace dataset.
        metrics : Union[List[str], MetricCollection]
            Metrics to be evaluated.
        model_names : Union[str, List[str]], optional
            Model names to be evaluated, if not specified all fitted models \
                will be used for evaluation, by default None
        preprocessor : Optional[ColumnTransformer], optional
            Transformations to be applied to the data before prediction, \
                by default None
        prediction_column_prefix : str, optional
            Name of the prediction column to be added to \
                the dataset, by default "predictions"
        slice_spec : Optional[SlicingConfig], optional
            Specifications for creating a slices of a dataset, by default None
        batch_size : int, optional
            Batch size for batched prediction and evaluation, \
                by default config.DEFAULT_MAX_BATCH_SIZE
        remove_columns : Optional[Union[str, List[str]]], optional
            Unnecessary columns to be removed from the dataset, by default None

        Returns
        -------
        Dict[str, Any]
            Dictionary with evaluation results.

        """
        if isinstance(metrics, list) and len(metrics):
            metrics = [
                create_metric(
                    m, task=self.task_type, num_labels=len(self.task_features)
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
                    loaded a pretrained model."
                )

            dataset = self.predict(
                dataset,
                model_name=model_name,
                preprocessor=preprocessor,
                proba=True,
                prediction_column_prefix=prediction_column_prefix,
                batch_size=batch_size,
                only_predictions=False,
            )

        return evaluate(
            dataset,
            metrics,
            target_columns=self.task_target,
            slice_spec=slice_spec,
            prediction_column_prefix=prediction_column_prefix,
            batch_size=batch_size,
            remove_columns=remove_columns,
        )
