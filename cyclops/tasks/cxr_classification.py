"""Chest X-ray Classification Task."""
import logging
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, get_args

import numpy as np
from datasets import Dataset
from monai.transforms import Compose

from cyclops.datasets.slicing import SlicingConfig as SliceSpec
from cyclops.evaluate.evaluator import Evaluator
from cyclops.evaluate.metrics.metric import Metric, MetricCollection, create_metric
from cyclops.models.catalog import _img_model_keys
from cyclops.models.utils import get_device
from cyclops.models.wrappers import WrappedModel
from cyclops.tasks.util import apply_image_transforms
from cyclops.utils.log import setup_logging

# from cyclops.datasets.slice import SliceSpec XXX: Add when the branch is merged


LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


class CXRClassification:
    """Chest X-ray classification task modeled as a multi-label classification task."""

    def __init__(
        self,
        models: Union[WrappedModel, Sequence[WrappedModel], Dict[str, WrappedModel]],
        task_features: List[str],
        task_target: str,
    ):
        """Chest X-ray classification task.

        Parameters
        ----------
        models : Union[WrappedModel, Sequence[WrappedModel], Dict[str, WrappedModel]]
            The model(s) to be used for prediction, and evaluation.
        task_features : List[str]
            List of feature names.
        task_target : str
            List of target names.

        """
        self.models = self._prepare_models(models)
        self.task_features = task_features
        self.task_target = task_target
        self.evaluator = Evaluator()
        self.device = get_device()

    @property
    def task_type(self):
        """The classification task type.

        Returns
        -------
        str
            Classification task type.

        """
        return "multilabel"

    @property
    def data_type(self):
        """The data type.

        Returns
        -------
        str
            The data type.

        """
        return "image"

    def _prepare_models(
        self,
        models: Union[WrappedModel, Sequence[WrappedModel], Dict[str, WrappedModel]],
    ) -> Dict[str, WrappedModel]:
        """Prepare and validate the model(s) for the task.

        Parameters
        ----------
        model : Union[WrappedModel, Sequence[WrappedModel], Dict[str, WrappedModel]]
            Initial model(s) of the task.

        Returns
        -------
        model : Dict[str, WrappedModel]
            A dictionary of model(s) with their names as the keys.

        """
        if isinstance(models, get_args(WrappedModel)):
            model_name = models.model.__name__
            assert (
                model_name in _img_model_keys
            ), "Model must be a image classification model."
            models = {model_name: models}  # type: ignore
        elif isinstance(models, (list, tuple)):
            assert all(isinstance(m, get_args(WrappedModel)) for m in models)
            assert all(
                m.model.__name__ in _img_model_keys for m in models
            ), "All models must be image classification model."
            models = {m.model.__name__: m for m in models}
        elif isinstance(models, dict):
            assert all(isinstance(m, get_args(WrappedModel)) for m in models.values())
            assert all(
                m.model.__name__ in _img_model_keys for m in models.values()
            ), "All models must be image classification model."
        else:
            raise ValueError(f"Invalid model type: {type(models)}")

        return models

    @property
    def models_count(self):
        """Number of models in the task.

        Returns
        -------
        int
            Number of models.

        """
        return len(self.models)

    def list_models(self):
        """List the names of the models in the task.

        Returns
        -------
        List[str]
            List of model names.

        """
        models = list(self.models.keys())
        return models if len(models) > 1 else models[0]

    def add_model(self, model: WrappedModel, model_name: Optional[str] = None):
        """Add a model to the task.

        Parameters
        ----------
        model : WrappedModel
            Model to be added.
        model_name : Optional[str], optional
            Model name, by default None

        """
        model_name = model_name if model_name else model.model.__name__
        if model_name in self.list_models():
            LOGGER.error(
                "Failed to add the model. A model with same name already exists."
            )
        else:
            model = self._prepare_models({model_name: model})
            self.models.update(model)
            LOGGER.info("%s is added to task models.", model_name)

    def get_model(self, model_name: str) -> Tuple[str, WrappedModel]:
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

        model_name = model_name if self.models_count > 1 else self.list_models()
        model = self.models[model_name]

        return model_name, model

    def predict(
        self,
        dataset: Union[np.ndarray, Dataset],
        model_name: Optional[str] = None,
        index_column: Optional[str] = None,
        transforms: Optional[Compose] = None,
        **kwargs,
    ) -> Union[np.ndarray, Dataset]:
        """Predict the pathologies on the given dataset.

        Parameters
        ----------
        dataset : Union[np.ndarray, Dataset]
            Image representation as a numpy array or a Hugging Face dataset.
        model_name : Optional[str], optional
             Model name, required if more than one model exists, by default None
        index_column : Optional[str], optional
            The name of the column to identify each item in the dataset,
            required if input is a dataset, by default None
        transforms : Optional[Compose], optional
            Transforms to be applied to the data, by default None

        Returns
        -------
        Union[np.ndarray, Dataset]
            Predicted labels or the Hugging Face dataset with predicted labels.

        """
        model_name, model = self.get_model(model_name)

        if transforms:
            transforms = partial(apply_image_transforms, transforms=transforms)

        if isinstance(dataset, Dataset):
            if not index_column:
                raise ValueError(
                    "Please specify the index_column when input is a dataset."
                )
            return model.predict_on_hf_dataset(
                dataset,
                index_column,
                self.task_features,
                transforms=transforms,
                model_name=model_name,
                **kwargs,
            )

        return model.predict(dataset)

    def evaluate(
        self,
        dataset: Dataset,
        index_column: str,
        metrics: Union[List[str], List[Metric], MetricCollection],
        model_names: Union[str, List[str]] = None,
        transforms: Optional[Compose] = None,
        prediction_column_prefix: str = "predictions",
        slice_spec: Optional[SliceSpec] = None,
        batch_size: int = 64,
        remove_columns: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Evaluate model(s) on a HuggingFace dataset.

        Parameters
        ----------
        dataset : Dataset
            HuggingFace dataset.
        index_column : str
            The name of the column to identify each item in the dataset
        metrics : Union[List[str], List[Metric], MetricCollection]
            Metrics to be evaluated.
        model_names : Union[str, List[str]], optional
            Model names to be evaluated, required if more than one model exists, \
                by default Nonee
        transforms : Optional[Compose], optional
            Transforms to be applied to the data, by default None
        prediction_column_prefix : str, optional
            Name of the prediction column to be added to the dataset, \
                by default "predictions"
        slice_spec : Optional[SliceSpec], optional
            Specifications for creating a slices of a dataset, by default None
        batch_size : int, optional
            Batch size for batched evaluation, by default 64
        remove_columns : Optional[Union[str, List[str]]], optional
            Unnecessary columns to be removed from the dataset, by default None

        Returns
        -------
        Dict[str, Any]
            Dictionary with evaluation results.

        """
        missing_labels = [
            label for label in self.task_target if label not in dataset.column_names
        ]
        if len(missing_labels):
            zeros_column = np.zeros(len(dataset))
            for label_name in missing_labels:
                dataset = dataset.add_column(label_name, zeros_column)

        if isinstance(metrics, list) and len(metrics):
            metrics = [create_metric(m, task=self.task_type) for m in metrics]
        metrics = self.evaluator.prepare_metrics(metrics)  # pylint: disable=no-member

        if isinstance(model_names, str):
            model_names = [model_names]
        elif model_names is None:
            model_names = self.list_models()

        for model_name in model_names:
            dataset = self.predict(
                dataset,
                model_name=model_name,
                index_column=index_column,
                transforms=transforms,
                prediction_column_prefix=prediction_column_prefix,
                batch_size=batch_size,
                only_predictions=False,
            )

        return self.evaluator.compute_metrics(  # pylint: disable=no-member
            dataset,
            metrics,
            slice_spec=slice_spec,
            target_columns=self.task_target,
            prediction_column_prefix=prediction_column_prefix,
            batch_size=batch_size,
            remove_columns=remove_columns,
        )
