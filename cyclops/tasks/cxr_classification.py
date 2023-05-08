"""Chest X-ray Classification Task."""
import logging
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from datasets import Dataset, DatasetDict
from monai.transforms import Compose

from cyclops.data.slicer import SliceSpec
from cyclops.evaluate.evaluator import evaluate
from cyclops.evaluate.metrics.factory import create_metric
from cyclops.evaluate.metrics.metric import MetricCollection
from cyclops.models.catalog import _img_model_keys, _model_names_mapping
from cyclops.models.utils import get_device, get_split
from cyclops.models.wrappers import WrappedModel
from cyclops.tasks.utils import CXR_TARGET, apply_image_transforms, prepare_models
from cyclops.utils.log import setup_logging

LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)

# pylint: disable=dangerous-default-value


class CXRClassification:
    """Chest X-ray classification task modeled as a multi-label classification task."""

    def __init__(
        self,
        models: Union[
            str,
            WrappedModel,
            Sequence[Union[str, WrappedModel]],
            Dict[str, WrappedModel],
        ],
        models_config_path: Union[str, Dict[str, str]] = None,
        task_features: List[str] = "image",
        task_target: Union[str, List[str]] = CXR_TARGET,
    ):
        """Chest X-ray classification task.

        Parameters
        ----------
        models
            The model(s) to be used for prediction, and evaluation.
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

    def _validate_models(self):
        """Validate the models for the task data type."""
        assert all(
            _model_names_mapping.get(model.model.__name__) in _img_model_keys
            for model in self.models.values()
        ), "All models must be image classification model."

        for model in self.models.values():
            model.initialize()

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
        return list(self.models.keys())

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

        model_name = model_name if model_name else self.list_models()[0]
        model = self.models[model_name]

        return model_name, model

    def predict(
        self,
        dataset: Union[np.ndarray, Dataset, DatasetDict],
        model_name: Optional[str] = None,
        transforms: Optional[Compose] = None,
        splits_mapping: dict = {"test": "test"},
        **kwargs,
    ) -> Union[np.ndarray, Dataset]:
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
        **kwargs: dict, optional
            Additional parameters for the prediction.

        Returns
        -------
        Union[np.ndarray, Dataset]
            Predicted labels or the Hugging Face dataset with predicted labels.

        """
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
        metrics: Union[List[str], MetricCollection],
        model_names: Union[str, List[str]] = None,
        transforms: Optional[Compose] = None,
        prediction_column_prefix: str = "predictions",
        splits_mapping: dict = {"test": "test"},
        slice_spec: Optional[SliceSpec] = None,
        batch_size: int = 64,
        remove_columns: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Evaluate model(s) on a HuggingFace dataset.

        Parameters
        ----------
        dataset : Union[Dataset, DatasetDict]
            HuggingFace dataset.
        metrics : Union[List[str], MetricCollection]
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

        Returns
        -------
        Dict[str, Any]
            Dictionary with evaluation results.

        """
        if isinstance(dataset, DatasetDict):
            split = get_split(dataset, "test", splits_mapping=splits_mapping)
            dataset = dataset[split]

        missing_labels = [
            label for label in self.task_target if label not in dataset.column_names
        ]
        if len(missing_labels):

            def add_missing_labels(examples):
                for label in missing_labels:
                    examples[label] = 0.0
                return examples

            dataset = dataset.map(add_missing_labels)

        if isinstance(metrics, list) and len(metrics):
            metrics = [
                create_metric(m, task=self.task_type, num_labels=len(self.task_target))
                for m in metrics
            ]
            metrics = MetricCollection(metrics)

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
            dataset,
            metrics,
            slice_spec=slice_spec,
            target_columns=self.task_target,
            prediction_column_prefix=prediction_column_prefix,
            batch_size=batch_size,
            remove_columns=remove_columns,
        )

        return results, dataset
