"""Base task class."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Union

from cyclops.models.utils import get_device
from cyclops.models.wrappers import WrappedModel
from cyclops.tasks.utils import prepare_models
from cyclops.utils.log import setup_logging


LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


class BaseTask(ABC):
    """Base task class."""

    def __init__(
        self,
        models: Union[
            str,
            WrappedModel,
            Sequence[Union[str, WrappedModel]],
            Dict[str, WrappedModel],
        ],
        task_features: List[str],
        task_target: Union[str, List[str]],
    ) -> None:
        """Initialize base task class.

        Parameters
        ----------
        models
            Models to use for the task. Can be a single model, a list of models, or a
            dictionary of models.
        task_features
            Features to use for the task.
        task_target
            Target to use for the task.

        """
        self.models = prepare_models(models)
        self._validate_models()
        self.task_features = (
            [task_features] if isinstance(task_features, str) else task_features
        )
        self.task_target = (
            [task_target] if isinstance(task_target, str) else task_target
        )
        self.device = get_device()
        self.trained_models = []
        self.pretrained_models = []

    @property
    def models_count(self):
        """Number of models in the task.

        Returns
        -------
        int
            Number of models.

        """
        return len(self.models)

    @property
    @abstractmethod
    def task_type(self) -> str:
        """The classification task type.

        Returns
        -------
        str
            Classification task type.

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def data_type(self) -> str:
        """The data type.

        Returns
        -------
        str
            The data type.

        """
        raise NotImplementedError

    def list_models(self):
        """List the names of the models in the task.

        Returns
        -------
        List[str]
            List of model names.

        """
        return list(self.models.keys())

    @abstractmethod
    def _validate_models(self):
        """Validate the models for the task data type."""
        raise NotImplementedError

    def add_model(
        self,
        model: Union[str, WrappedModel, Dict[str, WrappedModel]],
    ):
        """Add a model to the task.

        Parameters
        ----------
        model : Union[str, WrappedModel, Dict[str, WrappedModel]]
            Model to be added.

        """
        model_dict = prepare_models(model)
        if set(model_dict.keys()).issubset(self.list_models()):
            LOGGER.error(
                "Failed to add the model. A model with same name already exists.",
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
                "You can add the model using Task.add_model()",
            )

        model_name = model_name if model_name else self.list_models()[0]
        model = self.models[model_name]

        return model_name, model
