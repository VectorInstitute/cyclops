"""Base task class."""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from cyclops.models.wrappers import WrappedModel
from cyclops.tasks.utils import prepare_models
from cyclops.utils.log import setup_logging


LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


class BaseTask(ABC):
    """Base task class.

    Parameters
    ----------
    models
        Models to use for the task. Can be a single model, a list of models, or a
        dictionary of models.
    task_features
        Features to use for the task.
    task_target
        Target to use for the task.

    Attributes
    ----------
    models
        Models to use for the task.
    task_features
        Features to use for the task.
    task_target
        Target to use for the task.
    trained_models
        List of trained models.
    pretrained_models
        List of pretrained models.

    """

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
        """Initialize base task class."""
        self.models = prepare_models(models)
        self._validate_models()
        self.task_features = task_features
        self.task_target = (
            [task_target] if isinstance(task_target, str) else task_target
        )
        self.trained_models: List[str] = []
        self.pretrained_models: List[str] = []

    @property
    def models_count(self) -> int:
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

    def list_models(self) -> List[str]:
        """List the names of the models in the task.

        Returns
        -------
        List[str]
            List of model names.

        """
        return list(self.models.keys())

    @abstractmethod
    def _validate_models(self) -> None:
        """Validate the models for the task data type."""
        raise NotImplementedError

    def add_model(
        self,
        model: Union[str, WrappedModel, Dict[str, WrappedModel]],
    ) -> None:
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

    def save_model(
        self,
        filepath: Union[str, Dict[str, str]],
        model_names: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> None:
        """Save the model to a specified filepath.

        Parameters
        ----------
        filepath : Union[str, Dict[str, str]]
            The destination path(s) where the model(s) will be saved.
            Can be a dictionary of model names and their corresponding paths
            or a single parent directory.
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
        **kwargs: Any,
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

    def list_models_params(self) -> Dict[str, Any]:
        """List the parameters of the models in the task.

        Returns
        -------
        Dict[str, Any]
            Dictionary of model parameters.

        """
        return {n: m.get_params() for n, m in self.models.items()}
