"""Classification tasks."""


from cyclops.models.catalog import (
    _img_model_keys,
    _model_names_mapping,
    _static_model_keys,
)
from cyclops.tasks.base import BaseTask


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

    def _validate_models(self):
        """Validate the models for the task data type."""
        assert all(
            _model_names_mapping.get(model.model.__name__) in _static_model_keys
            for model in self.models.values()
        ), "All models must be static type model."


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

    def _validate_models(self):
        """Validate the models for the task data type."""
        assert all(
            _model_names_mapping.get(model.model.__name__) in _img_model_keys
            for model in self.models.values()
        ), "All models must be image type model."

        for model in self.models.values():
            model.initialize()
