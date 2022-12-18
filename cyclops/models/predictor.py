"""Predictor class to train and test models per use-case."""
import logging
from os import path
from typing import Optional

import numpy as np
import yaml

from cyclops.models.catalog import _model_catalog, _static_model_keys, create_model
from cyclops.models.constants import (
    CONFIG_ROOT,
    DATA_DIR,
    DATA_TYPES,
    DATASETS,
    SAVE_DIR,
    TASKS,
    USE_CASES,
)
from cyclops.models.data import PTDataset, VectorizedLoader
from cyclops.models.utils import metrics_binary
from cyclops.models.wrappers import PTModel, WrappedModel
from cyclops.utils.file import join
from cyclops.utils.log import setup_logging

LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


class Predictor:
    """Predictor class."""

    def __init__(  # pylint: disable=too-many-arguments,
        self,
        model_name: str,
        dataset_name: str,
        use_case: str,
        data_type: str,
        config_file: Optional[str] = None,
        data_dir: Optional[str] = None,
        save_dir: Optional[str] = None,
    ) -> None:
        """Initialize predictor.

        Parameters
        ----------
        model_name : str
            Predictor model name.
        dataset_name : str
            Dataset name to use for training and testing.
        use_case : str
            Use-case to be predicted.
        data_type : str
            Type of data (tabular, temporal, or combined).
        config_file : Optional[str]
            Path to the config file with parameters values.
        data_dir : Optional[str]
            Path to the directory of final vectorized data.
        save_dir : Optional[str]
            Path to the directory to save checkpoints.

        """
        self.model_name = model_name
        self.dataset_name = dataset_name.lower()
        self.use_case = use_case.lower()
        self.data_type = data_type.lower()
        self.config_file = (
            config_file if config_file else join(CONFIG_ROOT, self.model_name + ".yaml")
        )
        self.data_dir = (
            data_dir
            if data_dir
            else join(
                DATA_DIR,
                self.dataset_name,
                self.use_case,
                "data",
                "4-final",
            )
        )
        self.save_dir = save_dir if save_dir else SAVE_DIR

        self._validate()

        self.dataset = VectorizedLoader(
            self.dataset_name, self.use_case, self.data_type, self.data_dir
        )

        (
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
        ) = self.dataset.data

        with open(self.config_file, "r", encoding="utf8") as file:
            params = yaml.load(file, Loader=yaml.FullLoader)

        # save_path = join(SAVE_DIR, self.model_name, self.dataset_name, self.data_type)
        # process_dir_save_path(save_path)
        model_params = params.get("model_params", None) or params

        self.model = create_model(self.model_name, **model_params)

    def _validate(self) -> None:
        """Validate the input arguments."""
        assert self.model_name in _model_catalog, "[!] Invalid model name"
        assert self.dataset_name in DATASETS, "[!] Invalid dataset name"
        assert (
            self.use_case in USE_CASES.keys()  # pylint: disable=C0201
        ), "[!] Invalid use case"
        assert (
            self.dataset_name in USE_CASES[self.use_case]
        ), "[!] Unsupported use case for this dataset"
        assert self.data_type in DATA_TYPES, "[!] Invalid data type"
        if self.model_name in _static_model_keys:
            assert (
                self.data_type == "tabular"
            ), "[!] The data type and model are not compatiable"
        assert path.exists(self.data_dir), "[!] Data path does not exist."
        assert path.exists(self.config_file), "[!] Config path does not exist."

    def fit(self):
        """Train the model through appropriate wrapper.

        Returns
        -------
        self

        """
        self.model = self.model.fit(self.X_train, self.y_train)
        return self

    def predict(self) -> np.ndarray:
        """Make prediction by a trained model.

        Returns
        -------
        np.ndarray
            The model predictions.

        """
        if isinstance(self.model, PTModel):
            test_dataset = PTDataset(self.X_test, self.y_test)
            return self.model.predict(test_dataset)
        return self.model.predict(self.X_test)

    def load_model(self, filepath: str, **kwargs) -> WrappedModel:
        """Load model from a file.

        Parameters
        ----------
        filepath : str
            Path to the model file.
        **kwargs
            Additional arguments.

        Returns
        -------
        WrappedModel
            The loaded model object.

        """
        return self.model.load_model(filepath, **kwargs)

    def evaluate(self, verbose: bool) -> dict:
        """Evaluate a trained model based on various metrics.

        Parameters
        ----------
        verbose : bool
            Print the evaluation results.

        Returns
        -------
        dict
            Dictionary of evaluation metrics and values.

        """
        if self.use_case in TASKS["binary_classification"]:
            return self.evaluate_binary(verbose)
        return {}

    def evaluate_binary(self, verbose: bool) -> dict:
        """Evaluate a model on binary calssification metrics.

        Parameters
        ----------
        verbose : bool
            Print the evaluation results.

        Returns
        -------
        dict
            Dictionary of evaluation metrics and values.

        """
        preds = self.predict()
        preds = preds[self.y_test != -1]
        y_pred_labels = preds > 0.5  # HACK: hard-coded threshold
        y_test_labels = self.y_test[self.y_test != -1]

        return metrics_binary(y_test_labels, preds, y_pred_labels, verbose)
