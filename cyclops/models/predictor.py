"""Predictor class to train and test models per use-case."""

import logging
from os import path
from typing import Optional

import yaml

from cyclops.models.catalog import MODELS, PT_MODELS, STATIC_MODELS, _Model
from cyclops.models.constants import (
    CONFIG_FILE,
    DATA_TYPES,
    DATASETS,
    SAVE_DIR,
    TASKS,
    USE_CASES,
)
from cyclops.models.data import VectorizedLoader
from cyclops.models.util import get_device, metrics_binary
from cyclops.models.wrapper import PTModel, SKModel
from cyclops.utils.file import join, process_dir_save_path
from cyclops.utils.log import setup_logging

LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)

# mypy: ignore-errors
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments, invalid-name


class Predictor:
    """Predictor class."""

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        use_case: str,
        data_type: str,
        config_file: Optional[str],
    ) -> None:
        """Initialize predictor.

        Parameters
        ----------
        model_name : str
            predictor model name
        dataset_name : str
            dataset name to use for training and testing
        use_case : str
            use-case to be predicted
        data_type : str
            type of data (tabular, temporal, or combined)
        config_file : Optional[str]
            path to the config file with parameters values

        """
        self.model_name = model_name.lower()
        self.dataset_name = dataset_name.lower()
        self.use_case = use_case.lower()
        self.data_type = data_type.lower()
        self.config_file = config_file if config_file else CONFIG_FILE
        self.device = get_device()

        self._validate()

        self.dataset = VectorizedLoader(
            self.dataset_name, self.use_case, self.data_type
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
            params = yaml.load(file, Loader=yaml.FullLoader)[self.model_name]

        save_path = join(SAVE_DIR, self.model_name, self.dataset_name, self.data_type)
        process_dir_save_path(save_path)

        if self.model_name in PT_MODELS:
            params["model_params"]["input_dim"] = self.dataset.n_features
            params["model_params"]["device"] = self.device
            if params["train_params"]["reweight"] == "total":
                params["train_params"]["reweight"] = (self.y_train == 0).sum() / (
                    self.y_train == 1
                ).sum()
            self.model = PTModel(self.model_name, save_path, **params)
        else:
            self.model = SKModel(self.model_name, save_path, **params)

    def _validate(self) -> None:
        """Validate the input arguments."""
        assert self.model_name in MODELS, "[!] Invalid model name"
        assert self.dataset_name in DATASETS, "[!] Invalid dataset name"
        assert (
            self.use_case in USE_CASES.keys()  # pylint: disable=C0201
        ), "[!] Invalid use case"
        assert (
            self.dataset_name in USE_CASES[self.use_case]
        ), "[!] Unsupported use case for this dataset"
        assert self.data_type in DATA_TYPES, "[!] Invalid data type"
        if self.model_name in STATIC_MODELS:
            assert (
                self.data_type == "tabular"
            ), "[!] The data type and model are not compatiable"
        assert path.exists(self.config_file), "[!] Config path does not exist."

    def fit(self) -> _Model:
        """Train the model through appropriate wrapper.

        Returns
        -------
        _Model
            trained model

        """
        return self.model.fit(self.X_train, self.y_train, self.X_val, self.y_val)

    def predict(self, trained_model: _Model) -> tuple:
        """Make prediction by a trained model.

        Parameters
        ----------
        trained_model : _Model
            pretrained model

        Returns
        -------
        tuple
            test label, predicted values, predicted labels

        """
        return self.model.predict(trained_model, self.X_test, self.y_test)

    def load_model(self, model_path: Optional[str] = None) -> _Model:
        """Load model from a file.

        Parameters
        ----------
        model_path : Optional[str], optional
            path to the saved model/checkpoint, by default None

        Returns
        -------
        _Model
            model object

        """
        return self.model.load_model(model_path)

    def evaluate(self, trained_model: _Model, verbose: bool) -> dict:
        """Evaluate a trained model based on various metrics.

        Parameters
        ----------
        trained_model : _Model
            pretrained model
        verbose : bool
            print the evaluation results

        Returns
        -------
        dict
            dictionary of evaluation metrics and values

        """
        if self.use_case in TASKS["binary_classification"]:
            return self.evaluate_binary(trained_model, verbose)
        return {}

    def evaluate_binary(self, trained_model: _Model, verbose: bool) -> dict:
        """Evaluate a model on binary calssification metrics.

        Parameters
        ----------
        trained_model : _Model
            pretrained model
        verbose : bool
            print the evaluation results

        Returns
        -------
        dict
            dictionary of evaluation metrics and values

        """
        y_test_labels, y_pred_values, y_pred_labels = self.predict(trained_model)
        print(y_test_labels.shape)
        y_pred_values = y_pred_values[y_test_labels != -1]
        y_pred_labels = y_pred_labels[y_test_labels != -1]
        y_test_labels = y_test_labels[y_test_labels != -1]

        return metrics_binary(y_test_labels, y_pred_values, y_pred_labels, verbose)
