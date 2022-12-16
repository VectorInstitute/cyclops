"""Predictor class to train and test models per use-case."""

import logging
from os import path
from typing import Optional

import yaml

from cyclops.utils.file import process_dir_save_path
from cyclops.utils.log import setup_logging
from models.catalog import (
    _model_catalog,
    _static_model_keys,
    create_model,
)
from models.constants import DATA_TYPES, DATASETS, TASKS, USE_CASES
from models.data import VectorizedLoader, PTDataset
from models.utils import metrics_binary
from models.wrappers import WrappedModel

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
            The model name to use for prediction.
        dataset_name : str
            The dataset name to use for prediction.
        use_case : str
            The use case to use for prediction.
        data_type : str
            The data type to use for prediction.
            One of "tabular", "temporal", or "combined".
        config_file : Optional[str]
            The path to the config file to use for prediction.

        """
        self.model_name = model_name
        self.dataset_name = dataset_name.lower()
        self.use_case = use_case.lower()
        self.data_type = data_type.lower()
        self.config_file = config_file

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
            params = yaml.load(file, Loader=yaml.FullLoader)

        # save_path = join(SAVE_DIR, self.model_name, self.dataset_name, self.data_type)
        # process_dir_save_path(save_path)

        self.model = create_model(self.model_name, **params)

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
        assert path.exists(self.config_file), "[!] Config path does not exist."

    def fit(self) -> WrappedModel:
        """Train the model through appropriate wrapper.

        Returns
        -------
        WrappedModel
            Trained model object.

        """
        return self.model.fit(self.X_train, self.y_train)

    def predict(self) -> tuple:
        """Make prediction by a trained model.

        Returns
        -------
        np.ndarray
            The model predictions.

        """
        test_dataset = PTDataset(self.X_test, self.y_test)
        return self.model.predict(test_dataset)

    def load_model(self, model_path: Optional[str] = None) -> WrappedModel:
        """Load model from a file.

        Parameters
        ----------
        model_path : Optional[str], optional
            The path to the model file, by default None.

        Returns
        -------
        WrappedModel
            The loaded model object.

        """
        return self.model.load_model(model_path)

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
        test_dataset = PTDataset(self.X_test, self.y_test)
        preds = self.model.predict(test_dataset)
        print(self.y_test.shape)
        preds = preds[self.y_test != -1]
        y_pred_labels = preds > 0.5
        y_test_labels = self.y_test[self.y_test != -1]

        return metrics_binary(y_test_labels, preds, y_pred_labels, verbose)
