"""Base class for model wrappers."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union

import cyclops.models.wrappers.utils as wrapper_utils


class ModelWrapper(ABC):
    """Base class for model wrappers.

    All model wrappers should inherit from this class.

    """

    @abstractmethod
    def partial_fit(
        self,
        X,
        y=None,
        feature_columns: Optional[Union[str, List[str]]] = None,
        target_columns: Optional[Union[str, List[str]]] = None,
        splits_mapping: Optional[dict] = None,
        **fit_params,
    ):
        """Fit the model to the data.

        Parameters
        ----------
        X
            The features of the data.
        y
            The labels of the data. This is required when the input dataset is not \
                a huggingface dataset and only contains features, by default None
        feature_columns : Optional[Union[str, List[str]]], optional
            List of feature columns in the dataset. This is required when the input is \
                a Hugging Face Dataset, by default None
        target_columns : Optional[Union[str, List[str]]], optional
            List of target columns in the dataset. This is required when the input is \
                a Hugging Face Dataset, by default None
        splits_mapping: Optional[dict], optional
            Mapping from 'train', 'validation' and 'test' to dataset splits names \
        **fit_params : dict, optional
            Additional parameters to pass to the model's `forward` method.

        Returns
        -------
            The fitted model.

        """

    @abstractmethod
    def fit(
        self,
        X,
        y=None,
        feature_columns: Optional[Union[str, List[str]]] = None,
        target_columns: Optional[Union[str, List[str]]] = None,
        transforms=None,
        splits_mapping: Optional[dict] = None,
        **fit_params,
    ):
        """Fit the model on the given data.

        Parameters
        ----------
        X
            The input to the model.
        y
            The labels of the data. This is required when the input dataset is not \
                a huggingface dataset and only contains features, by default None
        feature_columns : Optional[Union[str, List[str]]], optional
            List of feature columns in the dataset. This is required when the input is \
                a Hugging Face Dataset, by default None
        target_columns : Optional[Union[str, List[str]]], optional
            List of target columns in the dataset. This is required when the input is \
                a Hugging Face Dataset, by default None
        transforms
            The transformation to be applied to the data before prediction, \
                This is used when the input is a Hugging Face Dataset, \
                by default None
        splits_mapping: Optional[dict], optional
            Mapping from 'train', 'validation' and 'test' to dataset splits names \
        **fit_params : dict, optional
            Additional parameters for fitting the model.

        Returns
        -------
            The fitted model.

        """

    @abstractmethod
    def find_best(
        self,
        parameters: Union[Dict, List[Dict]],
        X,
        y=None,
        feature_columns: Optional[Union[str, List[str]]] = None,
        target_columns: Optional[Union[str, List[str]]] = None,
        transforms: Optional[Callable] = None,
        metric: Optional[Union[str, Callable, Sequence, Dict]] = None,
        method: Literal["grid", "random"] = "grid",
        splits_mapping: Optional[dict] = None,
        **kwargs,
    ):
        """Find the best model from hyperparameter search.

        Parameters
        ----------
        parameters : dict or list of dicts
            The hyperparameters to be tuned.
        X
            The data features or a Hugging Face dataset containing features and labels.
        y
            The labels of the data. This is required when the input dataset is not \
                a huggingface dataset and only contains features, by default None
        feature_columns : Optional[Union[str, List[str]]], optional
            List of feature columns in the dataset. This is required when the input is \
                a Hugging Face Dataset, by default None
        target_columns : Optional[Union[str, List[str]]], optional
            List of target columns in the dataset. This is required when the input is \
                a Hugging Face Dataset, by default None
        transforms
            The transformation to be applied to the data before prediction, \
                This is used when the input is a Hugging Face Dataset, \
                by default None
        metric : str, callable, sequence, dict, optional
            The metric to be used for model evaluation.
        method : Literal["grid", "random"], default="grid"
            The tuning method to be used.
        splits_mapping: Optional[dict], optional
            Mapping from 'train', 'validation' and 'test' to dataset splits names, \
                used when input is a dataset dictionary,
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the search method.

        Returns
        -------
        self

        """

    @abstractmethod
    def predict(
        self,
        X,
        feature_columns: Optional[Union[str, List[str]]] = None,
        prediction_column_prefix: Optional[str] = None,
        model_name: Optional[str] = None,
        transforms=None,
        only_predictions: bool = False,
        splits_mapping: Optional[dict] = None,
        **predict_params,
    ):
        """Predict the output of the model for the given input.

        Parameters
        ----------
        X
            The input to the model.
        feature_columns : Optional[Union[str, List[str]]], optional
            List of feature columns in the dataset. This is required when the input is \
                a Hugging Face Dataset
        prediction_column_prefix : str, optional
            Name of the prediction column to be added to the dataset, This is used \
                when the input is a Hugging Face Dataset, by default "predictions"
        model_name : Optional[str], optional
            Model name used as suffix to the prediction column, This is used \
                when the input is a Hugging Face Dataset, by default None
        transforms :
            Transform function to be applied.
                This is used when the input is a Hugging Face Dataset, \
                by default None
        only_predictions : bool, optional
            Whether to return only the predictions rather than the dataset \
                with predictions when the input is a Hugging Face Datset, \
                by default False
        splits_mapping: Optional[dict], optional
            Mapping from 'train', 'validation' and 'test' to dataset splits names, \
                used when input is a dataset dictionary
        **predict_params : dict, optional
            Additional parameters for the prediction.

        Returns
        -------
            The output of the model.

        """

    @abstractmethod
    def predict_proba(
        self,
        X,
        feature_columns: Optional[Union[str, List[str]]] = None,
        prediction_column=None,
        **predict_params,
    ):
        """Return the output probabilities of the model output for the given input.

        Parameters
        ----------
        X
            The input to the model.
        feature_columns : Optional[Union[str, List[str]]], optional
            List of feature columns in the dataset. This is required when the input is \
                a Hugging Face Dataset, by default None
        prediction_column : Optional[Union[str, List[str]]],
            Name of the prediction column to be added to the dataset


        Returns
        -------
            The probabilities of the output of the model.

        """

    @abstractmethod
    def save_model(self, filepath: str, overwrite: bool = True, **kwargs):
        """Save model to file.

        Parameters
        ----------
        filepath : str
            Path to the file where the model is saved.
        overwrite : bool, default=True
            Whether to overwrite the file if it already exists.
        **kwargs : dict, optional
            Additional parameters for saving the model. To be used by the
            specific model wrapper.

        Returns
        -------
        None

        """

    @abstractmethod
    def load_model(self, filepath: str, **kwargs):
        """Load a saved model.

        Parameters
        ----------
        filepath : str
            Path to the file where the model is saved.
        **kwargs : dict, optional
            Additional parameters for loading the model. To be used by the
            specific model wrapper.

        Returns
        -------
        self

        """

    def get_params(self) -> Dict[str, Any]:
        """Get parameters for the wrapper.

        Returns
        -------
        dict
            Parameter names mapped to their values.

        """
        return wrapper_utils.get_params(self)

    def set_params(self, **params):
        """Set the parameters of this wrapper.

        Parameters
        ----------
        **params : dict
            Wrapper parameters.

        Returns
        -------
        self

        """
        wrapper_utils.set_params(self, **params)
