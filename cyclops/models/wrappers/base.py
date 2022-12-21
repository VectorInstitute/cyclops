"""Base class for model wrappers."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Sequence, Union

import cyclops.models.wrappers.utils as wrapper_utils


class ModelWrapper(ABC):
    """Base class for model wrappers.

    All model wrappers should inherit from this class.

    """

    @abstractmethod
    def partial_fit(self, X, y=None, **fit_params):
        """Fit the model on the given data incrementally.

        Parameters
        ----------
        X
            The features of the data.
        y
            The labels of the data.
        **fit_params : dict, optional
            Additional parameters for fitting the model.

        Returns
        -------
            The fitted model.

        """

    @abstractmethod
    def fit(self, X, y=None, **fit_params):
        """Fit the model on the given data.

        Parameters
        ----------
        X
            The input to the model.
        y
            The output of the model.
        **fit_params : dict, optional
            Additional parameters for fitting the model.

        Returns
        -------
            The fitted model.

        """

    @abstractmethod
    def find_best(
        self,
        X,
        y,
        parameters: Union[Dict, List[Dict]],
        metric: Union[str, Callable, Sequence, Dict] = None,
        method: Literal["grid", "random"] = "grid",
        **kwargs,
    ):
        """Find the best model from hyperparameter search.

        Parameters
        ----------
        X : np.ndarray or torch.utils.data.Dataset
            The features of the data.
        y : np.ndarray
            The labels of the data.
        parameters : dict or list of dicts
            The parameters to search over.
        metric : str or callable, optional
            The metric to use for scoring.
        method : str, default="grid"
            The method to use for hyperparameter search.
        **kwargs : dict, optional
            Additional parameters.

        Returns
        -------
        self

        """

    @abstractmethod
    def predict(self, X, **predict_params):
        """Predict the output of the model for the given input.

        Parameters
        ----------
        X
            The input to the model.
        **predict_params : dict, optional
            Additional parameters for the prediction.

        Returns
        -------
            The output of the model.

        """

    @abstractmethod
    def predict_proba(self, X, **predict_params):
        """Return the output probabilities of the model output for the given input.

        Parameters
        ----------
        X
            The input to the model.
        **predict_params : dict, optional
            Additional parameters for the prediction.

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
