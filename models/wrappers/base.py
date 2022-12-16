"""Base class for model wrappers."""
from abc import ABC, abstractmethod


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
