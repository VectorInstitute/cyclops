"""Scikit-learn model wrapper."""

import logging
import os
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
from datasets import Dataset, config
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator as SKBaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from cyclops.models.utils import is_out_of_core, is_sklearn_class, is_sklearn_instance
from cyclops.utils.file import join, load_pickle, save_pickle
from cyclops.utils.log import setup_logging

LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)

# pylint: disable=fixme


class SKModel:
    """Scikit-learn model wrapper.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        Scikit-learn model instance or class.
    **kwargs : dict, optional
        Additional keyword arguments to pass to model.

    Notes
    -----
    This wrapper does not inherit from models.wrappers.base.ModelWrapper
    because it uses the decorator pattern to expose the sklearn API, which
    is what the base wrapper is meant to abstract away.

    """

    def __init__(self, model: SKBaseEstimator, **kwargs) -> None:
        """Initialize wrapper."""
        self.model = model  # possibly uninstantiated class
        self.initialize_model(**kwargs)

    def initialize_model(self, **kwargs):
        """Initialize model.

        Parameters
        ----------
        kwargs : dict, optional
            Additional keyword arguments to pass to model.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If model is not an sklearn model instance or class.

        """
        if is_sklearn_instance(self.model) and not kwargs:
            self.model_ = self.model
        elif is_sklearn_instance(self.model) and kwargs:
            self.model_ = type(self.model)(**kwargs)
        elif is_sklearn_class(self.model):
            self.model_ = self.model(**kwargs)
        else:
            raise ValueError("Model must be an sklearn model instance or class.")

        return self

    def find_best(
        self,
        X: ArrayLike,
        y: ArrayLike,
        parameters: Union[Dict, List[Dict]],
        metric: Optional[Union[str, Callable, Sequence, Dict]] = None,
        method: Literal["grid", "random"] = "grid",
        **kwargs,
    ):
        """Tune model hyperparameters.

        Parameters
        ----------
        X : ArrayLike
            The feature matrix.
        y : ArrayLike
            The target vector.
        parameters : dict or list of dicts
            The hyperparameters to be tuned.
        metric : str, callable, sequence, dict, optional
            The metric to be used for model evaluation.
        method : Literal["grid", "random"], default="grid"
            The tuning method to be used.
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the tuning method.

        Returns
        -------
        self

        """
        # TODO: check the `metric` argument; allow using cyclops.evaluate.metrics

        # TODO: handle data splits
        # split_index = [-1] * len(X_train) + [0] * len(X_val)
        # X = np.concatenate((X_train, X_val), axis=0)
        # y = np.concatenate((y_train, y_val), axis=0)
        # pds = PredefinedSplit(test_fold=split_index)

        if method == "grid":
            clf = GridSearchCV(
                estimator=self.model_,
                param_grid=parameters,
                scoring=metric,
                cv=5,
                **kwargs,
            )
        elif method == "random":
            clf = RandomizedSearchCV(
                estimator=self.model_,
                param_distributions=parameters,
                scoring=metric,
                cv=5,
                **kwargs,
            )
        else:
            raise ValueError("Method must be either 'grid' or 'random'.")

        # TODO: allow passing group and fit_params to fit
        clf.fit(X, y)

        for key, value in clf.best_params_.items():
            LOGGER.info("Best %s: %f", key, value)

        self.model_ = (  # pylint: disable=attribute-defined-outside-init
            clf.best_estimator_
        )

        return self

    def fit_on_hf_dataset(
        self,
        dataset: Dataset,
        feature_columns: List[str],
        target_columns: List[str],
        preprocessor: Optional[ColumnTransformer] = None,
        batch_size: int = config.DEFAULT_MAX_BATCH_SIZE,
        **kwargs,
    ):
        """Fit the model on a Hugging Face dataset.

        Parameters
        ----------
        dataset : Dataset
             Hugging Face dataset containing features and labels.
        feature_columns : List[str]
            List of feature columns in the dataset.
        target_columns : List[str]
            List of target columns in the dataset.
        preprocessor : Optional[ColumnTransformer], optional
            Transformations to be applied to the data before fitting the model, \
                by default None
        batch_size : Optional[int], optional
            Batch size for batched fitting, used only for estimators with partial fit,
            by default config.DEFAULT_MAX_BATCH_SIZE

        Returns
        -------
        self : `SKModel`

        """

        def fit_model(examples):
            X_train = np.stack(
                [examples[feature] for feature in feature_columns], axis=1
            )
            if preprocessor is not None:
                try:
                    X_train = preprocessor.transform(X_train)
                except NotFittedError:
                    LOGGER.warning(
                        "Fitting preprocessor on batch of size %d", len(X_train)
                    )
                    X_train = preprocessor.fit_transform(X_train)

            y_train = np.stack([examples[target] for target in target_columns], axis=1)
            self.model_.partial_fit(
                X_train, y_train, classes=np.unique(y_train), **kwargs
            )
            return examples

        try:
            dataset.with_format("numpy", columns=feature_columns + target_columns).map(
                fit_model,
                batched=True,
                batch_size=batch_size,
            )
        except AttributeError as exc:
            LOGGER.info(
                "Model %s does not have a `partial_fit` method. \
                Calling `fit` directly.", self.model_.__class__.__name__
            )
            if is_out_of_core(dataset_size=dataset.dataset_size):
                raise ValueError("Dataset is too large to fit in memory.") from exc
            dataset = dataset.with_format("numpy", columns=feature_columns + target_columns)
            X_train = np.stack([dataset[feature] for feature in feature_columns], axis=1)
            if preprocessor is not None:
                try:
                    X_train = preprocessor.transform(X_train)
                except NotFittedError:
                    X_train = preprocessor.fit_transform(X_train)
            y_train = np.stack([dataset[target] for target in target_columns], axis=1)
            self.model_.fit(X_train, y_train)

        return self

    def predict_on_hf_dataset(
        self,
        dataset: Dataset,
        feature_columns: List[str],
        prediction_column_prefix: str = "predictions",
        model_name: Optional[str] = None,
        preprocessor: Optional[ColumnTransformer] = None,
        batched: bool = True,
        batch_size: int = config.DEFAULT_MAX_BATCH_SIZE,
        proba: bool = True,
        only_predictions: bool = False,
    ) -> Union[Dataset, np.ndarray]:
        """Predict the output of the model for the given Hugging Face dataset.

        Parameters
        ----------
        dataset : Dataset
            Hugging Face dataset containing features and possibly target labels.
        feature_columns : List[str]
            List of feature columns in the dataset.
        prediction_column_prefix : str, optional
            Name of the prediction column to be added to the dataset, \
                by default "predictions"
        model_name : Optional[str], optional
            Model name used as suffix to the prediction column, by default None
        preprocessor : Optional[ColumnTransformer], optional
            The transformation to be applied to the data before prediction, \
                by default None
        batched : bool, optional
            Whether to do prediction in mini-batches, by default True
        batch_size : int, optional
            Batch size for batched prediction, by default config.DEFAULT_MAX_BATCH_SIZE
        proba : bool, optional
            Whether to output the prediction probabilities rather than \
                the predicted classes, by default True
        only_predictions : bool, optional
            Whether to return only the predictions rather than \
            the dataset with predictions,
                by default False

        Returns
        -------
        Union[Dataset, np.ndarray]
            Dataset containing the predictions or the predictions array.

        """
        if model_name:
            pred_column = f"{prediction_column_prefix}.{model_name}"
        else:
            pred_column = f"{prediction_column_prefix}.{self.model_.__class__.__name__}"

        def get_predictions(examples: Dict[str, Union[List, np.ndarray]]) -> dict:
            X_eval = np.stack(
                [examples[feature] for feature in feature_columns], axis=1
            )
            if preprocessor is not None:
                try:
                    X_eval = preprocessor.transform(X_eval)
                except NotFittedError:
                    LOGGER.warning("Fitting preprocessor on evaluation data.")
                    X_eval = preprocessor.fit_transform(X_eval)

            if proba and hasattr(self.model_, "predict_proba"):
                examples[pred_column] = self.model_.predict_proba(X_eval)
            else:
                examples[pred_column] = self.model_.predict(X_eval)
            return examples

        ds_with_preds = dataset.with_format(
            "numpy", columns=feature_columns, output_all_columns=True
        ).map(
            get_predictions,
            batched=batched,
            batch_size=batch_size,
        )

        if only_predictions:
            return np.array(ds_with_preds[pred_column])

        return ds_with_preds

    def save_model(self, filepath: str, overwrite: bool = True, **kwargs):
        """Save model to file."""
        # filepath could be a directory or a file
        if os.path.isdir(filepath):
            filepath = join(filepath, self.model_.__class__.__name__, "model.pkl")

        if os.path.exists(filepath) and not overwrite:
            LOGGER.warning("The file already exists and will not be overwritten.")
            return

        save_pickle(self.model_, filepath, log=kwargs.get("log", True))

    def load_model(self, filepath: str, **kwargs):
        """Load a saved model.

        Parameters
        ----------
        filepath : str
            The path to the saved model.
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the load function.

        Returns
        -------
        self

        """
        try:
            model = load_pickle(filepath, log=kwargs.get("log", True))
            assert is_sklearn_instance(
                self.model_
            ), "The loaded model is not an instance of a scikit-learn estimator."
            self.model_ = model  # pylint: disable=attribute-defined-outside-init
        except FileNotFoundError:
            LOGGER.error("No saved model was found to load!")

        return self

    # dynamically offer every method and attribute of the sklearn model
    def __getattr__(self, name: str) -> Any:
        """Get attribute.

        Parameters
        ----------
        name : str
            attribute name.

        Returns
        -------
        The attribute value. If the attribute is a method that returns self,
        the wrapper instance is returned instead.

        """
        attr = getattr(self.__dict__["model_"], name)
        if callable(attr):

            @wraps(attr)
            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                if result is self.__dict__["model_"]:
                    self.__dict__["model_"] = result
                return result

            return wrapper
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute.

        If setting the model_ attribute, ensure that it is an sklearn model instance. If
        model has been instantiated and the attribute being set is in the model's
        __dict__, set the attribute in the model. Otherwise, set the attribute in the
        wrapper.

        """
        if "model_" in self.__dict__ and name == "model_":
            if not is_sklearn_instance(value):
                raise ValueError("Model must be an sklearn model instance.")
            self.__dict__["model_"] = value
        elif "model_" in self.__dict__ and name in self.__dict__["model_"].__dict__:
            setattr(self.__dict__["model_"], name, value)
        else:
            self.__dict__[name] = value

    def __delattr__(self, name: str) -> None:
        """Delete attribute."""
        delattr(self.__dict__["model_"], name)
