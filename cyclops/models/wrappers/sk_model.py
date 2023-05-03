"""Scikit-learn model wrapper."""

import logging
import os
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
from datasets import Dataset, DatasetDict
from datasets.combine import concatenate_datasets
from numpy.typing import ArrayLike
from scipy.sparse import issparse
from sklearn.base import BaseEstimator as SKBaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, PredefinedSplit, RandomizedSearchCV

from cyclops.data.utils import is_out_of_core
from cyclops.models.utils import get_split, is_sklearn_class, is_sklearn_instance
from cyclops.models.wrappers.utils import DatasetColumn
from cyclops.utils.file import join, load_pickle, save_pickle
from cyclops.utils.log import setup_logging

LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)

# pylint: disable=fixme, function-redefined, dangerous-default-value


class SKModel:
    """Scikit-learn model wrapper.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        Scikit-learn model instance or class.
    model_params: dict
        Scikit-learn estimator parameters
    batch_size: int, optional
        The batch size used when using Hugging Face Dataset, \
            by default 64
    **kwargs : dict, optional
        Additional keyword arguments to pass to model.

    Notes
    -----
    This wrapper does not inherit from models.wrappers.base.ModelWrapper
    because it uses the decorator pattern to expose the sklearn API, which
    is what the base wrapper is meant to abstract away.

    """

    def __init__(  # pylint: disable=dangerous-default-value
        self,
        model: SKBaseEstimator,
        model_params: dict,
        batch_size: int = 64,
        **kwargs,
    ) -> None:
        """Initialize wrapper."""
        self.model = model  # possibly uninstantiated class
        self.initialize_model(**model_params)
        self.batch_size = batch_size
        vars(self).update(kwargs)

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

    def find_best(  # pylint: disable=too-many-branches
        self,
        parameters: Union[Dict, List[Dict]],
        X: Union[ArrayLike, Dataset, DatasetDict],
        y: Optional[ArrayLike] = None,
        feature_columns: Optional[Union[str, List[str]]] = None,
        target_columns: Optional[Union[str, List[str]]] = None,
        transforms: Optional[Union[ColumnTransformer, Callable]] = None,
        metric: Optional[Union[str, Callable, Sequence, Dict]] = None,
        method: Literal["grid", "random"] = "grid",
        splits_mapping: dict = {"train": "train", "validation": "validation"},
        **kwargs,
    ):
        """Search on hyper parameters.

        Parameters
        ----------
        parameters : dict or list of dicts
            The hyperparameters to be tuned.
        X : Union[Dataset, ArrayLike]
            The data features or a Hugging Face dataset containing features and labels.
        y : Optional[ArrayLike], optional
            The labels of the data. This is required when the input dataset is not \
                a huggingface dataset and only contains features, by default None
        feature_columns : Optional[Union[str, List[str]]], optional
            List of feature columns in the dataset. This is required when the input is \
                a Hugging Face Dataset, by default None
        target_columns : Optional[Union[str, List[str]]], optional
            List of target columns in the dataset. This is required when the input is \
                a Hugging Face Dataset, by default None
        transforms : Optional[Union[ColumnTransformer, Callable]], optional
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
                by default {"train": "train", "validation": "validation"}
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the search method.

        Returns
        -------
        self : `SKModel`

        Raises
        ------
        ValueError
            If search method is not supported.
        ValueError
            If `X` is a Hugging Face Dataset and the feature column(s) is not provided.
        ValueError
            If `X` is a Hugging Face Dataset and the target column(s) is not provided.
        RuntimeErrot
            If dataset size is larger than the available memory.

        """
        # TODO: check the `metric` argument; allow using cyclops.evaluate.metrics
        # TODO: allow passing group

        if method == "grid":
            clf = GridSearchCV(
                estimator=self.model_,
                param_grid=parameters,
                scoring=metric,
                **kwargs,
            )
        elif method == "random":
            clf = RandomizedSearchCV(
                estimator=self.model_,
                param_distributions=parameters,
                scoring=metric,
                **kwargs,
            )
        else:
            raise ValueError("Method must be either 'grid' or 'random'.")

        if isinstance(X, (Dataset, DatasetDict)):
            if feature_columns is None:
                raise ValueError(
                    "Missing target columns 'target_columns'. Please provide \
                    the name of feature columns when using a \
                    Hugging Face dataset as the input."
                )
            if isinstance(feature_columns, str):
                feature_columns = [feature_columns]

            if target_columns is None:
                raise ValueError(
                    "Missing target columns 'target_columns'. Please provide \
                    the name of target columns when using a \
                    Hugging Face dataset as the input."
                )
            if isinstance(target_columns, str):
                target_columns = [target_columns]

            if isinstance(X, DatasetDict):
                train_split = get_split(X, "train", splits_mapping)
                try:
                    val_split = get_split(X, "validation", splits_mapping)
                except ValueError:
                    LOGGER.info("No validation split was found.")
                    val_split = None

                if val_split is None:
                    return self.find_best(
                        parameters,
                        X[train_split],
                        feature_columns=feature_columns,
                        target_columns=target_columns,
                        transforms=transforms,
                        metric=metric,
                        method=method,
                    )

                if X[train_split].dataset_size is not None and is_out_of_core(
                    X[train_split].dataset_size
                ):
                    raise RuntimeError("Dataset size cannot fit into memory!")

                format_kwargs = {}
                is_callable_transform = callable(transforms)
                if is_callable_transform:
                    format_kwargs["transform"] = transforms

                with X[train_split].formatted_as(
                    "custom" if is_callable_transform else "numpy",
                    columns=feature_columns + target_columns,
                    **format_kwargs,
                ):
                    X_train = np.stack(
                        [X[train_split][feature] for feature in feature_columns], axis=1
                    ).squeeze()

                    if transforms is not None and not is_callable_transform:
                        try:
                            X_train = transforms.transform(X_train)
                        except NotFittedError:
                            X_train = transforms.fit_transform(X_train)

                    y_train = np.stack(
                        [X[train_split][target] for target in target_columns], axis=1
                    ).squeeze()

                    if issparse(X_train):
                        X_train = X_train.toarray()

                with X[val_split].formatted_as(
                    "custom" if is_callable_transform else "numpy",
                    columns=feature_columns + target_columns,
                    **format_kwargs,
                ):
                    X_val = np.stack(
                        [X[val_split][feature] for feature in feature_columns], axis=1
                    ).squeeze()

                    if transforms is not None and not is_callable_transform:
                        try:
                            X_val = transforms.transform(X_val)
                        except NotFittedError:
                            X_val = transforms.fit_transform(X_val)

                    y_val = np.stack(
                        [X[val_split][target] for target in target_columns], axis=1
                    ).squeeze()

                    if issparse(X_val):
                        X_val = X_val.toarray()

                split_index = [-1] * len(X_train) + [0] * len(X_val)
                X = np.concatenate((X_train, X_val), axis=0)
                y = np.concatenate((y_train, y_val), axis=0)

                clf.cv = PredefinedSplit(test_fold=split_index)
                clf.fit(X, y)

            elif isinstance(X, Dataset):
                if X.dataset_size is not None and is_out_of_core(X.dataset_size):
                    raise RuntimeError("Dataset size cannot fit into memory!")

                format_kwargs = {}
                is_callable_transform = callable(transforms)
                if is_callable_transform:
                    format_kwargs["transform"] = transforms

                with X.formatted_as(
                    "custom" if is_callable_transform else "numpy",
                    columns=feature_columns + target_columns,
                    **format_kwargs,
                ):
                    X_train = np.stack(
                        [X[feature] for feature in feature_columns], axis=1
                    ).squeeze()

                    if transforms is not None and not is_callable_transform:
                        try:
                            X_train = transforms.transform(X_train)
                        except NotFittedError:
                            X_train = transforms.fit_transform(X_train)

                    y_train = np.stack(
                        [X[target] for target in target_columns], axis=1
                    ).squeeze()

                    if issparse(X_train):
                        X_train = X_train.toarray()

                    clf.fit(X_train, y_train)

        else:
            if y is None:
                LOGGER.warning(
                    "Missing data labels 'y'. Please provide the labels \
                    for supervised training when not using a \
                    Hugging Face dataset as the input."
                )
            clf.fit(X, y)

        for key, value in clf.best_params_.items():
            LOGGER.info("Best %s: %f", key, value)

        self.model_ = (  # pylint: disable=attribute-defined-outside-init
            clf.best_estimator_
        )

        return self

    def partial_fit(
        self,
        X: Union[ArrayLike, Dataset, DatasetDict],
        y: Optional[ArrayLike] = None,
        feature_columns: Optional[Union[str, List[str]]] = None,
        target_columns: Optional[Union[str, List[str]]] = None,
        transforms: Optional[Union[ColumnTransformer, Callable]] = None,
        classes: Optional[np.ndarray] = None,
        splits_mapping: dict = {"train": "train"},
        **kwargs,
    ):
        """Fit the model to the data incrementally.

        Parameters
        ----------
        X : Union[Dataset, ArrayLike]
            The data features or Hugging Face dataset containing features and labels.
        y : Optional[ArrayLike], optional
            The labels of the data. This is required when the input dataset is not \
                a huggingface dataset and only contains features, by default None
        feature_columns : Optional[Union[str, List[str]]], optional
            List of feature columns in the dataset. This is required when \
                the input is a Hugging Face Dataset, by default None
        target_columns : Optional[Union[str, List[str]]], optional
            List of target columns in the dataset. This is required when \
                the input is a Hugging Face Dataset, by default None
        transforms : Optional[Union[ColumnTransformer, Callable]], optional
            The transformation to be applied to the data before prediction, \
                This is used when the input is a Hugging Face Dataset, \
                by default None
        classes : Optional[np.ndarray], optional
            All the possible classes in the dataset. This is required when \
                the input dataset is not a huggingface dataset and \
                only contains features, by default None
        splits_mapping: Optional[dict], optional
            Mapping from 'train', 'validation' and 'test' to dataset splits names, \
                used when input is a dataset dictionary, by default {"train": "train"}

        Returns
        -------
        self: `SKModel`

        Raises
        ------
        AttributeError
            Model does not have partial_fit method.
        ValueError
            If `X` is a Hugging Face Dataset and the feature column(s) is not provided.
        ValueError
            If `X` is a Hugging Face Dataset and the target column(s) is not provided.

        """
        if not hasattr(self.model_, "partial_fit"):
            raise AttributeError(
                f"Model {self.model_.__class__.__name__}"
                "does not have a `partial_fit` method.",
            )
        # Train data is a Hugging Face Dataset Dictionary.
        if isinstance(X, DatasetDict):
            train_split = get_split(X, "train", splits_mapping=splits_mapping)
            return self.fit(
                X[train_split],
                feature_columns=feature_columns,
                target_columns=target_columns,
                transforms=transforms,
                classes=classes,
            )

        # Train data is a Hugging Face Dataset.
        if isinstance(X, Dataset):
            if feature_columns is None:
                raise ValueError(
                    "Missing feature columns 'feature_columns'. Please provide \
                    the name of feature columns when using a \
                    Hugging Face dataset as the input."
                )
            if isinstance(feature_columns, str):
                feature_columns = [feature_columns]

            if target_columns is None:
                raise ValueError(
                    "Missing target columns 'target_columns'. Please provide \
                    the name of target columns when using a \
                    Hugging Face dataset as the input."
                )
            if isinstance(target_columns, str):
                target_columns = [target_columns]

            format_kwargs = {}
            is_callable_transform = callable(transforms)
            if is_callable_transform:
                format_kwargs["transform"] = transforms

            def fit_model(examples):
                X_train = np.stack(
                    [examples[feature] for feature in feature_columns], axis=1
                ).squeeze()

                if transforms is not None:
                    if not is_callable_transform:
                        try:
                            X_train = transforms.transform(X_train)
                        except NotFittedError:
                            LOGGER.warning(
                                "Fitting preprocessor on batch of size %d", len(X_train)
                            )
                            X_train = transforms.fit_transform(X_train)

                y_train = np.stack(
                    [examples[target] for target in target_columns], axis=1
                ).squeeze()
                self.model_.partial_fit(
                    X_train, y_train, classes=np.unique(y_train), **kwargs
                )
                return examples

            with X.formatted_as(
                "custom" if is_callable_transform else "numpy",
                columns=feature_columns + target_columns,
                **format_kwargs,
            ):
                X.map(
                    fit_model,
                    batched=True,
                    batch_size=self.batch_size,
                )
        # Train data is not a Hugging Face Dataset.
        else:
            if y is None:
                LOGGER.warning(
                    "Missing data labels 'y'. Please provide the labels \
                    for supervised training when not using a \
                    Hugging Face dataset as the input."
                )
            if classes is None:
                LOGGER.warning(
                    "Missing unique class labels. Please provide a list of classes \
                    when using a numpy array or pandas dataframe as the input."
                )

            self.model_.partial_fit(X, y, classes=classes, **kwargs)

        return self

    def fit(
        self,
        X: Union[ArrayLike, Dataset, DatasetDict],
        y: Optional[ArrayLike] = None,
        feature_columns: Optional[Union[str, List[str]]] = None,
        target_columns: Optional[Union[str, List[str]]] = None,
        transforms: Optional[Union[ColumnTransformer, Callable]] = None,
        splits_mapping: dict = {"train": "train"},
        **fit_params,
    ):
        """Fit the model.

        Parameters
        ----------
        X : Union[Dataset, ArrayLike]
            The data features or a Hugging Face dataset containing features and labels.
        y : Optional[ArrayLike], optional
            The labels of the data. This is required when the input dataset is not \
                a huggingface dataset and only contains features, by default None
        feature_columns : Optional[Union[str, List[str]]], optional
            List of feature columns in the dataset. This is required when the input is \
                a Hugging Face Dataset, by default None
        target_columns : Optional[Union[str, List[str]]], optional
            List of target columns in the dataset. This is required when \
                the input is a Hugging Face Dataset, by default None
        transforms : Optional[Union[ColumnTransformer, Callable]], optional
            The transformation to be applied to the data before prediction,
                This is used when the input is a Hugging Face Dataset, \
                by default None
        splits_mapping: Optional[dict], optional
            Mapping from 'train', 'validation' and 'test' to dataset splits names, \
                used when input is a dataset dictionary, by default {"train": "train"}

        Returns
        -------
        self : `SKModel`

        Raises
        ------
        ValueError
            If `X` is a Hugging Face Dataset and the feature column(s) is not provided.
        ValueError
            If `X` is a Hugging Face Dataset and the target column(s) is not provided.

        """
        # Train data is a Hugging Face Dataset Dictionary.
        if isinstance(X, DatasetDict):
            train_split = get_split(X, "train", splits_mapping=splits_mapping)
            return self.fit(
                X[train_split],
                feature_columns=feature_columns,
                target_columns=target_columns,
                transforms=transforms,
            )

        # Train data is a Hugging Face Dataset.
        if isinstance(X, Dataset):
            if feature_columns is None:
                raise ValueError(
                    "Missing feature columns 'feature_columns'. Please provide \
                    the name of feature columns when using a \
                    Hugging Face dataset as the input."
                )
            if isinstance(feature_columns, str):
                feature_columns = [feature_columns]

            if target_columns is None:
                raise ValueError(
                    "Missing target columns 'target_columns'. Please provide \
                    the name of target columns when using a \
                    Hugging Face dataset as the input."
                )
            if isinstance(target_columns, str):
                target_columns = [target_columns]

            if X.dataset_size is not None and is_out_of_core(X.dataset_size):
                LOGGER.warning(
                    "Dataset size cannot fit into memory. Will call partial fit."
                )
                return self.partial_fit(
                    X,
                    feature_columns=feature_columns,
                    target_columns=target_columns,
                    transforms=transforms,
                )

            format_kwargs = {}
            is_callable_transform = callable(transforms)
            if is_callable_transform:
                format_kwargs["transform"] = transforms

            with X.formatted_as(
                "custom" if is_callable_transform else "numpy",
                columns=feature_columns + target_columns,
                **format_kwargs,
            ):
                X_train = np.stack(
                    [X[feature] for feature in feature_columns], axis=1
                ).squeeze()

                if transforms is not None and not is_callable_transform:
                    try:
                        X_train = transforms.transform(X_train)
                    except NotFittedError:
                        X_train = transforms.fit_transform(X_train)

                y_train = np.stack(
                    [X[target] for target in target_columns], axis=1
                ).squeeze()

                if issparse(X_train):
                    X_train = X_train.toarray()
                self.fit(X_train, y_train, **fit_params)
        # Train data is not a Hugging Face Dataset.
        else:
            if y is None:
                LOGGER.warning(
                    "Missing data labels 'y'. Please provide the labels \
                    for supervised training when not using a \
                    Hugging Face dataset as the input."
                )

            self.model_ = (  # pylint: disable=attribute-defined-outside-init
                self.model_.fit(X, y, **fit_params)
            )
        return self

    def predict_proba(
        self,
        X: Union[ArrayLike, Dataset, DatasetDict],
        feature_columns: Optional[Union[str, List[str]]] = None,
        prediction_column_prefix: str = "predictions",
        model_name: Optional[str] = None,
        transforms: Optional[Union[ColumnTransformer, Callable]] = None,
        only_predictions: bool = False,
        splits_mapping: dict = {"test": "test"},
    ) -> Union[Dataset, DatasetColumn, np.ndarray]:
        """Predict the probability output of the model.

        Parameters
        ----------
        X : Dataset
            The data features or Hugging Face dataset containing features and labels.
        feature_columns : Optional[Union[str, List[str]]], optional
            List of feature columns in the dataset. This is required when the input is \
                a Hugging Face Dataset, by default None
        prediction_column_prefix : str, optional
            Name of the prediction column to be added to the dataset, This is used \
                when the input is a Hugging Face Dataset, by default "predictions"
        model_name : Optional[str], optional
            Model name used as suffix to the prediction column, This is used \
                when the input is a Hugging Face Dataset, by default None
        transforms : Optional[Callable], optional
            The transformation to be applied to the data before prediction, \
                This is used when the input is a Hugging Face Dataset, \
                by default None
        only_predictions : bool, optional
            Whether to return only the predictions rather than the dataset \
                with predictions when the input is a Hugging Face Dataset, \
                by default False
        splits_mapping: Optional[dict], optional
            Mapping from 'train', 'validation' and 'test' to dataset splits names, \
                used when input is a dataset dictionary, by default {"test": "test"}

        Returns
        -------
        Union[Dataset, DatasetColumn, np.ndarray]
            Dataset containing the predictions or the predictions array.

        Raises
        ------
        AttributeError
            If model does not have predict_proba method.
        ValueError
            If `X` is a Hugging Face Dataset and the feature column(s) is not provided.

        """
        if not hasattr(self.model_, "predict_proba"):
            raise AttributeError(
                f"Model {self.model_.__class__.__name__}"
                "does not have a `predict_proba` method.",
            )
        # Data is a Hugging Face Dataset Dictionary.
        if isinstance(X, DatasetDict):
            test_split = get_split(X, "test", splits_mapping=splits_mapping)
            return self.predict_proba(
                X[test_split],
                feature_columns=feature_columns,
                prediction_column_prefix=prediction_column_prefix,
                model_name=model_name,
                transforms=transforms,
                only_predictions=only_predictions,
            )
        # Data is a Hugging Face Dataset.
        if isinstance(X, Dataset):
            if feature_columns is None:
                raise ValueError(
                    "Missing feature columns 'feature_columns'. Please provide \
                    the name of feature columns when using a \
                    Hugging Face dataset as the input."
                )
            if isinstance(feature_columns, str):
                feature_columns = [feature_columns]

            if model_name:
                pred_column = f"{prediction_column_prefix}.{model_name}"
            else:
                pred_column = (
                    f"{prediction_column_prefix}.{self.model_.__class__.__name__}"
                )

            format_kwargs = {}
            is_callable_transform = callable(transforms)
            if is_callable_transform:
                format_kwargs["transform"] = transforms

            def get_predictions(examples: Dict[str, Union[List, np.ndarray]]) -> dict:
                X_eval = np.stack(
                    [examples[feature] for feature in feature_columns], axis=1
                )
                if transforms is not None and not is_callable_transform:
                    try:
                        X_eval = transforms.transform(X_eval)
                    except NotFittedError:
                        LOGGER.warning("Fitting preprocessor on evaluation data.")
                        X_eval = transforms.fit_transform(X_eval)

                predictions = self.predict_proba(X_eval)

                return {pred_column: predictions}

            with X.formatted_as(
                "custom" if is_callable_transform else "numpy",
                columns=feature_columns,
                output_all_columns=True,
                **format_kwargs,
            ):
                pred_ds = X.map(
                    get_predictions,
                    batched=True,
                    batch_size=self.batch_size,
                    remove_columns=X.column_names,
                )
                if only_predictions:
                    return DatasetColumn(pred_ds.with_format("numpy"), pred_column)

                X = concatenate_datasets([X, pred_ds], axis=1)

            return X
        # Data is not a Hugging Face Dataset.
        return self.model_.predict_proba(X)

    def predict(
        self,
        X: Union[ArrayLike, Dataset],
        feature_columns: Optional[Union[str, List[str]]] = None,
        prediction_column_prefix: str = "predictions",
        model_name: Optional[str] = None,
        transforms: Optional[Union[ColumnTransformer, Callable]] = None,
        only_predictions: bool = False,
        splits_mapping: dict = {"test": "test"},
    ) -> Union[Dataset, DatasetColumn, np.ndarray]:
        """Predict the output of the model.

        Parameters
        ----------
        X : Dataset
            The data features or Hugging Face dataset containing features and labels.
        feature_columns : Optional[Union[str, List[str]]], optional
            List of feature columns in the dataset. This is required when the input is \
                a Hugging Face Dataset, by default None
        prediction_column_prefix : str, optional
            Name of the prediction column to be added to the dataset, This is used \
                when the input is a Hugging Face Dataset, by default "predictions"
        model_name : Optional[str], optional
            Model name used as suffix to the prediction column, This is used \
                when the input is a Hugging Face Dataset, by default None
        transforms : Optional[Callable], optional
            The transformation to be applied to the data before prediction, \
                This is used when the input is a Hugging Face Dataset, \
                by default None
        only_predictions : bool, optional
            Whether to return only the predictions rather than the dataset \
                with predictions when the input is a Hugging Face Dataset, \
                by default False
        splits_mapping: Optional[dict], optional
            Mapping from 'train', 'validation' and 'test' to dataset splits names, \
                used when input is a dataset dictionary, by default {"test": "test"}

        Returns
        -------
        Union[Dataset, DatasetColumn, np.ndarray]
            Dataset containing the predictions or the predictions array.

        Raises
        ------
        ValueError
            If `X` is a Hugging Face Dataset and the feature column(s) is not provided.

        """
        # Data is a Hugging Face Dataset Dictionary.
        if isinstance(X, DatasetDict):
            test_split = get_split(X, "test", splits_mapping=splits_mapping)
            return self.predict(
                X[test_split],
                feature_columns=feature_columns,
                prediction_column_prefix=prediction_column_prefix,
                model_name=model_name,
                transforms=transforms,
                only_predictions=only_predictions,
            )
        # Data is a Hugging Face Dataset.
        if isinstance(X, Dataset):
            if feature_columns is None:
                raise ValueError(
                    "Missing feature columns 'feature_columns'. Please provide \
                    the name of feature columns when using a \
                    Hugging Face dataset as the input."
                )
            if isinstance(feature_columns, str):
                feature_columns = [feature_columns]

            if model_name:
                pred_column = f"{prediction_column_prefix}.{model_name}"
            else:
                pred_column = (
                    f"{prediction_column_prefix}.{self.model_.__class__.__name__}"
                )

            format_kwargs = {}
            is_callable_transform = callable(transforms)
            if is_callable_transform:
                format_kwargs["transform"] = transforms

            def get_predictions(examples: Dict[str, Union[List, np.ndarray]]) -> dict:
                X_eval = np.stack(
                    [examples[feature] for feature in feature_columns], axis=1
                )
                if transforms is not None and not is_callable_transform:
                    try:
                        X_eval = transforms.transform(X_eval)
                    except NotFittedError:
                        LOGGER.warning("Fitting preprocessor on evaluation data.")
                        X_eval = transforms.fit_transform(X_eval)

                predictions = self.predict(X_eval)
                return {pred_column: predictions}

            with X.formatted_as(
                "custom" if is_callable_transform else "numpy",
                columns=feature_columns,
                output_all_columns=True,
                **format_kwargs,
            ):
                pred_ds = X.map(
                    get_predictions,
                    batched=True,
                    batch_size=self.batch_size,
                    remove_columns=X.column_names,
                )
                if only_predictions:
                    return DatasetColumn(pred_ds.with_format("numpy"), pred_column)

                X = concatenate_datasets([X, pred_ds], axis=1)

            return X
        # Data is not a Hugging Face Dataset.
        return self.model_.predict(X)

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
