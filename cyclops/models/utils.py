"""Utility functions for building models."""

import inspect
from typing import TYPE_CHECKING, Literal, Optional

from datasets import DatasetDict
from sklearn.base import BaseEstimator

from cyclops.utils.optional import import_optional_module


if TYPE_CHECKING:
    import torch
else:
    torch = import_optional_module("torch", error="warn")


def _has_sklearn_api(model: object) -> bool:
    """Check if model has a sklearn signature.

    Parameters
    ----------
    model : object
        The model to check.

    Returns
    -------
    bool
        True if ``model`` has a sklearn signature.

    """
    return (
        hasattr(model, "fit")
        or hasattr(model, "partial_fit")
        and hasattr(model, "predict")
        or hasattr(model, "predict_proba")
    )


def is_sklearn_instance(model: object) -> bool:
    """Check if model is an instance of a sklearn model.

    Parameters
    ----------
    model : object
        The model to check.

    Returns
    -------
    bool
        True if ``model`` is an instance of a sklearn model.

    """
    return isinstance(model, BaseEstimator) and _has_sklearn_api(model)


def is_sklearn_class(model: object) -> bool:
    """Check if model is a sklearn class.

    Parameters
    ----------
    model : object
        The model to check.

    Returns
    -------
    bool
        True if ``model`` is a sklearn class

    """
    return (
        inspect.isclass(model)
        and issubclass(model, BaseEstimator)
        and _has_sklearn_api(model)
    )


def is_sklearn_model(model: object) -> bool:
    """Check if model is a sklearn model.

    Parameters
    ----------
    model : object
        The model to check.

    Returns
    -------
    bool
        True if ``model`` is an sklearn model.

    """
    return is_sklearn_class(model) or is_sklearn_instance(model)


def is_pytorch_instance(model: object) -> bool:
    """Check if object is an instance of a PyTorch module.

    Parameters
    ----------
    model : object
        The model to check.

    Returns
    -------
    bool
        True if ``model`` is an instance of a PyTorch model.

    """
    return isinstance(model, torch.nn.Module)


def is_pytorch_class(model: object) -> bool:
    """Check if model is a PyTorch class.

    Parameters
    ----------
    model : object
        The model to check.

    Returns
    -------
    bool
        True if ``model`` is a PyTorch class

    """
    return inspect.isclass(model) and issubclass(model, torch.nn.Module)


def is_pytorch_model(model: object) -> bool:
    """Check if model is a PyTorch model.

    Parameters
    ----------
    model : object
        The model to check.

    Returns
    -------
    bool
        True if ``model`` is a PyTorch model

    """
    return is_pytorch_class(model) or is_pytorch_instance(model)


def get_split(
    dataset: DatasetDict,
    wanted_split: Literal["train", "validation", "test"],
    splits_mapping: Optional[dict] = None,
) -> str:
    """Get a dataset split name based on the purpose.

    Parameters
    ----------
    dataset : DatasetDict
        Dataset to choose a split from.
    wanted_split : Literal["train", "validation", "test"]
        Name of the split to look for
    splits_mapping: Optional[dict], optional
        Mapping from 'train', 'validation' and 'test' to dataset splits names \
            by default None


    Returns
    -------
    str
        Name of the chosen split.

    Raises
    ------
    ValueError
        If no split can be found.

    """
    available_splits = list(dataset.keys())

    if (
        wanted_split in list(splits_mapping.keys())
        and splits_mapping[wanted_split] in available_splits
    ):
        return splits_mapping[wanted_split]

    preferred_split_order = {
        "train": [
            "train",
            "training",
        ],
        "validation": [
            "validation",
            "val",
            "valid",
            "validate",
        ],
        "test": [
            "test",
            "testing",
            "eval",
            "evaluation",
        ],
    }

    for split in preferred_split_order[wanted_split]:
        if split in available_splits:
            return split

    raise ValueError(
        "The dataset split is not found! Pass the correct value to \
            the `splits_mapping` kwarg.",
    )
