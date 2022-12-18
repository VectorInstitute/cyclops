"""Utility functions for building models."""
import inspect
from difflib import get_close_matches
from typing import Dict, List

import numpy as np
import torch
from sklearn import metrics
from sklearn.base import BaseEstimator


def _get_class_members(
    module, include_only: List[str] = None, exclude: List[str] = None
) -> dict:
    """Get class members from module.

    Parameters
    ----------
    module : module
        Module to get class members from.
    include_only : list[str], optional
        List of class names to include.
    exclude : list[str], optional
        List of class names to exclude.

    Returns
    -------
    dict
        Dictionary of class members.

    """
    if include_only is None:
        include_only = []
    if exclude is None:
        exclude = []
    return {
        name: cls
        for name, cls in inspect.getmembers(module, inspect.isclass)
        if name in include_only or (not include_only and name not in exclude)
    }


####################
# Loss catalog     #
####################
_criterion_catalog: Dict[str, torch.nn.modules.loss._Loss] = _get_class_members(
    torch.nn.modules.loss, include_only=["BCELoss", "BCEWithLogitsLoss"]
)

#####################
# Optimizer catalog #
#####################
_optimizer_catalog: Dict[str, torch.optim.Optimizer] = _get_class_members(
    torch.optim, exclude=["Optimizer"]
)

#####################
# Scheduler catalog #
#####################
_lr_scheduler_catalog: Dict[
    str, torch.optim.lr_scheduler._LRScheduler
] = _get_class_members(
    torch.optim.lr_scheduler,
    exclude=["_LRScheduler", "Optimizer", "Counter", "ChainedScheduler"],
)

######################
# Activation catalog #
######################
_activation_catalog = _get_class_members(
    torch.nn.modules.activation,
    include_only=[
        "Hardtanh",
        "Sigmoid",
        "ReLU6",
        "Tanh",
        "Tanhshrink",
        "Hardshrink",
        "LeakyReLU",
        "Softshrink",
        "ReLU",
        "PReLU",
        "Softplus",
        "ELU",
        "SELU",
        "SiLU",
    ],
)
_activation_catalog.update(Id=torch.nn.Identity)


def get_module(module_type: str, module_name: str):
    """Get module.

    Parameters
    ----------
    module_type : str
        Module type.
    module_name : str
        Module name.

    Returns
    -------
    Any
        Module.

    Raises
    ------
    ValueError
        If module type is not supported.

    """
    if module_type == "criterion":
        catalog = _criterion_catalog
    elif module_type == "optimizer":
        catalog = _optimizer_catalog
    elif module_type == "lr_scheduler":
        catalog = _lr_scheduler_catalog
    elif module_type == "activation":
        catalog = _activation_catalog
    else:
        raise ValueError(f"Module type {module_type} is not supported.")

    module = catalog.get(module_name, None)
    if module is None:
        similar_keys_list: List[str] = get_close_matches(
            module_name, catalog.keys(), n=5
        )
        similar_keys: str = ", ".join(similar_keys_list)
        similar_keys = (
            f" Did you mean one of: {similar_keys}?"
            if similar_keys
            else "It may not be in the catalog."
        )
        raise ValueError(f"Module {module_name} not found.{similar_keys}")

    return module


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
        and hasattr(model, "predict")
        and hasattr(model, "predict_proba")
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


def is_pytorch_instance(module: object) -> bool:
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
    return isinstance(module, torch.nn.Module)


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


def get_device() -> torch.device:
    """Get device for PyTorch models.

    Returns
    -------
    torch.device
        cpu or cuda

    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class LossMeter:
    """Loss meter for PyTorch models."""

    def __init__(self, name: str) -> None:
        """Initialize loss meter.

        Parameters
        ----------
        name : str
            Loss name. Used for logging.

        """
        self.name = name
        self.losses: List[float] = []

    def reset(self) -> None:
        """Reset the list of losses."""
        self.losses.clear()

    def add(self, val: float) -> None:
        """Add to the list of losses.

        Parameters
        ----------
        val : float
            Loss value.

        """
        self.losses.append(val)

    def mean(self) -> float:
        """Get the mean of the loss values in the list.

        Returns
        -------
        float
            Mean values.

        """
        if not self.losses:
            return 0
        return np.mean(self.losses)

    def pop(self) -> float:
        """Get the last element of the list.

        Returns
        -------
        float
            Loss value.

        """
        return self.losses[-1]

    def sum(self) -> float:
        """Get the summation of all loss values.

        Returns
        -------
        float
            Sum value.

        """
        return sum(self.losses)


def metrics_binary(  # pylint: disable=too-many-locals, invalid-name
    y_test_labels: np.ndarray,
    y_pred_values: np.ndarray,
    y_pred_labels: np.ndarray,
    verbose: bool,
) -> dict:
    """Compute metrics for binary classification.

    Parameters
    ----------
    y_pred_values : np.ndarray
        predicted values/probs
    y_pred_labels : np.ndarray
        predicted labels

    verbose : bool
        print the metric values

    Returns
    -------
    dict
        dict of metric names and values

    """
    cf = metrics.confusion_matrix(y_test_labels, y_pred_labels)
    if verbose:
        print("confusion matrix:")
        print(cf)
    cf = cf.astype(np.float32)
    tn, fp, fn, tp = cf.ravel()
    acc = (tn + tp) / np.sum(cf)
    prec0 = tn / (tn + fn)
    prec1 = tp / (tp + fp)
    rec0 = tn / (tn + fp)
    rec1 = tp / (tp + fn)

    prec = (prec0 + prec1) / 2
    rec = (rec0 + rec1) / 2

    auroc = metrics.roc_auc_score(y_test_labels, y_pred_values)

    (precisions, recalls, _) = metrics.precision_recall_curve(
        y_test_labels, y_pred_values
    )
    auprc = metrics.auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])

    if verbose:
        print(f"accuracy = {acc}")
        print(f"precision class 0 = {prec0}")
        print(f"precision class 1 = {prec1}")
        print(f"recall class 0 = {rec0}")
        print(f"recall class 1 = {rec1}")
        print(f"AUC of ROC = {auroc}")
        print(f"AUC of PRC = {auprc}")
        print(f"min(+P, Se) = {minpse}")

    return {
        "confusion": cf,
        "accuracy": acc,
        "precision_0": prec0,
        "precision_1": prec1,
        "precision": prec,
        "recall_0": rec0,
        "recall_1": rec1,
        "recall": rec,
        "aucroc": auroc,
        "auprc": auprc,
        "minpse": minpse,
    }
