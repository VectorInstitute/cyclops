"""Utility functions for building models."""

from typing import List

import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.optim import SGD, Adagrad, Adam
from torch.optim.lr_scheduler import ExponentialLR, StepLR

ACTIVATIONS = {
    "none": None,
    "hardtanh": nn.Hardtanh(),
    "sigmoid": nn.Sigmoid(),
    "relu6": nn.ReLU6(),
    "tanh": nn.Tanh(),
    "tanhshrink": nn.Tanhshrink(),
    "hardshrink": nn.Hardshrink(),
    "leakyreluleakyrelu": nn.LeakyReLU(),
    "softshrink": nn.Softshrink(),
    "relu": nn.ReLU(),
    "prelu": nn.PReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "selu": nn.SELU(),
    "silu": nn.SiLU(),
}

CRITERIONS = {
    "bce": nn.BCELoss(),
    "bcelogits": nn.BCEWithLogitsLoss,
}

OPTIMIZERS = {
    "adam": Adam,
    "adagrad": Adagrad,
    "sgd": SGD,
}

SCHEDULERS = {"step": StepLR, "expo": ExponentialLR}

METRICS = {
    "accuracy": accuracy_score,
    "precision": average_precision_score,
    "roc_auc": roc_auc_score,
    "f1": f1_score,
    "recall": recall_score,
    "confusion": confusion_matrix,
}


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
            loss name, train or val

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
            loss value

        """
        self.losses.append(val)

    def mean(self) -> float:
        """Get the mean of the loss values in the list.

        Returns
        -------
        float
            mean values

        """
        if not self.losses:
            return 0
        return np.mean(self.losses)

    def pop(self) -> float:
        """Get the last element of the list.

        Returns
        -------
        float
            loss value

        """
        return self.losses[-1]

    def sum(self) -> float:
        """Get the summation of all loss values.

        Returns
        -------
        float
            sum value

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
