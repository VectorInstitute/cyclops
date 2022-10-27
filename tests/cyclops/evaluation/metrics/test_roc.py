"""Functions for testing ROC metrics."""
from typing import Any, List, Tuple

import numpy as np
import pytest
import scipy as sp
from metrics.helpers import _functional_test
from metrics.inputs import (
    NUM_CLASSES,
    NUM_LABELS,
    _binary_cases,
    _multiclass_cases,
    _multilabel_cases,
)
from sklearn.metrics import roc_curve as sk_roc_curve

from cyclops.evaluation.metrics.functional import roc_curve as cyclops_roc_curve
from cyclops.evaluation.metrics.utils import sigmoid


def _sk_binary_roc_curve(
    target: np.ndarray,
    preds: np.ndarray,
    pos_label: int = 1,
) -> List[Any]:
    """Compute ROC curve for binary case using sklearn."""
    if not ((0 < preds) & (preds < 1)).all():
        preds = sigmoid(preds)

    fpr, tpr, thresholds = sk_roc_curve(
        y_true=target, y_score=preds, pos_label=pos_label, drop_intermediate=False
    )
    thresholds[0] = 1.0

    return [np.nan_to_num(x, nan=0.0) for x in [fpr, tpr, thresholds]]


@pytest.mark.parametrize("inputs", _binary_cases[1:])
@pytest.mark.filterwarnings("ignore::UserWarning")  # np.nan_to_num warning
def test_binary_roc_curve(inputs):
    """Test binary ROC curve."""
    target, preds = inputs

    # test functional
    _functional_test(
        target,
        preds,
        cyclops_roc_curve,
        _sk_binary_roc_curve,
        {"task": "binary"},
    )


def _sk_multiclass_roc_curve(
    target: np.ndarray,
    preds: np.ndarray,
) -> List:
    """Compute ROC curve for multiclass case using sklearn."""
    # preds = np.moveaxis(preds.numpy(), 1, -1).reshape((-1, preds.shape[1]))
    # target = target.numpy().flatten()
    if not ((0 < preds) & (preds < 1)).all():
        preds = sp.special.softmax(preds, 1)

    fpr, tpr, thresholds = [], [], []
    for i in range(NUM_CLASSES):
        # target_temp = np.zeros_like(target)
        # target_temp[target == i] = 1
        res = sk_roc_curve(target, preds[:, i], pos_label=i, drop_intermediate=False)
        res[2][0] = 1.0

        fpr.append(res[0])
        tpr.append(res[1])
        thresholds.append(res[2])
    return [np.nan_to_num(x, nan=0.0) for x in [fpr, tpr, thresholds]]


@pytest.mark.parametrize("inputs", _multiclass_cases[1:])
@pytest.mark.filterwarnings("ignore::UserWarning")  # np.nan_to_num warning
def test_multiclass_roc_curve(inputs):
    """Test multiclass ROC curve."""
    target, preds = inputs

    # test functional
    _functional_test(
        target,
        preds,
        cyclops_roc_curve,
        _sk_multiclass_roc_curve,
        {"task": "multiclass", "num_classes": NUM_CLASSES},
    )


def _sk_multilabel_roc_curve(
    target: np.ndarray,
    preds: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Compute ROC curve for multilabel case using sklearn."""
    fpr, tpr, thresholds = [], [], []
    for i in range(NUM_LABELS):
        res = _sk_binary_roc_curve(target[:, i], preds[:, i])
        fpr.append(res[0])
        tpr.append(res[1])
        thresholds.append(res[2])

    return fpr, tpr, thresholds


@pytest.mark.parametrize("inputs", _multilabel_cases[1:])
@pytest.mark.filterwarnings("ignore::UserWarning")  # no positive samples
def test_multilabel_roc_curve(inputs):
    """Test multilabel ROC curve."""
    target, preds = inputs

    # test functional
    _functional_test(
        target,
        preds,
        cyclops_roc_curve,
        _sk_multilabel_roc_curve,
        {"task": "multilabel", "num_labels": NUM_LABELS},
    )
