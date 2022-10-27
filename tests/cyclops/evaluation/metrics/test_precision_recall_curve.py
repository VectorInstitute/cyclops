"""Functions for testing precision-recall curve metrics."""
from typing import List, Tuple

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
from sklearn.metrics import precision_recall_curve as sk_precision_recall_curve

from cyclops.evaluation.metrics.functional import (
    precision_recall_curve as cyclops_precision_recall_curve,
)
from cyclops.evaluation.metrics.utils import sigmoid


def _sk_binary_precision_recall_curve(
    target: np.ndarray,
    preds: np.ndarray,
    pos_label: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision-recall curve for binary case using sklearn."""
    if not ((0 < preds) & (preds < 1)).all():
        preds = sigmoid(preds)

    return sk_precision_recall_curve(
        y_true=target, probas_pred=preds, pos_label=pos_label, sample_weight=None
    )


@pytest.mark.parametrize("inputs", _binary_cases[1:])
def test_binary_precision_recall_curve(inputs):
    """Test binary precision-recall curve."""
    target, preds = inputs

    # test functional
    _functional_test(
        target,
        preds,
        cyclops_precision_recall_curve,
        _sk_binary_precision_recall_curve,
        {"task": "binary"},
    )


def _sk_multiclass_precision_recall_curve(
    target: np.ndarray,
    preds: np.ndarray,
) -> List:
    """Compute precision-recall curve for multiclass case using sklearn."""
    if not ((0 < preds) & (preds < 1)).all():
        preds = sp.special.softmax(preds, axis=1)

    precision, recall, thresholds = [], [], []
    for i in range(NUM_CLASSES):
        result = sk_precision_recall_curve(target, preds[:, i], pos_label=i)
        precision.append(result[1])
        recall.append(result[1])
        thresholds.append(result[2])

    return [np.nan_to_num(x, nan=0.0) for x in [precision, recall, thresholds]]


@pytest.mark.parametrize("inputs", _multiclass_cases[1:])
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_multiclass_precision_recall_curve(inputs):
    """Test multiclass precision-recall curve."""
    target, preds = inputs

    # test functional
    _functional_test(
        target,
        preds,
        cyclops_precision_recall_curve,
        _sk_multiclass_precision_recall_curve,
        {"task": "multiclass", "num_classes": NUM_CLASSES},
    )


def _sk_multilabel_precision_recall_curve(
    target: np.ndarray, preds: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    precision, recall, thresholds = [], [], []
    for i in range(NUM_LABELS):
        res = _sk_binary_precision_recall_curve(target[:, i], preds[:, i])
        precision.append(res[0])
        recall.append(res[1])
        thresholds.append(res[2])

    return precision, recall, thresholds


@pytest.mark.parametrize("inputs", _multilabel_cases[1:])
def test_multilabel_precision_recall_curve(inputs):
    """Test multilabel precision-recall curve."""
    target, preds = inputs

    # test functional
    _functional_test(
        target,
        preds,
        cyclops_precision_recall_curve,
        _sk_multilabel_precision_recall_curve,
        {"task": "multilabel", "num_labels": NUM_LABELS},
    )
