"""Functions for testing accuracy metrics."""
from functools import partial
from typing import Literal

import numpy as np
import pytest
from metrics.helpers import _functional_test
from metrics.inputs import (
    NUM_CLASSES,
    NUM_LABELS,
    THRESHOLD,
    _binary_cases,
    _multiclass_cases,
    _multilabel_cases,
)
from metrics.test_stat_scores import (
    _sk_stat_scores_multiclass,
    _sk_stat_scores_multilabel,
)
from numpy.typing import ArrayLike
from sklearn.metrics import accuracy_score as sk_accuracy_score
from sklearn.metrics import top_k_accuracy_score as sk_top_k_accuracy_score

from cyclops.evaluation.metrics.functional.accuracy import accuracy
from cyclops.evaluation.metrics.utils import sigmoid

np.seterr(divide="ignore", invalid="ignore")  # ignore divide by zero or nan


def _sk_binary_accuracy(
    target: np.ndarray, preds: np.ndarray, threshold: float
) -> np.ndarray:
    """Compute accuracy for binary case using sklearn."""
    if np.issubdtype(preds.dtype, np.floating):
        if not ((0 < preds) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= threshold).astype(np.uint8)

    return sk_accuracy_score(y_true=target, y_pred=preds)


@pytest.mark.parametrize("inputs", _binary_cases)
def test_binary_accuracy(inputs):
    """Test binary accuracy."""
    target, preds = inputs

    _functional_test(
        target,
        preds,
        accuracy,
        partial(_sk_binary_accuracy, threshold=THRESHOLD),
        {"task": "binary", "threshold": THRESHOLD},
    )


def _sk_multiclass_accuracy(
    target: np.ndarray,
    preds: np.ndarray,
    average: Literal["micro", "macro", "weighted", None],
) -> np.ndarray:
    """Compute accuracy for multiclass case using sklearn."""
    if preds.ndim == target.ndim + 1:
        preds = np.argmax(preds, axis=1)

    if average == "micro":
        return sk_accuracy_score(
            y_true=target,
            y_pred=preds,
            normalize=True,
        )

    # pylint: disable=invalid-name
    scores = _sk_stat_scores_multiclass(target, preds, classwise=True)
    tp = scores[:, 0]
    support = scores[:, 4]

    accuracy_per_class = tp / support
    accuracy_per_class[np.isnan(accuracy_per_class)] = 0.0

    if average in ["macro", "weighted"]:
        weights = None
        if average == "weighted":
            weights = support
        return np.average(accuracy_per_class, weights=weights)

    return accuracy_per_class


@pytest.mark.parametrize("inputs", _multiclass_cases)
@pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
def test_multiclass_accuracy(inputs, average):
    """Test multiclass case."""
    target, preds = inputs

    _functional_test(
        target,
        preds,
        accuracy,
        partial(_sk_multiclass_accuracy, average=average),
        {
            "task": "multiclass",
            "num_classes": NUM_CLASSES,
            "average": average,
            "zero_division": 0,
        },
    )


def _sk_multilabel_accuracy(
    target: np.ndarray,
    preds: np.ndarray,
    average: Literal["micro", "macro", "weighted", "samples", None],
    sample_weight: ArrayLike = None,
) -> np.ndarray:
    """Compute accuracy for multilabel input using sklearn."""
    if np.issubdtype(preds.dtype, np.floating):
        if not ((0 < preds) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)

    if average == "micro":
        target = target.flatten()
        preds = preds.flatten()
        return sk_accuracy_score(y_true=target, y_pred=preds)

    accuracy_per_class = []
    for i in range(preds.shape[1]):
        accuracy_per_class.append(
            sk_accuracy_score(target[:, i].flatten(), preds[:, i].flatten())
        )

    res = np.stack(accuracy_per_class, axis=0)

    if average in ["macro", "weighted", "samples"]:
        weights = None
        if average == "weighted":
            confmat = _sk_stat_scores_multilabel(
                target, preds, threshold=THRESHOLD, reduce="macro"
            )
            weights = confmat[:, 4]
        elif average == "samples":
            weights = sample_weight

        if weights is None:
            return np.average(res, weights=weights)

        weights_norm = np.sum(weights, axis=-1, keepdims=True)
        weights_norm[weights_norm == 0] = 1.0
        return np.sum((weights * res) / weights_norm, axis=-1)

    return res


@pytest.mark.parametrize("inputs", _multiclass_cases[1:])
@pytest.mark.parametrize("top_k", list(range(1, 10)))
def test_topk(inputs, top_k):
    """Test top-k multiclass accuracy."""
    target, preds = inputs

    _functional_test(
        target,
        preds,
        accuracy,
        partial(sk_top_k_accuracy_score, labels=list(range(NUM_CLASSES)), k=top_k),
        {
            "task": "multiclass",
            "num_classes": NUM_CLASSES,
            "average": "micro",
            "top_k": top_k,
            "zero_division": 0,
        },
    )


@pytest.mark.parametrize("inputs", _multilabel_cases)
@pytest.mark.parametrize("average", ["micro", "macro", "samples", "weighted", None])
def test_multilabel_accuracy(inputs, average):
    """Test multilabel case."""
    target, preds = inputs

    _functional_test(
        target,
        preds,
        accuracy,
        partial(_sk_multilabel_accuracy, average=average),
        {
            "task": "multilabel",
            "num_labels": NUM_LABELS,
            "threshold": THRESHOLD,
            "average": average,
            "zero_division": 0,
        },
    )
