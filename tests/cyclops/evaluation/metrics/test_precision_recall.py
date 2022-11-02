"""Functions for testing precision and recall metrics."""
from functools import partial
from typing import Callable, Literal

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
from sklearn.metrics import precision_score as sk_precision_score
from sklearn.metrics import recall_score as sk_recall_score

from cyclops.evaluation.metrics.functional.precision_recall import (
    precision as cyclops_precision,
)
from cyclops.evaluation.metrics.functional.precision_recall import (
    recall as cyclops_recall,
)
from cyclops.evaluation.metrics.utils import sigmoid


def _sk_binary_precision_recall(
    target: np.ndarray,
    preds: np.ndarray,
    sk_fn: Callable,
    threshold: float,
    zero_division: Literal["warn", 0, 1],
):
    """Compute precision score for binary case using sklearn."""
    if np.issubdtype(preds.dtype, np.floating):
        if not ((0 < preds) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= threshold).astype(np.uint8)

    return sk_fn(y_true=target, y_pred=preds, zero_division=zero_division)


@pytest.mark.parametrize("inputs", _binary_cases)
@pytest.mark.parametrize(
    "functional, sk_fn",
    [(cyclops_precision, sk_precision_score), (cyclops_recall, sk_recall_score)],
    ids=["precision", "recall"],
)
def test_binary_precision(inputs, functional, sk_fn):
    """Test binary precision."""
    target, preds = inputs

    # test functional
    _functional_test(
        target,
        preds,
        functional,
        partial(
            _sk_binary_precision_recall,
            sk_fn=sk_fn,
            threshold=THRESHOLD,
            zero_division=0,
        ),
        {"task": "binary", "threshold": THRESHOLD, "zero_division": 0},
    )


def _sk_multiclass_precision_recall(
    target: np.ndarray,
    preds: np.ndarray,
    sk_fn: Callable,
    average: Literal["micro", "macro", "weighted", None],
    zero_division: Literal["warn", 0, 1],
):
    """Compute precision score for multiclass case using sklearn."""
    if preds.ndim == target.ndim + 1:
        preds = np.argmax(preds, axis=1)

    return sk_fn(
        y_true=target,
        y_pred=preds,
        labels=list(range(NUM_CLASSES)),
        average=average,
        zero_division=zero_division,
    )


@pytest.mark.parametrize("inputs", _multiclass_cases)
@pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
@pytest.mark.parametrize(
    "functional, sk_fn",
    [(cyclops_precision, sk_precision_score), (cyclops_recall, sk_recall_score)],
    ids=["precision", "recall"],
)
def test_multiclass_precision(inputs, average, functional, sk_fn):
    """Test multiclass precision."""
    target, preds = inputs

    # test functional
    _functional_test(
        target,
        preds,
        functional,
        partial(
            _sk_multiclass_precision_recall,
            sk_fn=sk_fn,
            average=average,
            zero_division=0,
        ),
        {
            "task": "multiclass",
            "num_classes": NUM_CLASSES,
            "average": average,
            "zero_division": 0,
        },
    )


def _sk_multilabel_precision_recall(  # pylint: disable=too-many-arguments
    target: np.ndarray,
    preds: np.ndarray,
    sk_fn: Callable,
    threshold: float,
    average: Literal["samples", "micro", "macro", "weighted", None],
    zero_division: Literal["warn", 0, 1],
):
    """Compute precision score for multilabel case using sklearn."""
    if np.issubdtype(preds.dtype, np.floating):
        if not ((0 < preds) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= threshold).astype(np.uint8)

    return sk_fn(
        y_true=target,
        y_pred=preds,
        labels=list(range(NUM_LABELS)),
        average=average,
        zero_division=zero_division,
    )


@pytest.mark.parametrize("inputs", _multilabel_cases)
@pytest.mark.parametrize("average", ["samples", "micro", "macro", "weighted", None])
@pytest.mark.parametrize(
    "functional, sk_fn",
    [(cyclops_precision, sk_precision_score), (cyclops_recall, sk_recall_score)],
    ids=["precision", "recall"],
)
def test_multilabel_precision(inputs, average, functional, sk_fn):
    """Test multilabel precision."""
    target, preds = inputs

    # test functional
    _functional_test(
        target,
        preds,
        functional,
        partial(
            _sk_multilabel_precision_recall,
            sk_fn=sk_fn,
            threshold=THRESHOLD,
            average=average,
            zero_division=0,
        ),
        {
            "task": "multilabel",
            "num_labels": NUM_LABELS,
            "threshold": THRESHOLD,
            "average": average,
            "zero_division": 0,
        },
    )
