"""Functions for test F-beta and F1 score."""
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
from sklearn.metrics import fbeta_score as sk_fbeta_score

from cyclops.evaluation.metrics.functional.f_beta import f1_score, fbeta_score
from cyclops.evaluation.metrics.utils import sigmoid


def _sk_binary_fbeta_score(
    target: np.ndarray,
    preds: np.ndarray,
    beta: float,
    threshold: float,
    zero_division: Literal["warn", 0, 1],
):
    """Compute fbeta score for binary case using sklearn."""
    if np.issubdtype(preds.dtype, np.floating):
        if not ((0 < preds) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= threshold).astype(np.uint8)

    return sk_fbeta_score(
        y_true=target, y_pred=preds, beta=beta, zero_division=zero_division
    )


@pytest.mark.parametrize("inputs", _binary_cases)
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_binary_fbeta(inputs, beta):
    """Test binary fbeta."""
    target, preds = inputs

    # test functional
    _functional_test(
        target,
        preds,
        f1_score
        if beta == 1.0
        else partial(
            fbeta_score,
            beta=beta,
            task="binary",
        ),
        partial(
            _sk_binary_fbeta_score,
            beta=beta,
            threshold=THRESHOLD,
            zero_division=0,
        ),
        {"task": "binary", "threshold": THRESHOLD, "zero_division": 0},
    )


def _sk_multiclass_fbeta_score(
    target: np.ndarray,
    preds: np.ndarray,
    beta: float,
    average: Literal["micro", "macro", "weighted"],
    zero_division: Literal["warn", 0, 1],
):
    """Compute fbeta score for multiclass case using sklearn."""
    if np.issubdtype(preds.dtype, np.floating):
        preds = np.argmax(preds, axis=1)

    return sk_fbeta_score(
        y_true=target,
        y_pred=preds,
        beta=beta,
        average=average,
        zero_division=zero_division,
    )


@pytest.mark.parametrize("inputs", _multiclass_cases)
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("average", ["micro", "macro", "weighted"])
def test_multiclass_fbeta(inputs, beta, average):
    """Test multiclass fbeta."""
    target, preds = inputs

    # test functional
    _functional_test(
        target,
        preds,
        f1_score if beta == 1.0 else partial(fbeta_score, beta=beta),
        partial(
            _sk_multiclass_fbeta_score,
            beta=beta,
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


def _sk_multilabel_fbeta_score(  # pylint: disable=too-many-arguments
    target: np.ndarray,
    preds: np.ndarray,
    beta: float,
    threshold: float,
    average: Literal["micro", "macro", "weighted"],
    zero_division: Literal["warn", 0, 1],
):
    """Compute fbeta score for multilabel case using sklearn."""
    if np.issubdtype(preds.dtype, np.floating):
        if not ((0 < preds) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= threshold).astype(np.uint8)

    return sk_fbeta_score(
        y_true=target,
        y_pred=preds,
        beta=beta,
        average=average,
        zero_division=zero_division,
    )


@pytest.mark.parametrize("inputs", _multilabel_cases)
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("average", ["micro", "macro", "weighted"])
def test_multilabel_fbeta(inputs, beta, average):
    """Test multilabel fbeta."""
    target, preds = inputs

    # test functional
    _functional_test(
        target,
        preds,
        f1_score if beta == 1.0 else partial(fbeta_score, beta=beta),
        partial(
            _sk_multilabel_fbeta_score,
            beta=beta,
            average=average,
            threshold=THRESHOLD,
            zero_division=0,
        ),
        {
            "task": "multilabel",
            "num_labels": NUM_LABELS,
            "average": average,
            "threshold": THRESHOLD,
            "zero_division": 0,
        },
    )
