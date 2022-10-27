"""Functions for testing the AUCROC metrics."""
from functools import partial
from typing import Literal

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
from sklearn.metrics import roc_auc_score as sk_roc_auc_score

from cyclops.evaluation.metrics.functional import auroc as cyclops_auroc
from cyclops.evaluation.metrics.utils import sigmoid


def _sk_binary_auroc(
    target: np.ndarray,
    preds: np.ndarray,
    max_fpr: float = None,
) -> float:
    """Compute AUROC for binary case using sklearn."""
    if not ((0 < preds) & (preds < 1)).all():
        preds = sigmoid(preds)

    return sk_roc_auc_score(y_true=target, y_score=preds, max_fpr=max_fpr)


@pytest.mark.parametrize("inputs", _binary_cases[1:])
@pytest.mark.parametrize("max_fpr", [None, 0.7])
def test_binary_auroc(inputs, max_fpr):
    """Test binary AUROC."""
    target, preds = inputs

    # test functional
    _functional_test(
        target,
        preds,
        cyclops_auroc,
        partial(_sk_binary_auroc, max_fpr=max_fpr),
        {"task": "binary", "max_fpr": max_fpr},
    )


def _sk_multiclass_auroc(
    target: np.ndarray,
    preds: np.ndarray,
    average: Literal["macro", "weighted"] = "macro",
) -> float:
    """Compute AUROC for multiclass case using sklearn."""
    if not ((0 < preds) & (preds < 1)).all():
        preds = sp.special.softmax(preds, axis=1)

    if not np.array_equiv(np.unique(target), np.arange(NUM_CLASSES)):
        pytest.skip("sklearn does not support multiclass AUROC with missing classes.")

    return sk_roc_auc_score(
        y_true=target,
        y_score=preds,
        multi_class="ovr",
        average=average,
        labels=list(range(NUM_CLASSES)),
    )


@pytest.mark.parametrize("inputs", _multiclass_cases[1:])
@pytest.mark.parametrize("average", ["macro", "weighted"])
@pytest.mark.filterwarnings("ignore::UserWarning")  # no positive samples
def test_multiclass_auroc(inputs, average):
    """Test multiclass AUROC."""
    target, preds = inputs

    # test functional
    _functional_test(
        target,
        preds,
        cyclops_auroc,
        partial(_sk_multiclass_auroc, average=average),
        {
            "task": "multiclass",
            "num_classes": NUM_CLASSES,
            "average": average,
        },
    )


def _sk_multilabel_auroc(
    target: np.ndarray,
    preds: np.ndarray,
    average: Literal["micro", "macro", "weighted"] = "macro",
) -> float:
    """Compute AUROC for multilabel case using sklearn."""
    if not ((0 < preds) & (preds < 1)).all():
        preds = sigmoid(preds)

    return sk_roc_auc_score(
        target, preds, average=average, max_fpr=None, labels=list(range(NUM_LABELS))
    )


@pytest.mark.parametrize("inputs", _multilabel_cases[1:])
@pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
@pytest.mark.filterwarnings("ignore::UserWarning")  # no positive samples
def test_multilabel_auroc(inputs, average):
    """Test multilabel AUROC."""
    target, preds = inputs

    # test functional
    _functional_test(
        target,
        preds,
        cyclops_auroc,
        partial(_sk_multilabel_auroc, average=average),
        {
            "task": "multilabel",
            "num_labels": NUM_LABELS,
            "average": average,
        },
    )
