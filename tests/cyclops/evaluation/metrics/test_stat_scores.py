"""Test stat_scores functions."""
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
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import (
    multilabel_confusion_matrix as sk_multilabel_confusion_matrix,
)

from cyclops.evaluation.metrics.functional.stat_scores import stat_scores
from cyclops.evaluation.metrics.utils import sigmoid


def _sk_stat_scores_binary(
    target: np.ndarray, preds: np.ndarray, threshold: float
) -> np.ndarray:
    """Compute stat scores for binary case using sklearn."""
    # pylint: disable=invalid-name
    if np.issubdtype(preds.dtype, np.floating):
        if not ((0 < preds) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= threshold).astype(np.uint8)

    tn, fp, fn, tp = sk_confusion_matrix(
        y_true=target, y_pred=preds, labels=[0, 1]
    ).ravel()
    return np.array([tp, fp, tn, fn, tp + fn])


@pytest.mark.parametrize("inputs", _binary_cases)
def test_binary_stat_scores(inputs):
    """Test binary case."""
    target, preds = inputs

    # test functional
    _functional_test(
        target,
        preds,
        stat_scores,
        partial(_sk_stat_scores_binary, threshold=THRESHOLD),
        {"task": "binary", "threshold": THRESHOLD},
    )


def _sk_stat_scores_multiclass(
    target: np.ndarray, preds: np.ndarray, classwise: bool
) -> np.ndarray:
    """Compute stat scores for multiclass case using sklearn."""
    # pylint: disable=invalid-name
    if preds.ndim == target.ndim + 1:
        preds = np.argmax(preds, axis=1)
    confmat = sk_multilabel_confusion_matrix(
        y_true=target, y_pred=preds, labels=list(range(NUM_CLASSES))
    )

    tn = confmat[:, 0, 0]
    fn = confmat[:, 1, 0]
    tp = confmat[:, 1, 1]
    fp = confmat[:, 0, 1]

    if not classwise:
        tp = tp.sum(keepdims=True)
        fp = fp.sum(keepdims=True)
        tn = tn.sum(keepdims=True)
        fn = fn.sum(keepdims=True)
        return np.concatenate([tp, fp, tn, fn, tp + fn])
    return np.stack([tp, fp, tn, fn, tp + fn], axis=-1)


@pytest.mark.parametrize("inputs", _multiclass_cases)
@pytest.mark.parametrize("classwise", [True, False])
def test_multiclass_stat_scores(inputs, classwise):
    """Test multiclass case."""
    target, preds = inputs

    # test functional
    _functional_test(
        target,
        preds,
        stat_scores,
        partial(_sk_stat_scores_multiclass, classwise=classwise),
        {"task": "multiclass", "num_classes": NUM_CLASSES, "classwise": classwise},
    )


def _sk_stat_scores_multilabel(
    target: np.ndarray,
    preds: np.ndarray,
    threshold: float,
    reduce: Literal["micro", "macro", "samples"],
) -> np.ndarray:
    """Compute stat scores for multilabel case using sklearn."""
    # pylint: disable=invalid-name
    if np.issubdtype(preds.dtype, np.floating):
        if not ((0 < preds) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= threshold).astype(np.uint8)

    confmat = sk_multilabel_confusion_matrix(
        y_true=target,
        y_pred=preds,
        labels=list(range(NUM_LABELS)),
        samplewise=reduce == "samples",
    )

    tn = confmat[:, 0, 0]
    fn = confmat[:, 1, 0]
    tp = confmat[:, 1, 1]
    fp = confmat[:, 0, 1]

    if reduce == "micro":
        tp = tp.sum(keepdims=True)
        fp = fp.sum(keepdims=True)
        tn = tn.sum(keepdims=True)
        fn = fn.sum(keepdims=True)
        return np.concatenate([tp, fp, tn, fn, tp + fn])
    return np.stack([tp, fp, tn, fn, tp + fn], axis=-1)


@pytest.mark.parametrize("inputs", _multilabel_cases)
@pytest.mark.parametrize("reduce", ["micro", "macro", "samples"])
def test_multilabel_stat_scores(inputs, reduce):
    """Test multilabel case."""
    target, preds = inputs

    # test functional
    _functional_test(
        target,
        preds,
        stat_scores,
        partial(_sk_stat_scores_multilabel, threshold=THRESHOLD, reduce=reduce),
        {
            "task": "multilabel",
            "num_labels": NUM_LABELS,
            "threshold": THRESHOLD,
            "reduce": reduce,
        },
    )
