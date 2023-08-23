"""Test stat_scores functions."""

from functools import partial

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import (
    multilabel_confusion_matrix as sk_multilabel_confusion_matrix,
)
from sklearn.preprocessing import label_binarize

from cyclops.evaluate.metrics.functional.stat_scores import stat_scores
from cyclops.evaluate.metrics.stat_scores import StatScores
from cyclops.evaluate.metrics.utils import sigmoid
from metrics.helpers import MetricTester
from metrics.inputs import (
    NUM_CLASSES,
    NUM_LABELS,
    THRESHOLD,
    _binary_cases,
    _multiclass_cases,
    _multilabel_cases,
)


def _sk_stat_scores_binary(
    target: np.ndarray, preds: np.ndarray, threshold: float,
) -> np.ndarray:
    """Compute stat scores for binary case using sklearn."""
    if np.issubdtype(preds.dtype, np.floating):
        if not ((preds > 0) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= threshold).astype(np.uint8)

    if target.ndim == 0:
        target = label_binarize(np.expand_dims(target, axis=0), classes=[0, 1])
    if preds.ndim == 0:
        preds = label_binarize(np.expand_dims(preds, axis=0), classes=[0, 1])

    tn, fp, fn, tp = sk_confusion_matrix(
        y_true=target, y_pred=preds, labels=[0, 1],
    ).ravel()
    return np.array([tp, fp, tn, fn, tp + fn])


@pytest.mark.parametrize("inputs", _binary_cases)
class TestBinaryStatScores(MetricTester):
    """Test function and class for binary stat scores."""

    def test_binary_stat_scores_functional(self, inputs) -> None:
        """Test function for binary stat scores."""
        target, preds = inputs

        self.run_functional_test(
            target=target,
            preds=preds,
            metric_functional=stat_scores,
            sk_metric=partial(_sk_stat_scores_binary, threshold=THRESHOLD),
            metric_args={"task": "binary", "threshold": THRESHOLD},
        )

    def test_binary_stat_scores_class(self, inputs) -> None:
        """Test class for binary stat scores."""
        target, preds = inputs

        self.run_class_test(
            target=target,
            preds=preds,
            metric_class=StatScores,
            sk_metric=partial(_sk_stat_scores_binary, threshold=THRESHOLD),
            metric_args={"task": "binary", "threshold": THRESHOLD},
        )


def _sk_stat_scores_multiclass(
    target: np.ndarray, preds: np.ndarray, classwise: bool,
) -> np.ndarray:
    """Compute stat scores for multiclass case using sklearn."""
    if preds.ndim == target.ndim + 1:
        preds = np.argmax(preds, axis=-1)

    # convert 0D arrays to one-hot
    if target.ndim == 0:
        target = label_binarize(
            np.expand_dims(target, axis=0), classes=list(range(NUM_CLASSES)),
        )
    if preds.ndim == 0:
        preds = label_binarize(
            np.expand_dims(preds, axis=0), classes=list(range(NUM_CLASSES)),
        )

    confmat = sk_multilabel_confusion_matrix(
        y_true=target, y_pred=preds, labels=list(range(NUM_CLASSES)),
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
class TestMulticlassStatScores(MetricTester):
    """Test function and class for multiclass stat scores."""

    def test_multiclass_stat_scores_functional(self, inputs, classwise) -> None:
        """Test function for multiclass stat scores."""
        target, preds = inputs

        self.run_functional_test(
            target=target,
            preds=preds,
            metric_functional=stat_scores,
            sk_metric=partial(_sk_stat_scores_multiclass, classwise=classwise),
            metric_args={
                "task": "multiclass",
                "num_classes": NUM_CLASSES,
                "classwise": classwise,
            },
        )

    def test_multiclass_stat_scores_class(self, inputs, classwise) -> None:
        """Test class for multiclass stat scores."""
        target, preds = inputs

        self.run_class_test(
            target=target,
            preds=preds,
            metric_class=StatScores,
            sk_metric=partial(_sk_stat_scores_multiclass, classwise=classwise),
            metric_args={
                "task": "multiclass",
                "num_classes": NUM_CLASSES,
                "classwise": classwise,
            },
        )


def _sk_stat_scores_multilabel(
    target: np.ndarray,
    preds: np.ndarray,
    threshold: float,
    labelwise: bool,
) -> np.ndarray:
    """Compute stat scores for multilabel case using sklearn."""
    if np.issubdtype(preds.dtype, np.floating):
        if not ((preds > 0) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= threshold).astype(np.uint8)

    confmat = sk_multilabel_confusion_matrix(
        y_true=target,
        y_pred=preds,
        labels=list(range(NUM_LABELS)),
    )

    tn = confmat[:, 0, 0]
    fn = confmat[:, 1, 0]
    tp = confmat[:, 1, 1]
    fp = confmat[:, 0, 1]

    if not labelwise:
        tp = tp.sum(keepdims=True)
        fp = fp.sum(keepdims=True)
        tn = tn.sum(keepdims=True)
        fn = fn.sum(keepdims=True)
        return np.concatenate([tp, fp, tn, fn, tp + fn])
    return np.stack([tp, fp, tn, fn, tp + fn], axis=-1)


@pytest.mark.parametrize("inputs", _multilabel_cases)
@pytest.mark.parametrize("labelwise", [True, False])
class TestMultilabelStatScores(MetricTester):
    """Test function and class for multilabel stat scores."""

    def test_multilabel_stat_scores_functional(self, inputs, labelwise) -> None:
        """Test function for multilabel stat scores."""
        target, preds = inputs

        self.run_functional_test(
            target=target,
            preds=preds,
            metric_functional=stat_scores,
            sk_metric=partial(
                _sk_stat_scores_multilabel, threshold=THRESHOLD, labelwise=labelwise,
            ),
            metric_args={
                "task": "multilabel",
                "num_labels": NUM_LABELS,
                "threshold": THRESHOLD,
                "labelwise": labelwise,
            },
        )

    def test_multilabel_stat_scores_class(self, inputs, labelwise) -> None:
        """Test class for multilabel stat scores."""
        target, preds = inputs

        self.run_class_test(
            target=target,
            preds=preds,
            metric_class=StatScores,
            sk_metric=partial(
                _sk_stat_scores_multilabel, threshold=THRESHOLD, labelwise=labelwise,
            ),
            metric_args={
                "task": "multilabel",
                "num_labels": NUM_LABELS,
                "threshold": THRESHOLD,
                "labelwise": labelwise,
            },
        )
