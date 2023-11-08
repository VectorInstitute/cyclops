"""Functions for testing ROC metrics."""

from typing import Any, List, Tuple

import numpy as np
import pytest
import scipy as sp
from sklearn.metrics import roc_curve as sk_roc_curve

from cyclops.evaluate.metrics.functional import roc_curve as cyclops_roc_curve
from cyclops.evaluate.metrics.roc import ROCCurve
from cyclops.evaluate.metrics.utils import sigmoid

from .helpers import MetricTester
from .inputs import (
    NUM_CLASSES,
    NUM_LABELS,
    _binary_cases,
    _multiclass_cases,
    _multilabel_cases,
)


def _sk_binary_roc_curve(
    target: np.ndarray,
    preds: np.ndarray,
    pos_label: int = 1,
) -> List[Any]:
    """Compute ROC curve for binary case using sklearn."""
    if not ((preds > 0) & (preds < 1)).all():
        preds = sigmoid(preds)

    fpr, tpr, thresholds = sk_roc_curve(
        y_true=target,
        y_score=preds,
        pos_label=pos_label,
        drop_intermediate=False,
    )
    thresholds[0] = 1.0

    return [np.nan_to_num(x, nan=0.0) for x in [fpr, tpr, thresholds]]


@pytest.mark.parametrize("inputs", _binary_cases[2:])
@pytest.mark.filterwarnings("ignore::UserWarning")  # np.nan_to_num warning
class TestBinaryROCCurve(MetricTester):
    """Test function and class for computing the ROC curve for binary targets."""

    def test_binary_roc_curve_functional(self, inputs):
        """Test function for computing the ROC curve for binary targets."""
        target, preds = inputs

        self.run_functional_test(
            target=target,
            preds=preds,
            metric_functional=cyclops_roc_curve,
            sk_metric=_sk_binary_roc_curve,
            metric_args={"task": "binary"},
        )

    def test_binary_roc_curve_class(self, inputs):
        """Test class for computing the ROC curve for binary targets."""
        target, preds = inputs

        self.run_class_test(
            target=target,
            preds=preds,
            metric_class=ROCCurve,
            sk_metric=_sk_binary_roc_curve,
            metric_args={"task": "binary"},
        )


def _sk_multiclass_roc_curve(
    target: np.ndarray,
    preds: np.ndarray,
) -> List:
    """Compute ROC curve for multiclass case using sklearn."""
    if not ((preds > 0) & (preds < 1)).all():
        preds = sp.special.softmax(preds, 1)

    fpr, tpr, thresholds = [], [], []
    for i in range(NUM_CLASSES):
        res = sk_roc_curve(target, preds[:, i], pos_label=i, drop_intermediate=False)
        res[2][0] = 1.0

        fpr.append(res[0])
        tpr.append(res[1])
        thresholds.append(res[2])
    return [np.nan_to_num(x, nan=0.0) for x in [fpr, tpr, thresholds]]


@pytest.mark.parametrize("inputs", _multiclass_cases[2:])
@pytest.mark.filterwarnings("ignore::UserWarning")  # np.nan_to_num warning
class TestMulticlassROCCurve(MetricTester):
    """Test function and class for computing the ROC curve for multiclass input."""

    def test_multiclass_roc_curve_functional(self, inputs) -> None:
        """Test function for computing the ROC curve for multiclass input."""
        target, preds = inputs

        self.run_functional_test(
            target=target,
            preds=preds,
            metric_functional=cyclops_roc_curve,
            sk_metric=_sk_multiclass_roc_curve,
            metric_args={"task": "multiclass", "num_classes": NUM_CLASSES},
        )

    def test_multiclass_roc_curve_class(self, inputs) -> None:
        """Test class for computing the ROC curve for multiclass input."""
        target, preds = inputs

        self.run_class_test(
            target=target,
            preds=preds,
            metric_class=ROCCurve,
            sk_metric=_sk_multiclass_roc_curve,
            metric_args={"task": "multiclass", "num_classes": NUM_CLASSES},
        )


def _sk_multilabel_roc_curve(
    target: np.ndarray,
    preds: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Compute ROC curve for multilabel case using sklearn."""
    if preds.ndim == 1:
        preds = np.expand_dims(preds, axis=0)
    if target.ndim == 1:
        target = np.expand_dims(target, axis=0)

    fpr, tpr, thresholds = [], [], []
    for i in range(NUM_LABELS):
        res = _sk_binary_roc_curve(target[:, i], preds[:, i])
        fpr.append(res[0])
        tpr.append(res[1])
        thresholds.append(res[2])

    return fpr, tpr, thresholds


@pytest.mark.parametrize("inputs", _multilabel_cases[2:])
@pytest.mark.filterwarnings("ignore::UserWarning")  # no positive samples
class TestMultilabelROCCurve(MetricTester):
    """Test function and class for computing ROC curve for multilabel targets."""

    def test_multilabel_roc_curve_functional(self, inputs) -> None:
        """Test function for computing the ROC curve for multilabel targets."""
        target, preds = inputs

        self.run_functional_test(
            target=target,
            preds=preds,
            metric_functional=cyclops_roc_curve,
            sk_metric=_sk_multilabel_roc_curve,
            metric_args={"task": "multilabel", "num_labels": NUM_LABELS},
        )

    def test_multilabel_roc_curve_class(self, inputs) -> None:
        """Test class for computing the ROC curve for multilabel targets."""
        target, preds = inputs

        self.run_class_test(
            target=target,
            preds=preds,
            metric_class=ROCCurve,
            sk_metric=_sk_multilabel_roc_curve,
            metric_args={"task": "multilabel", "num_labels": NUM_LABELS},
        )
