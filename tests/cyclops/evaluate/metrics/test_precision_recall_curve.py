"""Functions for testing precision-recall curve metrics."""

from typing import List, Tuple

import numpy as np
import pytest
import scipy as sp
from sklearn.metrics import precision_recall_curve as sk_precision_recall_curve

from cyclops.evaluate.metrics.functional import (
    precision_recall_curve as cyclops_precision_recall_curve,
)
from cyclops.evaluate.metrics.precision_recall_curve import PrecisionRecallCurve
from cyclops.evaluate.metrics.utils import sigmoid
from metrics.helpers import MetricTester
from metrics.inputs import (
    NUM_CLASSES,
    NUM_LABELS,
    _binary_cases,
    _multiclass_cases,
    _multilabel_cases,
)


def _sk_binary_precision_recall_curve(
    target: np.ndarray,
    preds: np.ndarray,
    pos_label: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision-recall curve for binary case using sklearn."""
    if not ((preds > 0) & (preds < 1)).all():
        preds = sigmoid(preds)

    return sk_precision_recall_curve(
        y_true=target,
        probas_pred=preds,
        pos_label=pos_label,
    )


@pytest.mark.parametrize("inputs", _binary_cases[2:])
class TestBinaryPrecisionRecallCurve(MetricTester):
    """Test function and class for computing binary precision-recall curve."""

    def test_binary_precision_recall_curve_functional(self, inputs):
        """Test function for computing binary precision-recall curve."""
        target, preds = inputs

        self.run_functional_test(
            target=target,
            preds=preds,
            metric_functional=cyclops_precision_recall_curve,
            sk_metric=_sk_binary_precision_recall_curve,
            metric_args={"task": "binary"},
        )

    def test_binary_precision_recall_curve_class(self, inputs):
        """Test class for computing binary precision-recall curve."""
        target, preds = inputs

        self.run_class_test(
            target=target,
            preds=preds,
            metric_class=PrecisionRecallCurve,
            sk_metric=_sk_binary_precision_recall_curve,
            metric_args={"task": "binary"},
        )


def _sk_multiclass_precision_recall_curve(
    target: np.ndarray,
    preds: np.ndarray,
) -> List:
    """Compute precision-recall curve for multiclass case using sklearn."""
    if not ((preds > 0) & (preds < 1)).all():
        preds = sp.special.softmax(preds, axis=1)

    precision, recall, thresholds = [], [], []
    for i in range(NUM_CLASSES):
        result = sk_precision_recall_curve(target, preds[:, i], pos_label=i)
        precision.append(result[1])
        recall.append(result[1])
        thresholds.append(result[2])

    return [np.nan_to_num(x, nan=0.0) for x in [precision, recall, thresholds]]


@pytest.mark.parametrize("inputs", _multiclass_cases[2:])
@pytest.mark.filterwarnings("ignore::UserWarning")
class TestMulticlassPrecisionRecallCurve(MetricTester):
    """Test function and class for computing multiclass precision-recall curve."""

    def test_multiclass_precision_recall_curve_functional(self, inputs) -> None:
        """Test function for computing multiclass precision-recall curve."""
        target, preds = inputs

        self.run_functional_test(
            target=target,
            preds=preds,
            metric_functional=cyclops_precision_recall_curve,
            sk_metric=_sk_multiclass_precision_recall_curve,
            metric_args={"task": "multiclass", "num_classes": NUM_CLASSES},
        )

    def test_multiclass_precision_recall_curve_class(self, inputs) -> None:
        """Test class for computing multiclass precision-recall curve."""
        target, preds = inputs

        self.run_class_test(
            target=target,
            preds=preds,
            metric_class=PrecisionRecallCurve,
            sk_metric=_sk_multiclass_precision_recall_curve,
            metric_args={"task": "multiclass", "num_classes": NUM_CLASSES},
        )


def _sk_multilabel_precision_recall_curve(
    target: np.ndarray,
    preds: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    if preds.ndim == 1:
        preds = np.expand_dims(preds, axis=0)
    if target.ndim == 1:
        target = np.expand_dims(target, axis=0)

    precision, recall, thresholds = [], [], []
    for i in range(NUM_LABELS):
        res = _sk_binary_precision_recall_curve(target[:, i], preds[:, i])
        precision.append(res[0])
        recall.append(res[1])
        thresholds.append(res[2])

    return precision, recall, thresholds


@pytest.mark.parametrize("inputs", _multilabel_cases[2:])
class TestMultilabelPrecisionRecallCurve(MetricTester):
    """Test function and class for computing multilabel precision-recall curve."""

    def test_multilabel_precision_recall_curve_functional(self, inputs) -> None:
        """Test function for computing multilabel precision-recall curve."""
        target, preds = inputs

        self.run_functional_test(
            target=target,
            preds=preds,
            metric_functional=cyclops_precision_recall_curve,
            sk_metric=_sk_multilabel_precision_recall_curve,
            metric_args={"task": "multilabel", "num_labels": NUM_LABELS},
        )

    def test_multilabel_precision_recall_curve_class(self, inputs) -> None:
        """Test class for computing multilabel precision-recall curve."""
        target, preds = inputs

        self.run_class_test(
            target=target,
            preds=preds,
            metric_class=PrecisionRecallCurve,
            sk_metric=_sk_multilabel_precision_recall_curve,
            metric_args={"task": "multilabel", "num_labels": NUM_LABELS},
        )
