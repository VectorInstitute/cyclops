"""Functions for testing precision and recall metrics."""

from functools import partial
from typing import Callable, Literal

import numpy as np
import pytest
from sklearn.metrics import precision_score as sk_precision_score
from sklearn.metrics import recall_score as sk_recall_score

from cyclops.evaluate.metrics.functional.precision_recall import (
    precision as cyclops_precision,
)
from cyclops.evaluate.metrics.functional.precision_recall import (
    recall as cyclops_recall,
)
from cyclops.evaluate.metrics.precision_recall import Precision, Recall
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


def _sk_binary_precision_recall(
    target: np.ndarray,
    preds: np.ndarray,
    sk_fn: Callable,
    threshold: float,
    zero_division: Literal["warn", 0, 1],
):
    """Compute precision score for binary case using sklearn."""
    if np.issubdtype(preds.dtype, np.floating):
        if not ((preds > 0) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= threshold).astype(np.uint8)

    return sk_fn(y_true=target, y_pred=preds, zero_division=zero_division)


@pytest.mark.parametrize("inputs", _binary_cases)
class TestBinaryPrecisionRecall(MetricTester):
    """Test function and class for binary precision and recall."""

    @pytest.mark.parametrize(
        "cyclops_func, sk_func",
        [(cyclops_precision, sk_precision_score), (cyclops_recall, sk_recall_score)],
        ids=["precision", "recall"],
    )
    def test_binary_precision_recall_functional(
        self,
        inputs,
        cyclops_func,
        sk_func,
    ) -> None:
        """Test function for binary precision and recall."""
        target, preds = inputs

        self.run_functional_test(
            target=target,
            preds=preds,
            metric_functional=cyclops_func,
            sk_metric=partial(
                _sk_binary_precision_recall,
                sk_fn=sk_func,
                threshold=THRESHOLD,
                zero_division=0,
            ),
            metric_args={"task": "binary", "threshold": THRESHOLD, "zero_division": 0},
        )

    @pytest.mark.parametrize(
        "cyclops_class, sk_func",
        [(Precision, sk_precision_score), (Recall, sk_recall_score)],
        ids=["precision", "recall"],
    )
    def test_binary_precision_recall_class(
        self,
        inputs,
        cyclops_class,
        sk_func,
    ) -> None:
        """Test class for binary precision and recall."""
        target, preds = inputs

        self.run_class_test(
            target=target,
            preds=preds,
            metric_class=cyclops_class,
            sk_metric=partial(
                _sk_binary_precision_recall,
                sk_fn=sk_func,
                threshold=THRESHOLD,
                zero_division=0,
            ),
            metric_args={"task": "binary", "threshold": THRESHOLD, "zero_division": 0},
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
class TestMulticlassPrecisionRecall(MetricTester):
    """Test function and class for multiclass precision and recall."""

    @pytest.mark.parametrize(
        "cyclops_func, sk_func",
        [(cyclops_precision, sk_precision_score), (cyclops_recall, sk_recall_score)],
        ids=["precision", "recall"],
    )
    def test_multiclass_precision_recall_functional(
        self,
        inputs,
        average,
        cyclops_func,
        sk_func,
    ) -> None:
        """Test functions for multiclass precision and recall."""
        target, preds = inputs

        self.run_functional_test(
            target=target,
            preds=preds,
            metric_functional=cyclops_func,
            sk_metric=partial(
                _sk_multiclass_precision_recall,
                sk_fn=sk_func,
                average=average,
                zero_division=0,
            ),
            metric_args={
                "task": "multiclass",
                "num_classes": NUM_CLASSES,
                "average": average,
                "zero_division": 0,
            },
        )

    @pytest.mark.parametrize(
        "cyclops_class, sk_func",
        [(Precision, sk_precision_score), (Recall, sk_recall_score)],
        ids=["precision", "recall"],
    )
    def test_multiclass_precision_recall_class(
        self,
        inputs,
        average,
        cyclops_class,
        sk_func,
    ) -> None:
        """Test classes for multiclass precision and recall."""
        target, preds = inputs

        self.run_class_test(
            target=target,
            preds=preds,
            metric_class=cyclops_class,
            sk_metric=partial(
                _sk_multiclass_precision_recall,
                sk_fn=sk_func,
                average=average,
                zero_division=0,
            ),
            metric_args={
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
    average: Literal["micro", "macro", "weighted", None],
    zero_division: Literal["warn", 0, 1],
):
    """Compute precision score for multilabel case using sklearn."""
    if np.issubdtype(preds.dtype, np.floating):
        if not ((preds > 0) & (preds < 1)).all():
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
@pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
class TestMultilabelPrecisionRecall(MetricTester):
    """Test classes and functions for multilabel precision and recall."""

    @pytest.mark.parametrize(
        "cyclops_func, sk_func",
        [(cyclops_precision, sk_precision_score), (cyclops_recall, sk_recall_score)],
        ids=["precision", "recall"],
    )
    def test_multilabel_precision_recall_functional(
        self,
        inputs,
        average,
        cyclops_func,
        sk_func,
    ) -> None:
        """Test multilabel precision."""
        target, preds = inputs

        self.run_functional_test(
            target=target,
            preds=preds,
            metric_functional=cyclops_func,
            sk_metric=partial(
                _sk_multilabel_precision_recall,
                sk_fn=sk_func,
                threshold=THRESHOLD,
                average=average,
                zero_division=0,
            ),
            metric_args={
                "task": "multilabel",
                "num_labels": NUM_LABELS,
                "threshold": THRESHOLD,
                "average": average,
                "zero_division": 0,
            },
        )

    @pytest.mark.parametrize(
        "cyclops_class, sk_func",
        [(Precision, sk_precision_score), (Recall, sk_recall_score)],
        ids=["precision", "recall"],
    )
    def test_multilabel_precision(self, inputs, average, cyclops_class, sk_func):
        """Test classes for multilabel precision and recall."""
        target, preds = inputs

        self.run_class_test(
            target=target,
            preds=preds,
            metric_class=cyclops_class,
            sk_metric=partial(
                _sk_multilabel_precision_recall,
                sk_fn=sk_func,
                threshold=THRESHOLD,
                average=average,
                zero_division=0,
            ),
            metric_args={
                "task": "multilabel",
                "num_labels": NUM_LABELS,
                "threshold": THRESHOLD,
                "average": average,
                "zero_division": 0,
            },
        )
