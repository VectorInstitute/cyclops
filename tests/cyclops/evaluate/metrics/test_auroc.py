"""Functions for testing the AUCROC metrics."""

from functools import partial
from typing import Literal, Optional

import numpy as np
import pytest
import scipy as sp
from sklearn.metrics import roc_auc_score as sk_roc_auc_score

from cyclops.evaluate.metrics.auroc import AUROC
from cyclops.evaluate.metrics.functional import auroc as cyclops_auroc
from cyclops.evaluate.metrics.utils import sigmoid

from .helpers import MetricTester
from .inputs import (
    NUM_CLASSES,
    NUM_LABELS,
    _binary_cases,
    _multiclass_cases,
    _multilabel_cases,
)


def _sk_binary_auroc(
    target: np.ndarray,
    preds: np.ndarray,
    max_fpr: Optional[float] = None,
) -> float:
    """Compute AUROC for binary case using sklearn."""
    if not ((preds > 0) & (preds < 1)).all():
        preds = sigmoid(preds)

    return sk_roc_auc_score(y_true=target, y_score=preds, max_fpr=max_fpr)


@pytest.mark.parametrize("inputs", _binary_cases[2:])
@pytest.mark.parametrize("max_fpr", [None, 0.7])
class TestBinaryAUROC(MetricTester):
    """Test function and class for computing AUCROC for binary targets."""

    def test_binary_auroc_functional(self, inputs, max_fpr) -> None:
        """Test function for computing binary AUROC."""
        target, preds = inputs

        self.run_functional_test(
            target=target,
            preds=preds,
            metric_functional=cyclops_auroc,
            sk_metric=partial(_sk_binary_auroc, max_fpr=max_fpr),
            metric_args={"task": "binary", "max_fpr": max_fpr},
        )

    def test_binary_auroc_classl(self, inputs, max_fpr) -> None:
        """Test class for computing binary AUROC."""
        target, preds = inputs

        self.run_class_test(
            target=target,
            preds=preds,
            metric_class=AUROC,
            sk_metric=partial(_sk_binary_auroc, max_fpr=max_fpr),
            metric_args={"task": "binary", "max_fpr": max_fpr},
        )


def _sk_multiclass_auroc(
    target: np.ndarray,
    preds: np.ndarray,
    average: Literal["macro", "weighted"] = "macro",
) -> float:
    """Compute AUROC for multiclass case using sklearn."""
    if not ((preds > 0) & (preds < 1)).all():
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


@pytest.mark.parametrize("inputs", _multiclass_cases[2:])
@pytest.mark.parametrize("average", ["macro", "weighted"])
@pytest.mark.filterwarnings("ignore::UserWarning")  # no positive samples
class TestMulticlassAUROC(MetricTester):
    """Test function and class for computing multiclass AUROC."""

    def test_multiclass_auroc_functional(self, inputs, average) -> None:
        """Test function for computing multiclass AUROC."""
        target, preds = inputs

        self.run_functional_test(
            target=target,
            preds=preds,
            metric_functional=cyclops_auroc,
            sk_metric=partial(_sk_multiclass_auroc, average=average),
            metric_args={
                "task": "multiclass",
                "num_classes": NUM_CLASSES,
                "average": average,
            },
        )

    def test_multiclass_auroc_class(self, inputs, average) -> None:
        """Test class for computing multiclass AUROC."""
        target, preds = inputs

        self.run_class_test(
            target=target,
            preds=preds,
            metric_class=AUROC,
            sk_metric=partial(_sk_multiclass_auroc, average=average),
            metric_args={
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
    if not ((preds > 0) & (preds < 1)).all():
        preds = sigmoid(preds)

    return sk_roc_auc_score(
        target,
        preds,
        average=average,
        max_fpr=None,
        labels=list(range(NUM_LABELS)),
    )


@pytest.mark.parametrize("inputs", _multilabel_cases[2:])
@pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
@pytest.mark.filterwarnings("ignore::UserWarning")  # no positive samples
class TestMultilabelAUROC(MetricTester):
    """Test function and class for computing AUROC for multilabel targets."""

    def test_multilabel_auroc_functional(self, inputs, average) -> None:
        """Test function for computing multilabel AUROC."""
        target, preds = inputs

        self.run_functional_test(
            target=target,
            preds=preds,
            metric_functional=cyclops_auroc,
            sk_metric=partial(_sk_multilabel_auroc, average=average),
            metric_args={
                "task": "multilabel",
                "num_labels": NUM_LABELS,
                "average": average,
            },
        )

    def test_multilabel_auroc_class(self, inputs, average) -> None:
        """Test class for computing multilabel AUROC."""
        target, preds = inputs

        self.run_class_test(
            target=target,
            preds=preds,
            metric_class=AUROC,
            sk_metric=partial(_sk_multilabel_auroc, average=average),
            metric_args={
                "task": "multilabel",
                "num_labels": NUM_LABELS,
                "average": average,
            },
        )
