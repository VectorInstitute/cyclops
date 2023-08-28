"""Functions for test F-beta and F1 score."""

from functools import partial
from typing import Literal

import numpy as np
import pytest
from sklearn.metrics import fbeta_score as sk_fbeta_score

from cyclops.evaluate.metrics.f_beta import F1Score, FbetaScore
from cyclops.evaluate.metrics.functional.f_beta import f1_score, fbeta_score
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


def _sk_binary_fbeta_score(
    target: np.ndarray,
    preds: np.ndarray,
    beta: float,
    threshold: float,
    zero_division: Literal["warn", 0, 1],
):
    """Compute fbeta score for binary case using sklearn."""
    if np.issubdtype(preds.dtype, np.floating):
        if not ((preds > 0) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= threshold).astype(np.uint8)

    return sk_fbeta_score(
        y_true=target,
        y_pred=preds,
        beta=beta,
        zero_division=zero_division,
    )


@pytest.mark.parametrize("inputs", _binary_cases)
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
class TestBinaryFbetaScore(MetricTester):
    """Test binary F-beta function and class."""

    def test_binary_fbeta_score_functional(self, inputs, beta) -> None:
        """Test binary F-beta function."""
        target, preds = inputs

        self.run_functional_test(
            target=target,
            preds=preds,
            metric_functional=f1_score
            if beta == 1.0
            else partial(
                fbeta_score,
                beta=beta,
                task="binary",
            ),
            sk_metric=partial(
                _sk_binary_fbeta_score,
                beta=beta,
                threshold=THRESHOLD,
                zero_division=0,
            ),
            metric_args={"task": "binary", "threshold": THRESHOLD, "zero_division": 0},
        )

    def test_binary_fbeta_score_class(self, inputs, beta) -> None:
        """Test class binary F-beta score."""
        target, preds = inputs

        self.run_class_test(
            target=target,
            preds=preds,
            metric_class=F1Score
            if beta == 1.0
            else partial(
                FbetaScore,
                beta=beta,
                task="binary",
            ),
            sk_metric=partial(
                _sk_binary_fbeta_score,
                beta=beta,
                threshold=THRESHOLD,
                zero_division=0,
            ),
            metric_args={"task": "binary", "threshold": THRESHOLD, "zero_division": 0},
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
class TestMulticlassFbetaScore(MetricTester):
    """Test multiclass F-beta function and class."""

    def test_multiclass_fbeta_functional(self, inputs, beta, average) -> None:
        """Test multiclass F-beta function."""
        target, preds = inputs

        self.run_functional_test(
            target=target,
            preds=preds,
            metric_functional=f1_score
            if beta == 1.0
            else partial(fbeta_score, beta=beta),
            sk_metric=partial(
                _sk_multiclass_fbeta_score,
                beta=beta,
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

    def test_multiclass_fbeta_class(self, inputs, beta, average) -> None:
        """Test class for multiclass F-beta score."""
        target, preds = inputs

        self.run_class_test(
            target=target,
            preds=preds,
            metric_class=F1Score if beta == 1.0 else partial(FbetaScore, beta=beta),
            sk_metric=partial(
                _sk_multiclass_fbeta_score,
                beta=beta,
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


def _sk_multilabel_fbeta_score(
    target: np.ndarray,
    preds: np.ndarray,
    beta: float,
    threshold: float,
    average: Literal["micro", "macro", "weighted"],
    zero_division: Literal["warn", 0, 1],
):
    """Compute fbeta score for multilabel case using sklearn."""
    if np.issubdtype(preds.dtype, np.floating):
        if not ((preds > 0) & (preds < 1)).all():
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
class TestMultilabelFbetaScore(MetricTester):
    """Test function and class for multilabel F-beta score."""

    def test_multilabel_fbeta_score_functional(self, inputs, beta, average) -> None:
        """Test function for multilabel F-beta score."""
        target, preds = inputs

        self.run_functional_test(
            target=target,
            preds=preds,
            metric_functional=f1_score
            if beta == 1.0
            else partial(fbeta_score, beta=beta),
            sk_metric=partial(
                _sk_multilabel_fbeta_score,
                beta=beta,
                average=average,
                threshold=THRESHOLD,
                zero_division=0,
            ),
            metric_args={
                "task": "multilabel",
                "num_labels": NUM_LABELS,
                "average": average,
                "threshold": THRESHOLD,
                "zero_division": 0,
            },
        )

    def test_multilabel_fbeta_score_class(self, inputs, beta, average) -> None:
        """Test class for multilabel F-beta score."""
        target, preds = inputs

        self.run_class_test(
            target=target,
            preds=preds,
            metric_class=F1Score if beta == 1.0 else partial(FbetaScore, beta=beta),
            sk_metric=partial(
                _sk_multilabel_fbeta_score,
                beta=beta,
                average=average,
                threshold=THRESHOLD,
                zero_division=0,
            ),
            metric_args={
                "task": "multilabel",
                "num_labels": NUM_LABELS,
                "average": average,
                "threshold": THRESHOLD,
                "zero_division": 0,
            },
        )
