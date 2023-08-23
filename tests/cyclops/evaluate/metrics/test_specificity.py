"""Functions for testing specificity metrics."""

from functools import partial
from typing import Literal, Union

import numpy as np
import pytest

from cyclops.evaluate.metrics.functional.specificity import specificity
from cyclops.evaluate.metrics.specificity import Specificity
from metrics.helpers import MetricTester
from metrics.inputs import (
    NUM_CLASSES,
    NUM_LABELS,
    THRESHOLD,
    _binary_cases,
    _multiclass_cases,
    _multilabel_cases,
)
from metrics.test_stat_scores import (
    _sk_stat_scores_binary,
    _sk_stat_scores_multiclass,
    _sk_stat_scores_multilabel,
)


def _reduce_specificity(
    tn: np.ndarray,
    fp: np.ndarray,
    support: np.ndarray,
    average: Literal["micro", "macro", "weighted", None],
) -> Union[float, np.ndarray]:
    """Compute specificity and apply `average`.

    Parameters
    ----------
    tn : np.ndarray
        True negatives.
    fp : np.ndarray
        False positives.
    support : np.ndarray
        Support. Number of true instances for each class.
    average : Literal["micro", "macro", "weighted", None]
        If not None, return the average specificity.

    Returns
    -------
    specificity : float or np.ndarray (if average is None).

    """
    if average == "micro":
        return _calc_specificity(tn.sum(), fp.sum())

    res = _calc_specificity(tn, fp)
    if average is not None:
        weights = None
        if average == "weighted":
            weights = support

        if weights is not None and np.sum(weights) == 0:
            return np.zeros_like(res)

        return np.average(res, weights=weights)

    return res


def _calc_specificity(
    tn: Union[int, np.ndarray],
    fp: Union[int, np.ndarray],
) -> Union[float, np.ndarray]:
    """Calculate specificity.

    Returns 0 if both `tn` and `fp` are 0.

    Parameters
    ----------
    tn : np.ndarray or int
        True negatives.
    fp : np.ndarray or int
        False positives.

    Returns
    -------
    specificity : np.ndarray

    """
    denominator = tn + fp
    if np.isscalar(tn):
        if denominator == 0.0:
            return 0.0
        return tn / denominator

    mask = denominator == 0.0
    denominator[mask] = 1.0  # type: ignore

    result: np.ndarray = tn / denominator

    result[mask] = 0.0  # type: ignore

    return result


def _sk_binary_specificity(
    target: np.ndarray,
    preds: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Compute specificity for binary case using sklearn."""
    _, fp, tn, _, _ = _sk_stat_scores_binary(target, preds, threshold=threshold)

    return _calc_specificity(tn, fp)


@pytest.mark.parametrize("inputs", _binary_cases)
class TestBinarySpecificity(MetricTester):
    """Test class and function for binary specificity metric."""

    def test_binary_specificity_functional(self, inputs) -> None:
        """Test function for binary specificity."""
        target, preds = inputs

        self.run_functional_test(
            target=target,
            preds=preds,
            metric_functional=specificity,
            sk_metric=partial(_sk_binary_specificity, threshold=THRESHOLD),
            metric_args={"task": "binary", "threshold": THRESHOLD, "zero_division": 0},
        )

    def test_binary_specificity_class(self, inputs) -> None:
        """Test class for binary specificity."""
        target, preds = inputs

        self.run_class_test(
            target=target,
            preds=preds,
            metric_class=Specificity,
            sk_metric=partial(_sk_binary_specificity, threshold=THRESHOLD),
            metric_args={"task": "binary", "threshold": THRESHOLD, "zero_division": 0},
        )


def _sk_multiclass_specificity(
    target: np.ndarray,
    preds: np.ndarray,
    average: Literal["micro", "macro", "weighted", None],
) -> np.ndarray:
    """Compute specificity for multiclass case using sklearn."""
    scores = _sk_stat_scores_multiclass(target, preds, classwise=True)
    tn = scores[:, 2]
    fp = scores[:, 1]
    support = scores[:, 4]

    return _reduce_specificity(tn, fp, support, average)


@pytest.mark.parametrize("inputs", _multiclass_cases)
@pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
class TestMulticlassSpecificity(MetricTester):
    """Test function and class for multiclass specificity metric."""

    def test_multiclass_specificity_functional(self, inputs, average) -> None:
        """Test function for multiclass specificity metric."""
        target, preds = inputs

        self.run_functional_test(
            target=target,
            preds=preds,
            metric_functional=specificity,
            sk_metric=partial(_sk_multiclass_specificity, average=average),
            metric_args={
                "task": "multiclass",
                "num_classes": NUM_CLASSES,
                "average": average,
                "zero_division": 0,
            },
        )

    def test_multiclass_specificity_class(self, inputs, average) -> None:
        """Test class for multiclass specificity metric."""
        target, preds = inputs

        self.run_class_test(
            target=target,
            preds=preds,
            metric_class=Specificity,
            sk_metric=partial(_sk_multiclass_specificity, average=average),
            metric_args={
                "task": "multiclass",
                "num_classes": NUM_CLASSES,
                "average": average,
                "zero_division": 0,
            },
        )


def _sk_multilabel_specificity(
    target: np.ndarray,
    preds: np.ndarray,
    average: Literal["micro", "macro", "weighted", None],
) -> np.ndarray:
    """Compute specificity for multilabel case using sklearn."""
    scores = _sk_stat_scores_multilabel(
        target,
        preds,
        threshold=THRESHOLD,
        labelwise=True,
    )
    tn = scores[:, 2]
    fp = scores[:, 1]
    support = scores[:, 4]

    return _reduce_specificity(tn, fp, support, average)


@pytest.mark.parametrize("inputs", _multilabel_cases)
@pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
class TestMultilabelSpecificity(MetricTester):
    """Test function and class for multilabel specificity metric."""

    def test_multilabel_specificity_functional(self, inputs, average) -> None:
        """Test function for multilabel specificity."""
        target, preds = inputs

        self.run_functional_test(
            target=target,
            preds=preds,
            metric_functional=specificity,
            sk_metric=partial(_sk_multilabel_specificity, average=average),
            metric_args={
                "task": "multilabel",
                "num_labels": NUM_LABELS,
                "threshold": THRESHOLD,
                "average": average,
                "zero_division": 0,
            },
        )

    def test_multilabel_specificity_class(self, inputs, average) -> None:
        """Test class for multilabel specificity."""
        target, preds = inputs

        self.run_class_test(
            target=target,
            preds=preds,
            metric_class=Specificity,
            sk_metric=partial(_sk_multilabel_specificity, average=average),
            metric_args={
                "task": "multilabel",
                "num_labels": NUM_LABELS,
                "threshold": THRESHOLD,
                "average": average,
                "zero_division": 0,
            },
        )
