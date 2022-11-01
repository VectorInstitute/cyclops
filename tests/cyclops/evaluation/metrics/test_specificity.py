"""Functions for testing specificity metrics."""
from functools import partial
from typing import Literal, Union

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
from metrics.test_stat_scores import (
    _sk_stat_scores_binary,
    _sk_stat_scores_multiclass,
    _sk_stat_scores_multilabel,
)
from numpy.typing import ArrayLike

from cyclops.evaluation.metrics.functional.specificity import specificity


def _reduce_specificity(
    tn: np.ndarray,
    fp: np.ndarray,
    support: np.ndarray,
    average: Literal["micro", "macro", "weighted", "samples", None],
    sample_weight: ArrayLike = None,
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
        average : Literal["micro", "macro", "weighted", "samples", None]
            If not None, return the average specificity.
        sample_weight : ArrayLike, optional
            Sample weights, by default None

    Returns
    -------
        specificity : float or np.ndarray (if average is None).

    """
    # pylint: disable=invalid-name
    if average == "micro":
        return _calc_specificity(tn.sum(), fp.sum())

    res = _calc_specificity(tn, fp)
    if average is not None:
        weights = None
        if average == "weighted":
            weights = support
        elif average == "samples":
            weights = sample_weight

        if weights is not None and np.sum(weights) == 0:
            return np.zeros_like(res)

        return np.average(res, weights=weights)

    return res


def _calc_specificity(  # pylint: disable=invalid-name
    tn: Union[int, np.ndarray], fp: Union[int, np.ndarray]
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
    # pylint: disable=invalid-name
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
    target: np.ndarray, preds: np.ndarray, threshold: float
) -> np.ndarray:
    """Compute specificity for binary case using sklearn."""
    # pylint: disable=invalid-name
    _, fp, tn, _, _ = _sk_stat_scores_binary(target, preds, threshold=threshold)

    return _calc_specificity(tn, fp)


@pytest.mark.parametrize("inputs", _binary_cases)
def test_binary_specificity(inputs):
    """Test binary case."""
    target, preds = inputs

    # test functional
    _functional_test(
        target,
        preds,
        specificity,
        partial(_sk_binary_specificity, threshold=THRESHOLD),
        {"task": "binary", "threshold": THRESHOLD, "zero_division": 0},
    )


def _sk_multiclass_specificity(
    target: np.ndarray,
    preds: np.ndarray,
    average: Literal["micro", "macro", "weighted", None],
) -> np.ndarray:
    """Compute specificity for multiclass case using sklearn."""
    # pylint: disable=invalid-name
    scores = _sk_stat_scores_multiclass(target, preds, classwise=True)
    tn = scores[:, 2]
    fp = scores[:, 1]
    support = scores[:, 4]

    return _reduce_specificity(tn, fp, support, average)


@pytest.mark.parametrize("inputs", _multiclass_cases)
@pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
def test_multiclass_specificity(inputs, average):
    """Test multiclass case."""
    target, preds = inputs

    # test functional
    _functional_test(
        target,
        preds,
        specificity,
        partial(_sk_multiclass_specificity, average=average),
        {
            "task": "multiclass",
            "num_classes": NUM_CLASSES,
            "average": average,
            "zero_division": 0,
        },
    )


def _sk_multilabel_specificity(
    target: np.ndarray,
    preds: np.ndarray,
    average: Literal["micro", "macro", "weighted", "samples", None],
    sample_weight: ArrayLike = None,
) -> np.ndarray:
    """Compute specificity for multilabel case using sklearn."""
    # pylint: disable=invalid-name
    scores = _sk_stat_scores_multilabel(
        target,
        preds,
        threshold=THRESHOLD,
        reduce="samples" if average == "samples" else "macro",
    )
    tn = scores[:, 2]
    fp = scores[:, 1]
    support = scores[:, 4]

    return _reduce_specificity(tn, fp, support, average, sample_weight)


@pytest.mark.parametrize("inputs", _multilabel_cases)
@pytest.mark.parametrize("average", ["micro", "macro", "samples", "weighted", None])
def test_multilabel_specificity(inputs, average):
    """Test multilabel case."""
    target, preds = inputs

    # test functional
    _functional_test(
        target,
        preds,
        specificity,
        partial(_sk_multilabel_specificity, average=average),
        {
            "task": "multilabel",
            "num_labels": NUM_LABELS,
            "threshold": THRESHOLD,
            "average": average,
            "zero_division": 0,
        },
    )
