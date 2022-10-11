"""Functions for computing precision and recall scores on different input types."""
from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics._classification import _prf_divide

from .stat_scores import (
    _binary_stat_scores_update,
    _multiclass_stat_scores_update,
    _multilabel_stat_scores_update,
)


def _precision_recall_reduce(  # pylint: disable=invalid-name, too-many-arguments
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    metric: Literal["precision", "recall"],
    average: Literal["micro", "macro", "weighted", "samples", None],
    sample_weight: Optional[ArrayLike] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> Union[np.ndarray, float]:
    """Compute precision or recall scores and apply specified average.

    Arguements
    -----------
        tp: np.ndarray
            True positives.
        fp: np.ndarray
            False positives.
        fn: np.ndarray
            False negatives.
        metric: Literal["precision", "recall"]
            Metric to compute.
        average: Literal["micro", "macro", "weighted", "samples", None]
            Average to apply. If None, return scores for each class.
        sample_weight: Optional[ArrayLike]
            Sample weights.
        zero_division: Literal["warn", 0, 1]
            Value to return when there are no true positives or true negatives.
            If set to "warn", this acts as 0, but warnings are also raised.

    Returns
    --------
        scores: np.ndarray or float (if average is not None).

    """
    numerator = tp
    different_metric = fp if metric == "precision" else fn
    denominator = numerator + different_metric

    if average == "micro":
        numerator = np.array(np.sum(numerator))
        denominator = np.array(np.sum(denominator))

    modifier = "predicted" if metric == "precision" else "true"

    score = _prf_divide(
        numerator,
        denominator,
        metric,
        modifier,
        average,
        warn_for=metric,
        zero_division=zero_division,
    )

    if average == "weighted":
        weights = tp + fn
        if np.sum(weights) == 0:
            result = np.ones_like(score)
            if zero_division in ["warn", 0]:
                result = np.zeros_like(score)
            return result
    elif average == "samples":
        weights = sample_weight
    else:
        weights = None

    if average is not None and score.ndim != 0 and len(score) > 1:
        result = np.average(score, weights=weights)
    else:
        result = score

    return result


def binary_precision(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    pos_label: int = 1,
    threshold: float = 0.5,
    sample_weight: Optional[ArrayLike] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> float:
    """Compute precision score for binary classification.

    Arguments
    ---------
        target: ArrayLike
            Ground truth (correct) target values.
        preds: ArrayLike
            Predictions as returned by a classifier.
        pos_label: int
            Label of the positive class.
        threshold: float
            Threshold for positive class predictions. Default is 0.5.
        sample_weight: Optional[ArrayLike]
            Sample weights.
        zero_division: Literal["warn", 0, 1]
            Value to return when there are no true positives or true negatives.
            If set to "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
        precision: float

    """
    # pylint: disable=invalid-name
    tp, fp, tn, fn = _binary_stat_scores_update(
        target,
        preds,
        pos_label=pos_label,
        threshold=threshold,
        sample_weight=sample_weight,
    )

    if tp.ndim == 0:
        tp = np.array([tp])
        fp = np.array([fp])
        tn = np.array([tn])
        fn = np.array([fn])

    precision = _precision_recall_reduce(
        tp,
        fp,
        fn,
        metric="precision",
        average=None,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )

    return np.squeeze(precision)


def multiclass_precision(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    num_classes: int,
    top_k: Optional[int] = None,
    sample_weight: Optional[ArrayLike] = None,
    average: Literal["micro", "macro", "weighted", None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> Union[float, np.ndarray]:
    """Compute precision score for multiclass classification.

    Arguments
    ---------
        target: ArrayLike
            Ground truth (correct) target values.
        preds: ArrayLike
            Predictions as returned by a classifier.
        num_classes: int
            Number of classes.
        top_k: Optional[int]
            If not None, calculate precision only on the top k highest
            probability predictions. Default is None. If None, calculate
            precision on all predictions.
        sample_weight: Optional[ArrayLike]
            Sample weights.
        average: Literal["micro", "macro", "weighted", None]
            Average to apply. If None, return scores for each class. Default is
            None. One of:
                - "micro": Calculate metrics globally by counting the total true
                    positives, false negatives and false positives.
                - "macro": Calculate metrics for each label, and find their
                    unweighted mean. This does not take label imbalance into
                    account.
                - "weighted": Calculate metrics for each label, and find their
                    average weighted by support (the number of true instances
                    for each label). This alters "macro" to account for label
                    imbalance.
            Note that "samples" is not supported. Use multilabel_precision
            instead.
        zero_division: Literal["warn", 0, 1]
            Value to return when there are no true positives or true negatives.
            If set to "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
        precision: float or np.ndarray (if average is None).

    """
    # pylint: disable=invalid-name
    if average not in ["micro", "macro", "weighted", None]:
        raise ValueError(
            f"Argument average has to be one of 'micro', 'macro', 'weighted', "
            f"or None, got {average}."
        )

    tp, fp, _, fn = _multiclass_stat_scores_update(
        target,
        preds,
        num_classes,
        sample_weight=sample_weight,
        classwise=True,
        top_k=top_k,
    )

    return _precision_recall_reduce(
        tp,
        fp,
        fn,
        metric="precision",
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )


def multilabel_precision(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    num_labels: int,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    sample_weight: Optional[ArrayLike] = None,
    average: Literal["micro", "macro", "samples", "weighted", None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
):
    """Compute precision score for multilabel classification.

    The input is expected to be an array-like of shape (N, L), where N is the
    number of samples and L is the number of labels. The input is expected to
    be a binary array-like, where 1 indicates the presence of a label and 0
    indicates its absence.

    Arguments
    ---------
        target: ArrayLike
            Ground truth (correct) target values.
        preds: ArrayLike
            Predictions as returned by a classifier.
        num_labels: int
            Number of labels.
        threshold: float
            Threshold for positive class predictions. Default is 0.5.
        top_k: Optional[int]
            If not None, calculate precision only on the top k highest
            probability predictions. Default is None. If None, calculate
            precision on all predictions.
        sample_weight: Optional[ArrayLike]
            Sample weights.
        average: Literal["micro", "macro", "samples", "weighted", None]
            Average to apply. If None, return scores for each class. Default is
            None. One of:
                - "micro": Calculate metrics globally by counting the total true
                    positives, false negatives and false positives.
                - "macro": Calculate metrics for each label, and find their
                    unweighted mean. This does not take label imbalance into
                    account.
                - "weighted": Calculate metrics for each label, and find their
                    average weighted by support (the number of true instances
                    for each label). This alters "macro" to account for label
                    imbalance.
                - "samples": Calculate metrics for each instance, and find their
                    average (only meaningful for multilabel classification).
        zero_division: Literal["warn", 0, 1]
            Value to return when there are no true positives or true negatives.
            If set to "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
        precision: float or np.ndarray (if average is None).

    """
    if average not in ["micro", "macro", "samples", "weighted", None]:
        raise ValueError(
            f"Argument `average` has to be one of 'micro', 'macro', 'samples', "
            f"'weighted', or None, got `{average}.`"
        )

    # pylint: disable=invalid-name
    tp, fp, _, fn = _multilabel_stat_scores_update(
        target,
        preds,
        num_labels,
        top_k=top_k,
        threshold=threshold,
        sample_weight=sample_weight,
        reduce="samples" if average == "samples" else "macro",
    )

    return _precision_recall_reduce(
        tp,
        fp,
        fn,
        metric="precision",
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )


def binary_recall(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    pos_label: int = 1,
    threshold: float = 0.5,
    sample_weight: Optional[ArrayLike] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
):
    """Compute recall score for binary classification.

    Arguments
    ---------
        target: ArrayLike
            Ground truth (correct) target values.
        preds: ArrayLike
            Predictions as returned by a classifier.
        pos_label: int
            Label considered as positive. Default is 1.
        threshold: float
            Threshold for positive class predictions. Default is 0.5.
        sample_weight: Optional[ArrayLike]
            Sample weights.
        zero_division: Literal["warn", 0, 1]
            Value to return when there are no true positives or true negatives.
            If set to "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
        recall: float

    """
    # pylint: disable=invalid-name
    tp, fp, tn, fn = _binary_stat_scores_update(
        target,
        preds,
        pos_label=pos_label,
        threshold=threshold,
        sample_weight=sample_weight,
    )

    if tp.ndim == 0:
        tp = np.expand_dims(tp, axis=-1)
        fp = np.expand_dims(fp, axis=-1)
        tn = np.expand_dims(tn, axis=-1)
        fn = np.expand_dims(fn, axis=-1)

    precision = _precision_recall_reduce(
        tp,
        fp,
        fn,
        metric="recall",
        average=None,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )

    return np.squeeze(precision)


def multiclass_recall(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    num_classes: int,
    top_k: Optional[int] = None,
    sample_weight: Optional[ArrayLike] = None,
    average: Literal["micro", "macro", "weighted", None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
):
    """Compute recall score for multiclass classification.

    Arguments
    ---------
        target: ArrayLike
            Ground truth (correct) target values.
        preds: ArrayLike
            Predictions as returned by a classifier.
        num_classes: int
            Number of classes.
        top_k: Optional[int]
            If not None, calculate recall only on the top k highest
            probability predictions. Default is None. If None, calculate
            recall on all predictions.
        sample_weight: Optional[ArrayLike]
            Sample weights.
        average: Literal["micro", "macro", "weighted", None]
            Average to apply. If None, return scores for each class. Default is
            None. One of:
                - "micro": Calculate metrics globally by counting the total true
                    positives, false negatives and false positives.
                - "macro": Calculate metrics for each label, and find their
                    unweighted mean. This does not take label imbalance into
                    account.
                - "weighted": Calculate metrics for each label, and find their
                    average weighted by support (the number of true instances
                    for each label). This alters "macro" to account for label
                    imbalance.
        zero_division: Literal["warn", 0, 1]
            Value to return when there are no true positives or true negatives.
            If set to "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
        recall: float or np.ndarray (if average is None).

    """
    if average not in ["micro", "macro", "weighted", None]:
        raise ValueError(
            f"Argument `average` has to be one of 'micro', 'macro', 'weighted', "
            f"or None, got `{average}`."
        )

    # pylint: disable=invalid-name
    tp, fp, _, fn = _multiclass_stat_scores_update(
        target,
        preds,
        num_classes,
        sample_weight=sample_weight,
        classwise=True,
        top_k=top_k,
    )

    return _precision_recall_reduce(
        tp,
        fp,
        fn,
        metric="recall",
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )


def multilabel_recall(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    num_labels: int,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    sample_weight: Optional[ArrayLike] = None,
    average: Literal["micro", "macro", "samples", "weighted", None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
):
    """Compute recall score for multilabel classification.

    The input is expected to be an array-like of shape (N, L), where N is the
    number of samples and L is the number of labels. The input is expected to
    be a binary array-like, where 1 indicates the presence of a label and 0
    indicates its absence.

    Arguments
    ---------
        target: ArrayLike
            Ground truth (correct) target values.
        preds: ArrayLike
            Predictions as returned by a classifier.
        num_labels: int
            Number of labels.
        threshold: float
            Threshold for positive class predictions. Default is 0.5.
        top_k: Optional[int]
            If not None, calculate recall only on the top k highest
            probability predictions. Default is None. If None, calculate
            recall on all predictions.
        sample_weight: Optional[ArrayLike]
            Sample weights.
        average: Literal["micro", "macro", "samples", "weighted", None]
            Average to apply. If None, return scores for each class. Default is
            None. One of:
                - "micro": Calculate metrics globally by counting the total true
                    positives, false negatives and false positives.
                - "macro": Calculate metrics for each label, and find their
                    unweighted mean. This does not take label imbalance into
                    account.
                - "samples": Calculate metrics for each instance, and find their
                    average (only meaningful for multilabel classification where
                    this differs from accuracy_score).
                - "weighted": Calculate metrics for each label, and find their
                    average weighted by support (the number of true instances
                    for each label). This alters "macro" to account for label
                    imbalance.
        zero_division: Literal["warn", 0, 1]
            Value to return when there are no true positives or true negatives.
            If set to "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
        recall: float or np.ndarray (if average is None).

    """
    if average not in ["micro", "macro", "samples", "weighted", None]:
        raise ValueError(
            f"Argument average has to be one of 'micro', 'macro', 'samples', "
            f"'weighted', or None, got {average}."
        )

    # pylint: disable=invalid-name
    tp, fp, _, fn = _multilabel_stat_scores_update(
        target,
        preds,
        num_labels,
        top_k=top_k,
        threshold=threshold,
        sample_weight=sample_weight,
        reduce="samples" if average == "samples" else "macro",
    )

    return _precision_recall_reduce(
        tp,
        fp,
        fn,
        metric="recall",
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )
