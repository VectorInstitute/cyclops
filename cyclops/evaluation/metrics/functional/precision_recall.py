"""Functions for computing precision and recall scores on different input types."""
from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics._classification import _prf_divide

from cyclops.evaluation.metrics.functional.stat_scores import (
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

    Parameters
    ----------
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
    -------
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
        np.array(numerator) if np.isscalar(tp) else numerator,
        np.array(denominator) if np.isscalar(tp) else denominator,
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


def precision(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    task: Literal["binary", "multiclass", "multilabel"],
    pos_label: int = 1,
    num_classes: Optional[int] = None,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    num_labels: Optional[int] = None,
    average: Literal["micro", "macro", "weighted", "samples", None] = None,
    sample_weight: Optional[ArrayLike] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> Union[float, np.ndarray]:
    """Compute precision score for different classification tasks.

    Precision is the ratio of correctly predicted positive observations to the
    total predicted positive observations.

    Parameters
    ----------
        target: ArrayLike
            Ground truth (correct) target values.
        preds: ArrayLike
            Predictions as returned by a classifier.
        task: Literal["binary", "multiclass", "multilabel"]
            Classification task. One of:
                - ``binary``: binary classification.
                    Example: [0, 1, 1, 0, 1] or [0.1, 0.9, 0.8, 0.2, 0.4]
                - ``multiclass``: multiclass classification.
                    Example: [0, 1, 2, 0, 1] or [[0.1, 0.9, 0.0], [0.0, 0.8, 0.2], ...]
                - ``multilabel``: multilabel classification.
                    Example: [[0, 1], [1, 0], [1, 1], [0, 0], [1, 0]] or
                    [[0.1, 0.9], [0.0, 0.8], ...]
        pos_label: int
            Label of the positive class. Only used for binary classification.
        num_classes: Optional[int]
            Number of classes. Only used for multiclass classification.
        threshold: float
            Threshold for positive class predictions. Default is 0.5.
        top_k: Optional[int]
            Number of highest probability or logits predictions to consider when
            computing multiclass or multilabel metrics. Default is None.
        num_labels: Optional[int]
            Number of labels. Only used for multilabel classification.
        average: Literal["micro", "macro", "weighted", "samples", None]
            Average to apply. If None, return scores for each class. Default is
            None.
            One of:
                - ``micro``: Calculate metrics globally by counting the total true
                  positives, false negatives and false positives.
                - ``macro``: Calculate metrics for each label, and find their
                  unweighted mean. This does not take label imbalance into account.
                - ``weighted``: Calculate metrics for each label, and find their
                  average weighted by support (the number of true instances for
                  each label). This alters ``macro`` to account for label imbalance.
                - ``samples``: Calculate metrics for each instance, and find their
                  average (only meaningful for multilabel classification).
        sample_weight: Optional[ArrayLike]
            Sample weights.
        zero_division: Literal["warn", 0, 1]
            Value to return when there are no true positives or true negatives.
            If set to ``warn``, this acts as 0, but warnings are also raised.

    Returns
    -------
        precision: float (if ``average`` is None or ``task`` is ``binary``) or
        np.ndarray (if ``average`` is not None).

    Raises
    ------
        ValueError
            If task is not one of ``binary``, ``multiclass`` or ``multilabel``.

    """
    if task == "binary":
        precision_score = binary_precision(
            target,
            preds,
            pos_label=pos_label,
            threshold=threshold,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )

    elif task == "multiclass":
        assert (
            isinstance(num_classes, int) and num_classes > 0
        ), "Number of classes must be specified for multiclass classification."
        precision_score = multiclass_precision(
            target,
            preds,
            num_classes=num_classes,
            average=average,  # type: ignore
            top_k=top_k,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )

    elif task == "multilabel":
        assert (
            isinstance(num_labels, int) and num_labels > 0
        ), "Number of labels must be specified for multilabel classification."
        precision_score = multilabel_precision(
            target,
            preds,
            num_labels=num_labels,
            threshold=threshold,
            average=average,
            top_k=top_k,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )

    else:
        raise ValueError(
            f"Task '{task}' not supported, expected 'binary', 'multiclass' or "
            f"'multilabel'."
        )

    return precision_score


def binary_precision(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    pos_label: int = 1,
    threshold: float = 0.5,
    sample_weight: Optional[ArrayLike] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> float:
    """Compute precision score for binary classification.

    Parameters
    ----------
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
            If set to ``warn``, this acts as 0, but warnings are also raised.

    Returns
    -------
        precision: float

    """
    # pylint: disable=invalid-name
    tp, fp, _, fn = _binary_stat_scores_update(
        target,
        preds,
        pos_label=pos_label,
        threshold=threshold,
        sample_weight=sample_weight,
    )

    precision_score = _precision_recall_reduce(
        tp,
        fp,
        fn,
        metric="precision",
        average=None,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )

    return np.squeeze(precision_score)


def multiclass_precision(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    num_classes: int,
    top_k: Optional[int] = None,
    average: Literal["micro", "macro", "weighted", None] = None,
    sample_weight: Optional[ArrayLike] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> Union[float, np.ndarray]:
    """Compute precision score for multiclass classification.

    Parameters
    ----------
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
        average: Literal["micro", "macro", "weighted", None]
            Average to apply. If None, return scores for each class. Default is
            None. One of:
                - ``micro``: Calculate metrics globally by counting the total true
                  positives, false negatives and false positives.
                - ``macro``: Calculate metrics for each label, and find their
                  unweighted mean. This does not take label imbalance into account.
                - ``weighted``: Calculate metrics for each label, and find their
                  average weighted by support (the number of true instances
                  for each label). This alters "macro" to account for label
                  imbalance.

            Note that ``samples`` is not supported. Use multilabel_precision
            instead.
        sample_weight: Optional[ArrayLike]
            Sample weights.
        zero_division: Literal["warn", 0, 1]
            Value to return when there are no true positives or true negatives.
            If set to ``warn``, this acts as 0, but warnings are also raised.

    Returns
    -------
        precision: float or np.ndarray (if average is None).

    Raises
    ------
        ValueError
            If ``average`` is not one of ``micro``, ``macro``, ``weighted``
            or ``None``.

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
    average: Literal["micro", "macro", "samples", "weighted", None] = None,
    sample_weight: Optional[ArrayLike] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
):
    """Compute precision score for multilabel classification.

    The input is expected to be an array-like of shape (N, L), where N is the
    number of samples and L is the number of labels. The input is expected to
    be a binary array-like, where 1 indicates the presence of a label and 0
    indicates its absence.

    Parameters
    ----------
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
        average: Literal["micro", "macro", "samples", "weighted", None]
            Average to apply. If None, return scores for each class. Default is
            None. One of:
                - ``micro``: Calculate metrics globally by counting the total true
                  positives, false negatives and false positives.
                - ``macro``: Calculate metrics for each label, and find their
                  unweighted mean. This does not take label imbalance into
                  account.
                - ``weighted``: Calculate metrics for each label, and find their
                  average weighted by support (the number of true instances
                  for each label). This alters ``macro`` to account for label
                  imbalance.
                - ``samples``: Calculate metrics for each instance, and find their
                  average (only meaningful for multilabel classification).
        sample_weight: Optional[ArrayLike]
            Sample weights.
        zero_division: Literal["warn", 0, 1]
            Value to return when there are no true positives or true negatives.
            If set to ``warn``, this acts as 0, but warnings are also raised.

    Returns
    -------
        precision: float or np.ndarray (if ``average`` is None).

    Raises
    ------
        ValueError
            If average is not one of ``micro``, ``macro``, ``samples``, ``weighted``,
            or ``None``.

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


def recall(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    task: Literal["binary", "multiclass", "multilabel"],
    pos_label: int = 1,
    num_classes: int = None,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    num_labels: int = None,
    average: Literal["micro", "macro", "weighted", "samples", None] = None,
    sample_weight: Optional[ArrayLike] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> Union[float, np.ndarray]:
    """Compute recall score for different classification tasks.

    Recall is the ratio tp / (tp + fn) where tp is the number of true positives
    and fn the number of false negatives. The recall is intuitively the ability
    of the classifier to find all the positive samples.

    Parameters
    ----------
        target: ArrayLike
            Ground truth (correct) target values.
        preds: ArrayLike
            Predictions as returned by a classifier.
        task: Literal["binary", "multiclass", "multilabel"]
            Classification task. One of:
                - ``binary``: binary classification.
                    Example: [0, 1, 1, 0, 1] or [0.1, 0.9, 0.8, 0.2, 0.4]
                - ``multiclass``: multiclass classification.
                    Example: [0, 1, 2, 0, 1] or [[0.1, 0.9, 0.0], [0.0, 0.8, 0.2], ...]
                - ``multilabel``: multilabel classification.
                    Example: [[0, 1], [1, 0], [1, 1], [0, 0], [1, 0]] or
                    [[0.1, 0.9], [0.0, 0.8], ...]
        pos_label: int
            Label of the positive class. Only used for binary classification.
        num_classes: Optional[int]
            Number of classes. Only used for multiclass classification.
        threshold: float
            Threshold for positive class predictions. Default is 0.5.
        top_k: Optional[int]
            Number of highest probability or logits predictions to consider when
            computing multiclass or multilabel metrics. Default is None.
        num_labels: Optional[int]
            Number of labels. Only used for multilabel classification.
        average: Literal["micro", "macro", "weighted", "samples", None]
            Average to apply. If None, return scores for each class. Default is
            None.
            One of:
                - ``micro``: Calculate metrics globally by counting the total true
                  positives, false negatives and false positives.
                - ``macro``: Calculate metrics for each label, and find their
                  unweighted mean. This does not take label imbalance into account.
                - ``weighted``: Calculate metrics for each label, and find their
                  average weighted by support (the number of true instances for
                  each label). This alters ``macro`` to account for label imbalance.
                - ``samples``: Calculate metrics for each instance, and find their
                  average (only meaningful for multilabel classification).
        sample_weight: Optional[ArrayLike]
            Sample weights.
        zero_division: Literal["warn", 0, 1]
            Value to return when there are no true positives or true negatives.
            If set to ``warn``, this acts as 0, but warnings are also raised.

    Returns
    -------
        recall: float (if ``average`` is None or ``task`` is ``binary``) or np.ndarray
        (if ``average`` is not None).

    Raises
    ------
        ValueError
            If ``task`` is not one of ``binary``, ``multiclass`` or ``multilabel``.

    """
    if task == "binary":
        recall_score = binary_recall(
            target,
            preds,
            pos_label=pos_label,
            threshold=threshold,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )

    elif task == "multiclass":
        assert (
            isinstance(num_classes, int) and num_classes > 0
        ), "Number of classes must be specified for multiclass classification."
        recall_score = multiclass_recall(
            target,
            preds,
            num_classes=num_classes,
            average=average,  # type: ignore
            top_k=top_k,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )

    elif task == "multilabel":
        assert (
            isinstance(num_labels, int) and num_labels > 0
        ), "Number of labels must be specified for multilabel classification."
        recall_score = multilabel_recall(
            target,
            preds,
            num_labels=num_labels,
            threshold=threshold,
            average=average,
            top_k=top_k,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )

    else:
        raise ValueError(
            f"Task '{task}' not supported, expected 'binary', 'multiclass' or "
            f"'multilabel'."
        )

    return recall_score


def binary_recall(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    pos_label: int = 1,
    threshold: float = 0.5,
    sample_weight: Optional[ArrayLike] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
):
    """Compute recall score for binary classification.

    Parameters
    ----------
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
            If set to ``warn``, this acts as 0, but warnings are also raised.

    Returns
    -------
        recall: float

    """
    # pylint: disable=invalid-name
    tp, fp, _, fn = _binary_stat_scores_update(
        target,
        preds,
        pos_label=pos_label,
        threshold=threshold,
        sample_weight=sample_weight,
    )

    recall_score = _precision_recall_reduce(
        tp,
        fp,
        fn,
        metric="recall",
        average=None,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )

    return np.squeeze(recall_score)


def multiclass_recall(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    num_classes: int,
    top_k: Optional[int] = None,
    average: Literal["micro", "macro", "weighted", None] = None,
    sample_weight: Optional[ArrayLike] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
):
    """Compute recall score for multiclass classification.

    Parameters
    ----------
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
        average: Literal["micro", "macro", "weighted", None]
            Average to apply. If None, return scores for each class. Default is
            None. One of:
                - ``micro``: Calculate metrics globally by counting the total true
                  positives, false negatives and false positives.
                - ``macro``: Calculate metrics for each label, and find their
                  unweighted mean. This does not take label imbalance into
                  account.
                - ``weighted``: Calculate metrics for each label, and find their
                  average weighted by support (the number of true instances
                  for each label). This alters "macro" to account for label
                  imbalance.

        sample_weight: Optional[ArrayLike]
            Sample weights.
        zero_division: Literal["warn", 0, 1]
            Value to return when there are no true positives or true negatives.
            If set to ``warn``, this acts as 0, but warnings are also raised.

    Returns
    -------
        recall: float or np.ndarray (if average is None).

    Raises
    ------
        ValueError
            If ``average`` is not one of ``micro``, ``macro``, ``weighted``
            or ``None``.

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
    average: Literal["micro", "macro", "samples", "weighted", None] = None,
    sample_weight: Optional[ArrayLike] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
):
    """Compute recall score for multilabel classification.

    The input is expected to be an array-like of shape (N, L), where N is the
    number of samples and L is the number of labels. The input is expected to
    be a binary array-like, where 1 indicates the presence of a label and 0
    indicates its absence.

    Parameters
    ----------
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
        average: Literal["micro", "macro", "samples", "weighted", None]
            Average to apply. If None, return scores for each class. Default is
            None. One of:
                - ``micro``: Calculate metrics globally by counting the total true
                  positives, false negatives and false positives.
                - ``macro``: Calculate metrics for each label, and find their
                  unweighted mean. This does not take label imbalance into
                  account.
                - ``samples``: Calculate metrics for each instance, and find their
                  average (only meaningful for multilabel classification where
                  this differs from accuracy_score).
                - ``weighted``: Calculate metrics for each label, and find their
                  average weighted by support (the number of true instances
                  for each label). This alters ``macro`` to account for label
                  imbalance.
        sample_weight: Optional[ArrayLike]
            Sample weights.
        zero_division: Literal["warn", 0, 1]
            Value to return when there are no true positives or true negatives.
            If set to ``warn``, this acts as 0, but warnings are also raised.

    Returns
    -------
        recall: float or np.ndarray (if ``average`` is None).

    Raises
    ------
        ValueError
            If ``average`` is not one of ``micro``, ``macro``, ``samples``, ``weighted``
            or ``None``.

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
