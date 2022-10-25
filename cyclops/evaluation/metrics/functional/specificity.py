"""Functions to compute the specificity metric."""
from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics._classification import _prf_divide

from cyclops.evaluation.metrics.functional.stat_scores import (
    _binary_stat_scores_update,
    _multiclass_stat_scores_update,
    _multilabel_stat_scores_update,
)


def _specificity_reduce(  # pylint: disable=too-many-arguments, invalid-name
    tp: Union[int, np.ndarray],
    fp: Union[int, np.ndarray],
    tn: Union[int, np.ndarray],
    fn: Union[int, np.ndarray],
    average: Literal["micro", "macro", "weighted", "samples", None],
    sample_weight: Optional[ArrayLike] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> Union[float, np.ndarray]:
    """Reduce specificity.

    Parameters
    ----------
        tp : np.ndarray or int
            True positives.
        fp : np.ndarray or int
            False positives.
        tn : np.ndarray or int
            True negatives.
        fn : np.ndarray or int
            False negatives.
        average : Literal["micro", "macro", "weighted", "samples", None]
            If None, return the specificity for each class, otherwise return the
            average specificity. Average options are:
                - ``micro``: Calculate metrics globally by counting the total
                  true positives, false negatives and false positives.
                - ``macro``: Calculate metrics for each label, and find their unweighted
                  mean. This does not take label imbalance into account.
                - ``weighted``: Calculate metrics for each label, and find their
                  average, weighted by support (the number of true instances for
                  each label).
                - ``samples``: Calculate metrics for each instance, and find their
                  average.
        sample_weight : Optional[ArrayLike]
            Sample weights.
        zero_division : Literal["warn", 0, 1]
            Sets the value to return when there is a zero division. If set to ``warn``,
            this acts as 0, but warnings are also raised.

    Returns
    -------
        specificity : float or np.ndarray (if average is None).

    """
    numerator = tn
    denominator = tn + fp

    if average == "micro":
        numerator = np.array(np.sum(numerator))
        denominator = np.array(np.sum(denominator))

    score = _prf_divide(
        numerator,
        denominator,
        metric="specificity",
        modifier="score",
        average=average,
        warn_for=("specificity",),
        zero_division=zero_division,
    )

    if average == "weighted":
        weights = tp + fn
    elif average == "samples":
        weights = sample_weight
    else:
        weights = None

    if weights is not None and np.sum(weights) == 0:
        result = np.ones_like(score)
        if zero_division in ["warn", 0]:
            result = np.zeros_like(score)
        return result

    if average is not None and score.ndim != 0 and len(score) > 1:
        result = np.average(score, weights=weights)
    else:
        result = score

    return result


def specificity(  # pylint: disable=too-many-arguments
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
    """Compute specificity.

    The specificity is the ratio of true negatives to the sum of true negatives and
    false positives.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Estimated targets (predictions) as returned by a classifier.
        task : Literal["binary", "multiclass", "multilabel"]
            The task type. One of:
                - ``binary``: binary classification.
                    Example: [0, 1, 1, 0, 1] or [0.1, 0.9, 0.8, 0.2, 0.4]
                - ``multiclass``: multiclass classification.
                    Example: [0, 1, 2, 0, 1] or [[0.1, 0.9, 0.0], [0.0, 0.8, 0.2], ...]
                - ``multilabel``: multilabel classification.
                    Example: [[0, 1], [1, 0], [1, 1], [0, 0], [1, 0]] or
                    [[0.1, 0.9], [0.0, 0.8], ...]
        pos_label : int
            The class to report if task is binary. Defaults to 1.
        num_classes : int
            Number of classes. Necessary for ``multiclass`` tasks.
        threshold : float
            The threshold value for converting probability or logit scores to
            binary. A sigmoid function is first applied to logits to convert them
            to probabilities. Defaults to 0.5.
        top_k : Optional[int]
            The number of highest probability or logit score predictions considered
            to find the correct label. Only works when ``preds`` contain
            probabilities/logits.
        num_labels : int
            Number of labels. Necessary for ``multilabel`` tasks.
        average : Literal["micro", "macro", "weighted", "samples", None]
            If None, return the specificity for each class, otherwise return the
            average specificity. Average options are:
                - ``micro``: Calculate metrics globally by counting the total
                  true positives, false negatives and false positives.
                - ``macro``: Calculate metrics for each label, and find their
                  unweighted mean. This does not take label imbalance into account.
                - ``weighted``: Calculate metrics for each label, and find
                  their average, weighted by support (the number of true instances
                  for each label).
                - ``samples``: Calculate metrics for each instance, and find their
                  average.
        sample_weight : Optional[ArrayLike]
            Sample weights.
        zero_division : Literal["warn", 0, 1]
            Sets the value to return when there is a zero division. If set to
            ``warn``, this acts as 0, but warnings are also raised.

    Returns
    -------
        specificity : float or np.ndarray (if average is None).

    Raises
    ------
        ValueError
            If ``task`` is not one of ``binary``, ``multiclass``, or
            ``multilabel``.

    """
    if task == "binary":
        score = binary_specificity(
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
        score = multiclass_specificity(
            target,
            preds,
            num_classes,
            top_k=top_k,
            sample_weight=sample_weight,
            average=average,  # type: ignore
            zero_division=zero_division,
        )
    elif task == "multilabel":
        assert (
            isinstance(num_labels, int) and num_labels > 0
        ), "Number of labels must be specified for multilabel classification."
        score = multilabel_specificity(
            target,
            preds,
            num_labels,
            threshold=threshold,
            top_k=top_k,
            average=average,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )
    else:
        raise ValueError(
            f"Task {task} is not supported, expected one of 'binary', 'multiclass'"
            " or 'multilabel'"
        )

    return score


def binary_specificity(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    pos_label: int = 1,
    threshold: float = 0.5,
    sample_weight: Optional[ArrayLike] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> float:
    """Compute specificity for binary classification.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Estimated targets (predictions) as returned by a classifier.
        pos_label : int
            The class to report. Can be 0 or 1. Defaults to 1.
        threshold : float
            The threshold value for converting probability or logit scores to
            binary. A sigmoid function is first applied to logits to convert them
            to probabilities. Defaults to 0.5.
        sample_weight : Optional[ArrayLike]
            Sample weights.
        zero_division : Literal["warn", 0, 1]
            Sets the value to return when there is a zero division. If set to ``warn``,
            this acts as 0, but warnings are also raised.

    Returns
    -------
        specificity : float

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
        fn = np.array([fn])

    specificity_score = _specificity_reduce(
        tp,
        fp,
        tn,
        fn,
        average=None,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )

    return np.squeeze(specificity_score)


def multiclass_specificity(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    num_classes: int,
    top_k: Optional[int] = None,
    sample_weight: Optional[ArrayLike] = None,
    average: Literal["micro", "macro", "weighted", None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> Union[float, np.ndarray]:
    """Compute specificity for multiclass classification.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Estimated targets (predictions) as returned by a classifier.
        num_classes : int
            Number of classes.
        top_k : Optional[int]
            Number of highest probability or logit score predictions considered
            to find the correct label. Only works when ``preds`` contain
            probabilities/logits.
        sample_weight : Optional[ArrayLike]
            Sample weights.
        average : Literal["micro", "macro", "weighted", None]
            If None, return the specificity for each class, otherwise return the
            average specificity. Average options are:
                - ``micro``: Calculate metrics globally by counting the total true
                  positives, false negatives and false positives.
                - ``macro``: Calculate metrics for each label, and find their unweighted
                  mean. This does not take label imbalance into account.
                - ``weighted``: Calculate metrics for each label, and find their
                  average, weighted by support (the number of true instances for each
                  label).
        zero_division : Literal["warn", 0, 1]
            Sets the value to return when there is a zero division. If set to ``warn``,
            this acts as 0, but warnings are also raised.

    Returns
    -------
        specificity : float or np.ndarray (if average is None).

    """
    # pylint: disable=invalid-name
    tp, fp, tn, fn = _multiclass_stat_scores_update(
        target,
        preds,
        num_classes,
        sample_weight=sample_weight,
        classwise=True,
        top_k=top_k,
    )

    return _specificity_reduce(
        tp,
        fp,
        tn,
        fn,
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )


def multilabel_specificity(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    num_labels: int,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    sample_weight: Optional[ArrayLike] = None,
    average: Literal["micro", "macro", "samples", "weighted", None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> Union[float, np.ndarray]:
    """Compute specificity for multilabel classification.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Estimated targets (predictions) as returned by a classifier.
        num_labels : int
            Number of labels.
        threshold : float
            The threshold value for converting probability or logit scores to
            binary. A sigmoid function is first applied to logits to convert them
            to probabilities. Defaults to 0.5.
        top_k : Optional[int]
            Number of highest probability or logit score predictions considered
            to find the correct label. Only works when ``preds`` contain
            probabilities/logits.
        sample_weight : Optional[ArrayLike]
            Sample weights.
        average : Literal["micro", "macro", "samples", "weighted", None]
            If None, return the specificity for each class, otherwise return the
            average specificity. Average options are:
            - ``micro``: Calculate metrics globally by counting the total true
              positives, false negatives and false positives.
            - ``macro``: Calculate metrics for each label, and find their unweighted
              mean. This does not take label imbalance into account.
            - ``weighted``: Calculate metrics for each label, and find their average,
              weighted by support (the number of true instances for each label).
            - ``samples``: Calculate metrics for each instance, and find their average.
        zero_division : Literal["warn", 0, 1]
            Sets the value to return when there is a zero division. If set to ``warn``,
            this acts as 0, but warnings are also raised.

    Returns
    -------
        specificity : float or np.ndarray (if average is None).

    """
    # pylint: disable=invalid-name
    tp, fp, tn, fn = _multilabel_stat_scores_update(
        target,
        preds,
        num_labels,
        top_k=top_k,
        threshold=threshold,
        sample_weight=sample_weight,
        reduce="samples" if average == "samples" else "macro",
    )

    return _specificity_reduce(
        tp,
        fp,
        tn,
        fn,
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )
