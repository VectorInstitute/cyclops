"""Functions for computing F-beta and F1 scores for different input types."""
from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics._classification import _prf_divide

from cyclops.evaluation.metrics.functional.stat_scores import (
    _binary_stat_scores_update,
    _multiclass_stat_scores_update,
    _multilabel_stat_scores_update,
)


def _fbeta_reduce(  # pylint: disable=too-many-arguments, invalid-name
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    beta: float,
    average: Literal["micro", "macro", "weighted", "samples", None],
    sample_weight: Optional[ArrayLike] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> Union[float, np.ndarray]:
    """Compute the F-beta score, a generalization of F-measure.

    Parameters
    ----------
        tp: np.ndarray
            True positives per class or sample.
        fp: np.ndarray
            False positives per class or sample.
        fn: np.ndarray
            False negatives per class or sample.
        beta: float
            Weight of precision in harmonic mean (beta < 1 lends more weight to
            precision, beta > 1 favors recall).
        average: Literal["micro", "macro", "weighted", "samples", None]
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:
                - ``micro``: Calculate metrics globally by counting the total
                  true positives, false negatives and false positives.
                - ``macro``: Calculate metrics for each label, and find their
                  unweighted mean.  This does not take label imbalance into account.
                - ``weighted``: Calculate metrics for each label, and find their
                  average, weighted by support (the number of true instances for each
                  label). This alters 'macro' to account for label imbalance;
                  it can result in an F-score that is not between precision and
                  recall.
                - ``samples``: Calculate metrics for each instance, and find
                  their average (only meaningful for multilabel classification).
        sample_weight: Optional[ArrayLike]
            Sample weights.
        zero_division: Literal["warn", 0, 1]
            Sets the value to return when there is a zero division:
                - ``warn``: return 0 if the denominator is zero, otherwise return
                  the result
                - ``0``: return 0 if the denominator is zero, otherwise return
                  the result
                - ``1``: return 1 if the denominator is zero, otherwise return
                  the result

    Returns
    -------
       F-beta score: float or np.ndarray (if `average` is `None`).

    Raises
    ------
        ValueError
            if beta is less than 0.

    """
    if beta < 0:
        raise ValueError("beta should be >=0 in the F-beta score")

    beta2 = beta**2

    numerator = (1 + beta2) * tp
    denominator = (1 + beta2) * tp + beta2 * fn + fp

    if average == "micro":
        numerator = np.array(np.sum(numerator))
        denominator = np.array(np.sum(denominator))

    score = _prf_divide(
        numerator,
        denominator,
        metric="f-score",
        modifier="true nor predicted",
        average=average,
        warn_for="f-score",
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


def fbeta_score(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    beta: float,
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
    """Compute the F-beta score, a generalization of F-measure.

    The F-beta score is the weighted harmonic mean of precision and recall,
    reaching its optimal value at 1 and its worst value at 0. The relative
    contribution of precision and recall to the F-beta score are determined
    by the `beta` parameter, beta = 1.0 means recall and precision are equally
    important. ``beta < 1`` lends more weight to precision, while ``beta > 1``
    favors recall (beta -> inf: precision, beta -> 0: recall).

    Parameters
    ----------
        target: ArrayLike
            Ground truth (correct) target values.
        preds: ArrayLike
            Estimated targets as returned by a classifier.
        beta: float
            Weight of precision in harmonic mean.
        task: Literal["binary", "multiclass", "multilabel"]
            The task type. One of:
                - ``binary``: binary classification.
                    Example: [0, 1, 1, 0, 1] or [0.1, 0.9, 0.8, 0.2, 0.4]
                - ``multiclass``: multiclass classification.
                    Example: [0, 1, 2, 0, 1] or [[0.1, 0.9, 0.0], [0.0, 0.8, 0.2], ...]
                - ``multilabel``: multilabel classification.
                    Example: [[0, 1], [1, 0], [1, 1], [0, 0], [1, 0]] or
                    [[0.1, 0.9], [0.0, 0.8], ...]
        pos_label: int
            The class to report if task is binary. Defaults to 1.
        num_classes: Optional[int]
            Number of classes. Necessary for multiclass task.
        threshold: float
            Threshold value for binary or multilabel probabilities. Defaults to
            0.5
        top_k: Optional[int]
            Number of highest probability entries for multiclass. Defaults to None.
        num_labels: Optional[int]
            Number of labels for multilabel task. Defaults to None.
        average: Literal["micro", "macro", "weighted", "samples", None]
            If not None, this determines the type of averaging performed on the data:
                - ``micro``: Calculate metrics globally by counting the total true
                  positives, false negatives and false positives.
                - ``macro``: Calculate metrics for each label, and find their unweighted
                  mean.  This does not take label imbalance into account.
                - ``weighted``: Calculate metrics for each label, and find their
                  average, weighted by support (the number of true instances for
                  each label). This alters ``macro`` to account for label imbalance;
                  it can result in an F-score that is not between precision and recall.
                - ``samples``: Calculate metrics for each instance, and find their
                  average (only meaningful for multilabel classification).
                - ``None``: The scores for each class are returned.
        sample_weight: Optional[ArrayLike]
            Sample weights.
        zero_division: Literal["warn", 0, 1]
            Value to return when there are no true positives or true negatives.
            If set to ``warn``, this acts as 0, but warnings are also raised.

    Returns
    -------
        F-beta score: float (if average is not None or task is binary) or
        np.ndarray (if average is None).

    Raises
    ------
        ValueError
            If ``task`` is not one of ``binary``, ``multiclass``, or
            ``multilabel``.

    """
    if task == "binary":
        score = binary_fbeta_score(
            target,
            preds,
            beta,
            pos_label=pos_label,
            threshold=threshold,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )
    elif task == "multiclass":
        assert (
            isinstance(num_classes, int) and num_classes > 0
        ), "Number of classes must be specified for multiclass classification."
        score = multiclass_fbeta_score(
            target,
            preds,
            beta,
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
        score = multilabel_fbeta_score(
            target,
            preds,
            beta,
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


def binary_fbeta_score(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    beta: float,
    pos_label: int = 1,
    threshold: float = 0.5,
    sample_weight: Optional[ArrayLike] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> float:
    """Compute the F-beta score for binary data.

    Parameters
    ----------
        target: ArrayLike
            Ground truth (correct) target values.
        preds: ArrayLike
            Predictions as returned by a classifier.
        beta: float
            Weight of precision in harmonic mean.
        pos_label: int
            The positive class label. Default is 1. One of [0, 1].
        threshold: float
            Threshold value for converting probabilities and logits to binary.
            Default: 0.5
        sample_weight: Optional[ArrayLike]
            Sample weights.
        zero_division: Literal["warn", 0, 1]
            Value to return when there are no true positives or true negatives.
            If set to ``warn``, this acts as 0, but warnings are also raised.

    Returns
    -------
        F-beta score: float

    Raises
    ------
        ValueError
            beta is less than 0.

    """
    if beta < 0:
        raise ValueError("beta should be >=0 in the F-beta score")

    # pylint: disable=invalid-name
    tp, fp, _, fn = _binary_stat_scores_update(
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

    score = _fbeta_reduce(
        tp,
        fp,
        fn,
        beta,
        average=None,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )

    return np.squeeze(score)


def multiclass_fbeta_score(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    beta: float,
    num_classes: int,
    top_k: Optional[int] = None,
    sample_weight: Optional[ArrayLike] = None,
    average: Literal["micro", "macro", "weighted", None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> Union[float, np.ndarray]:
    """Compute the F-beta score for multiclass data.

    Parameters
    ----------
        target: ArrayLike
            Ground truth (correct) target values.
        preds: ArrayLike
            Predictions as returned by a classifier.
        beta: float
            Weight of precision in harmonic mean.
        num_classes: int
            Number of classes.
        top_k: Optional[int]
            Number of highest probability entries for each sample to convert
            to 1s. If not set, top_k = 1.
        sample_weight: Optional[ArrayLike]
            Sample weights.
        average: Literal["micro", "macro", "weighted", None]
            If not None, this determines the type of averaging performed on the data:
                - ``micro``: Calculate metrics globally by counting the total true
                  positives, false negatives and false positives.
                - ``macro``: Calculate metrics for each label, and find their unweighted
                  mean.  This does not take label imbalance into account.
                - ``weighted``: Calculate metrics for each label, and find their
                  average, weighted by support (the number of true instances for each
                  label). This alters ``macro`` to account for label imbalance;
                  it can result in an F-score that is not between precision and recall.
                - ``None``: The scores for each class are returned.
        zero_division: Literal["warn", 0, 1]
            Value to return when there are no true positives or true negatives.
            If set to ``warn``, this acts as 0, but warnings are also raised.

    Returns
    -------
        F-beta score: float or np.ndarray (if average is ``None``).

    Raises
    ------
        ValueError
            ``average`` is not one of ``micro``, ``macro``, ``weighted``, or ``None``.

    """
    if average not in ["micro", "macro", "weighted", None]:
        raise ValueError(
            f"Argument average has to be one of 'micro', 'macro', 'weighted', "
            f"or None, got {average}."
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

    return _fbeta_reduce(
        tp,
        fp,
        fn,
        beta,
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )


def multilabel_fbeta_score(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    beta: float,
    num_labels: int,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    sample_weight: Optional[ArrayLike] = None,
    average: Literal["micro", "macro", "samples", "weighted", None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
):
    """Compute the F-beta score for multilabel data.

    Parameters
    ----------
        target: ArrayLike
            Ground truth (correct) target values.
        preds: ArrayLike
            Predictions as returned by a classifier.
        beta: float
            Weight of precision in harmonic mean.
        num_labels: int
            Number of labels.
        threshold: float
            Threshold value for converting probabilities and logits to binary.
            Default: 0.5.
        top_k: Optional[int]
            Number of highest probability entries for each sample to convert
            to 1s. If not set, top_k = 1.
        sample_weight: Optional[ArrayLike]
            Sample weights.
        average: Literal["micro", "macro", "samples", "weighted", None]
            If not None, this determines the type of averaging performed on the data:
                - ``micro``: Calculate metrics globally by counting the total true
                  positives, false negatives and false positives.
                - ``macro``: Calculate metrics for each label, and find their unweighted
                  mean.  This does not take label imbalance into account.
                - ``samples``: Calculate metrics for each instance, and find their
                  average.
                - ``weighted``: Calculate metrics for each label, and find their
                  average, weighted by support (the number of true instances for each
                  label). This alters ``macro`` to account for label imbalance;
                  it can result in an F-score that is not between precision and recall.
                - ``None``: The scores for each class are returned. Default: None.
        zero_division: Literal["warn", 0, 1]
            Value to return when there are no true positives or true negatives.
            If set to ``warn``, this acts as 0, but warnings are also raised.

    Returns
    -------
        F-beta score: float or np.ndarray (if average is ``None``).

    Raises
    ------
        ValueError
            ``average`` is not one of ``micro``, ``macro``, ``samples``,
            ``weighted``, or ``None``.

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

    return _fbeta_reduce(
        tp,
        fp,
        fn,
        beta,
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )


def f1_score(  # pylint: disable=too-many-arguments
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
    """Compute the F1 score for multiclass data.

    Parameters
    ----------
        target: ArrayLike
            Ground truth (correct) target values.
        preds: ArrayLike
            Estimated targets as returned by a classifier.
        task: Literal["binary", "multiclass", "multilabel"]
            The task type. One of:
                - ``binary``: binary classification.
                    Example: [0, 1, 1, 0, 1] or [0.1, 0.9, 0.8, 0.2, 0.4]
                - ``multiclass``: multiclass classification.
                    Example: [0, 1, 2, 0, 1] or [[0.1, 0.9, 0.0], [0.0, 0.8, 0.2], ...]
                - ``multilabel``: multilabel classification.
                    Example: [[0, 1], [1, 0], [1, 1], [0, 0], [1, 0]] or
                    [[0.1, 0.9], [0.0, 0.8], ...]
        pos_label: int
            The class to report if task is binary. Defaults to 1.
        num_classes: Optional[int]
            Number of classes. Necessary for multiclass task.
        threshold: float
            Threshold value for binary or multilabel probabilities. Defaults to
            0.5
        top_k: Optional[int]
            Number of highest probability entries for multiclass. Defaults to None.
        num_labels: Optional[int]
            Number of labels for multilabel task. Defaults to None.
        average: Literal["micro", "macro", "weighted", "samples", None]
            If not None, this determines the type of averaging performed on the data:
                - ``micro``: Calculate metrics globally by counting the total true
                  positives, false negatives and false positives.
                - ``macro``: Calculate metrics for each label, and find their unweighted
                  mean.  This does not take label imbalance into account.
                - ``weighted``: Calculate metrics for each label, and find their
                  average, weighted by support (the number of true instances for each
                  label). This alters ``macro`` to account for label imbalance; it can
                  result in an F-score that is not between precision and recall.
                - ``samples``: Calculate metrics for each instance, and find their
                  average (only meaningful for multilabel classification).
                - ``None``: The scores for each class are returned.
        sample_weight: Optional[ArrayLike]
            Sample weights.
        zero_division: Literal["warn", 0, 1]
            Value to return when there are no true positives or true negatives.
            If set to ``warn``, this acts as 0, but warnings are also raised.

    Returns
    -------
        F1 score: float (if average is not None or task is binary) or
        np.ndarray (if average is None).

    """
    return fbeta_score(
        target,
        preds,
        1.0,
        task,
        pos_label=pos_label,
        num_classes=num_classes,
        threshold=threshold,
        top_k=top_k,
        num_labels=num_labels,
        sample_weight=sample_weight,
        average=average,
        zero_division=zero_division,
    )


def binary_f1_score(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    pos_label: int = 1,
    threshold: float = 0.5,
    sample_weight: Optional[ArrayLike] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> float:
    """Compute the F1 score for binary data.

    Parameters
    ----------
        target: ArrayLike
            Ground truth (correct) target values.
        preds: ArrayLike
            Predictions as returned by a classifier.
        pos_label: int
            The class to report. Defaults to 1. One of 0 or 1
        threshold: float
            Threshold value for converting probabilities and logits to binary.
            Default: 0.5
        sample_weight: Optional[ArrayLike]
            Sample weights.
        zero_division: Literal["warn", 0, 1]
            Value to return when there are no true positives or true negatives.
            If set to ``warn``, this acts as 0, but warnings are also raised.

    Returns
    -------
        F1 score: float

    """
    return binary_fbeta_score(
        target,
        preds,
        beta=1.0,
        pos_label=pos_label,
        threshold=threshold,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )


def multiclass_f1_score(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    num_classes: int,
    top_k: Optional[int] = None,
    sample_weight: Optional[ArrayLike] = None,
    average: Literal["micro", "macro", "weighted", None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> Union[float, np.ndarray]:
    """Compute the F1 score for multiclass data.

    Parameters
    ----------
        target: ArrayLike
            Ground truth (correct) target values.
        preds: ArrayLike
            Predictions as returned by a classifier.
        num_classes: int
            Number of classes.
        top_k: Optional[int]
            Number of highest probability entries for each sample to convert
            to 1s. If not set, top_k = 1.
        sample_weight: Optional[ArrayLike]
            Sample weights.
        average: Literal["micro", "macro", "weighted", None]
            If not None, this determines the type of averaging performed on the data:
                - ``micro``: Calculate metrics globally by counting the total true
                  positives, false negatives and false positives.
                - ``macro``: Calculate metrics for each label, and find their unweighted
                  mean.  This does not take label imbalance into account.
                - ``weighted``: Calculate metrics for each label, and find their
                  average, weighted by support (the number of true instances for each
                  label). This alters ``macro`` to account for label imbalance;
                  it can result in an F-score that is not between precision and recall.
                - ``None``: The scores for each class are returned. Default: None.
        zero_division: Literal["warn", 0, 1]
            Value to return when there are no true positives or true negatives.
            If set to ``warn``, this acts as 0, but warnings are also raised.

    Returns
    -------
        F1 score: float or np.ndarray (if average is None).

    """
    return multiclass_fbeta_score(
        target,
        preds,
        beta=1.0,
        num_classes=num_classes,
        top_k=top_k,
        sample_weight=sample_weight,
        average=average,
        zero_division=zero_division,
    )


def multilabel_f1_score(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    num_labels: int,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    sample_weight: Optional[ArrayLike] = None,
    average: Literal["micro", "macro", "samples", "weighted", None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
):
    """Compute the F1 score for multilabel data.

    Parameters
    ----------
        target: ArrayLike
            Ground truth (correct) target values.
        preds: ArrayLike
            Predictions as returned by a classifier.
        num_labels: int
            Number of labels.
        threshold: float
            Threshold value for converting probabilities and logits to binary.
            Default: 0.5.
        top_k: Optional[int]
            Number of highest probability entries for each sample to convert
            to 1s. If not set, top_k = 1.
        sample_weight: Optional[ArrayLike]
            Sample weights.
        average: Literal["micro", "macro", "samples", "weighted", None]
            If not None, this determines the type of averaging performed on the data:
                - ``micro``: Calculate metrics globally by counting the total true
                  positives, false negatives and false positives.
                - ``macro``: Calculate metrics for each label, and find their unweighted
                  mean.  This does not take label imbalance into account.
                - ``samples``: Calculate metrics for each instance, and find their
                  average.
                - ``weighted``: Calculate metrics for each label, and find their
                  average, weighted by support (the number of true instances for each
                  label). This alters ``macro`` to account for label imbalance;
                  it can result in an F-score that is not between precision and recall.
                - ``None``: The scores for each class are returned. Default: None.
        zero_division: Literal["warn", 0, 1]
            Value to return when there are no true positives or true negatives.
            If set to ``warn``, this acts as 0, but warnings are also raised.

    Returns
    -------
        F1 score: float or np.ndarray (if average is None).

    """
    return multilabel_fbeta_score(
        target,
        preds,
        beta=1.0,
        num_labels=num_labels,
        threshold=threshold,
        top_k=top_k,
        sample_weight=sample_weight,
        average=average,
        zero_division=zero_division,
    )
