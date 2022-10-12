"""Functions for computing F-beta and F1 scores for different input types."""
from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics._classification import _prf_divide

from .stat_scores import (
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

    The F-beta score is the weighted harmonic mean of precision and recall,
    reaching its optimal value at 1 and its worst value at 0. The relative
    contribution of precision and recall to the F-beta score are determined
    by the `beta` parameter, beta = 1.0 means recall and precision are equally
    important. `beta < 1` lends more weight to precision, while `beta > 1`
    favors recall (beta -> inf: precision, beta -> 0: recall).

    Parameters
    ----------
        tp: np.ndarray
            True positives per class
        fp: np.ndarray
            False positives per class
        fn: np.ndarray
            False negatives per class
        beta: float
            Weight of precision in harmonic mean.
        average: Literal["micro", "macro", "weighted", "samples", None]
            If not None, this determines the type of averaging performed on the data:
            ``"micro"``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``"macro"``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``"weighted"``:
                Calculate metrics for each label, and find their average, weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.
            ``"samples"``:
                Calculate metrics for each instance, and find their average (only
                meaningful for multilabel classification).
            ``None``:
                The scores for each class are returned.
        sample_weight: Optional[ArrayLike]
            Sample weights.
        zero_division: Literal["warn", 0, 1]
            Sets the value to return when there is a zero division:
            ``"warn"``:
                return 0 if the denominator is zero, otherwise return the result
            ``0``:
                return 0 if the denominator is zero, otherwise return the result
            ``1``:
                return 1 if the denominator is zero, otherwise return the result

    Returns
    -------
       F-beta score: float or np.ndarray (if average is None).

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
        "f-score",
        "true nor predicted",
        average,
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


def binary_fbeta_score(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    beta: float,
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
        threshold: float
            Threshold value for converting probabilities and logits to binary.
            Default: 0.5
        sample_weight: Optional[ArrayLike]
            Sample weights.
        zero_division: Literal["warn", 0, 1]
            Sets the value to return when there is a zero division:
            ``"warn"``:
                return 0 if the denominator is zero, otherwise return the result.
            ``0``:
                return 0 if the denominator is zero, otherwise return the result.
            ``1``:
                return 1 if the denominator is zero, otherwise return the result.

    Returns
    -------
        F-beta score: float

    """
    if beta < 0:
        raise ValueError("beta should be >=0 in the F-beta score")

    # pylint: disable=invalid-name
    tp, fp, _, fn = _binary_stat_scores_update(
        target, preds, sample_weight=sample_weight, threshold=threshold
    )

    if tp.ndim == 0:
        tp = np.array([tp])
        fp = np.array([fp])
        fn = np.array([fn])

    return _fbeta_reduce(
        tp,
        fp,
        fn,
        beta,
        None,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )


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
            ``"micro"``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``"macro"``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``"weighted"``:
                Calculate metrics for each label, and find their average, weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.
            ``None``:
                The scores for each class are returned.
        zero_division: Literal["warn", 0, 1]
            Sets the value to return when there is a zero division:
            ``"warn"``:
                return 0 if the denominator is zero, otherwise return the result.
            ``0``:
                return 0 if the denominator is zero, otherwise return the result.
            ``1``:
                return 1 if the denominator is zero, otherwise return the result.

    Returns
    -------
        F-beta score: float or np.ndarray (if average is None).

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
        average,
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
            ``"micro"``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``"macro"``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``"samples"``:
                Calculate metrics for each instance, and find their average.
            ``"weighted"``:
                Calculate metrics for each label, and find their average, weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.
            ``None``: The scores for each class are returned.
            Default: None.
        zero_division: Literal["warn", 0, 1]
            Sets the value to return when there is a zero division:
            ``"warn"``:
                return 0 if the denominator is zero, otherwise return the result.
            ``0``:
                return 0 if the denominator is zero, otherwise return the result.
            ``1``:
                return 1 if the denominator is zero, otherwise return the result.

    Returns
    -------
        F-beta score: float or np.ndarray (if average is None).

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
        average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )


def binary_f1_score(
    target: ArrayLike,
    preds: ArrayLike,
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
            Predictions as returned by a classifier..
        threshold: float
            Threshold value for converting probabilities and logits to binary.
            Default: 0.5
        sample_weight: Optional[ArrayLike]
            Sample weights.
        zero_division: Literal["warn", 0, 1]
            Sets the value to return when there is a zero division:
            ``"warn"``:
                return 0 if the denominator is zero, otherwise return the result.
            ``0``:
                return 0 if the denominator is zero, otherwise return the result.
            ``1``:
                return 1 if the denominator is zero, otherwise return the result.

    Returns
    -------
        F1 score: float

    """
    return binary_fbeta_score(
        target,
        preds,
        beta=1.0,
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
            ``"micro"``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``"macro"``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``"weighted"``:
                Calculate metrics for each label, and find their average, weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.
            ``None``: The scores for each class are returned.
            Default: None.
        zero_division: Literal["warn", 0, 1]
            Sets the value to return when there is a zero division:
            ``"warn"``:
                return 0 if the denominator is zero, otherwise return the result.
            ``0``:
                return 0 if the denominator is zero, otherwise return the result.
            ``1``:
                return 1 if the denominator is zero, otherwise return the result.

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
            ``"micro"``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``"macro"``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``"samples"``:
                Calculate metrics for each instance, and find their average.
            ``"weighted"``:
                Calculate metrics for each label, and find their average, weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.
            ``None``: The scores for each class are returned.
            Default: None.
        zero_division: Literal["warn", 0, 1]
            Sets the value to return when there is a zero division:
            ``"warn"``:
                return 0 if the denominator is zero, otherwise return the result.
            ``0``:
                return 0 if the denominator is zero, otherwise return the result.
            ``1``:
                return 1 if the denominator is zero, otherwise return the result.

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
