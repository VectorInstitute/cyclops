"""Functions for computing F-beta and F1 scores for different input types."""
from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics._classification import _prf_divide

from cyclops.evaluate.metrics.functional.stat_scores import (
    _binary_stat_scores_args_check,
    _binary_stat_scores_format,
    _binary_stat_scores_update,
    _multiclass_stat_scores_format,
    _multiclass_stat_scores_update,
    _multilabel_stat_scores_format,
    _multilabel_stat_scores_update,
)
from cyclops.evaluate.metrics.utils import (
    _check_average_arg,
    _get_value_if_singleton_array,
)


def _fbeta_reduce(  # pylint: disable=too-many-arguments, invalid-name
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    beta: float,
    average: Literal["micro", "macro", "weighted", None],
    zero_division: Literal["warn", 0, 1] = "warn",
) -> Union[float, np.ndarray]:
    """Compute the F-beta score, a generalization of F-measure.

    Parameters
    ----------
        tp : numpy.ndarray
            True positives per class.
        fp : numpy.ndarray
            False positives per class.
        fn : numpy.ndarray
            False negatives per class.
        beta : float
            Weight of precision in harmonic mean (beta < 1 lends more weight to
            precision, beta > 1 favors recall).
       average : Literal["micro", "macro", "weighted", None], default=None
            If ``None``, return the score for each class. Otherwise,
            use one of the following options to compute the average score:
                - ``micro``: Calculate metric globally from the total count of true
                  positives, false positives and false negatives.
                - ``macro``: Calculate metric for each label, and find their
                  unweighted mean. This does not take label/class imbalance
                  into account.
                - ``weighted``: Calculate metric for each label/class, and find their
                  average weighted by the support (the number of true instances
                  for each label/class). This alters "macro" to account for
                  label/class imbalance.
        zero_division : Literal["warn", 0, 1], default="warn"
            Value to return when there is a zero division. If set to "warn", this
            acts as 0, but warnings are also raised.

    Returns
    -------
        result : float or numpy.ndarray
            F-beta score or array of scores if ``average=None``.

    Raises
    ------
        ValueError
            if beta is less than 0.

    """
    _check_beta(beta=beta)

    beta2 = beta**2

    numerator = (1 + beta2) * tp
    denominator = (1 + beta2) * tp + beta2 * fn + fp

    if average == "micro":
        numerator = np.array(np.sum(numerator))
        denominator = np.array(np.sum(denominator))

    score = _prf_divide(
        np.array(numerator) if np.isscalar(tp) else numerator,
        np.array(denominator) if np.isscalar(tp) else denominator,
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
    else:
        weights = None

    if average is not None and score.ndim != 0 and len(score) > 1:
        result = np.average(score, weights=weights)
    else:
        result = _get_value_if_singleton_array(score)

    return result


def _check_beta(beta: float) -> None:
    """Check the ``beta`` argument for F-beta metrics."""
    if beta < 0:
        raise ValueError("beta should be >=0 in the F-beta score")


def binary_fbeta_score(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    beta: float,
    pos_label: int = 1,
    threshold: float = 0.5,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> float:
    """Compute the F-beta score for binary classification tasks.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Predictions as returned by a classifier.
        beta : float
            Weight of precision in harmonic mean.
        pos_label : int, default=1
            The positive class label. One of [0, 1].
        threshold : float, default=0.5
            Threshold value for converting probabilities and logits to binary.
        zero_division : Literal["warn", 0, 1], default="warn"
            Value to return when there are no true positives or true negatives.
            If set to ``warn``, this acts as 0, but warnings are also raised.

    Returns
    -------
        float
            The binary F-beta score.

    Raises
    ------
        ValueError
            beta is less than 0.

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import binary_fbeta_score
        >>> target = [0, 1, 0, 1]
        >>> preds = [0, 1, 1, 1]
        >>> binary_fbeta_score(target, preds, beta=0.5)
        0.7142857142857143

    """
    _check_beta(beta=beta)
    _binary_stat_scores_args_check(threshold=threshold, pos_label=pos_label)

    target, preds = _binary_stat_scores_format(
        target=target, preds=preds, threshold=threshold
    )

    # pylint: disable=invalid-name
    tp, fp, _, fn = _binary_stat_scores_update(
        target=target, preds=preds, pos_label=pos_label
    )

    return _fbeta_reduce(
        tp=tp,
        fp=fp,
        fn=fn,
        beta=beta,
        average=None,
        zero_division=zero_division,
    )


def multiclass_fbeta_score(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    beta: float,
    num_classes: int,
    top_k: Optional[int] = None,
    average: Literal["micro", "macro", "weighted", None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> Union[float, np.ndarray]:
    """Compute the F-beta score for multiclass data.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Predictions as returned by a classifier.
        beta : float
            Weight of precision in harmonic mean.
        num_classes : int
            The number of classes in the dataset.
        top_k : int, optional
            If given, and predictions are probabilities/logits, the score will
            be computed only for the top k classes. Otherwise, ``top_k`` will be
            set to 1.
        average : Literal["micro", "macro", "weighted", None], default=None
           If ``None``, return the score for each class. Otherwise,
           use one of the following options to compute the average score:
                - ``micro``: Calculate metric globally from the total count of true
                  positives, false positives and false negatives.
                - ``macro``: Calculate metric for each class, and find their
                  unweighted mean. This does not take label imbalance into account.
                - ``weighted``: Calculate metric for each class, and find their
                  average weighted by the support (the number of true instances
                  for each class). This alters "macro" to account for class
                  imbalance.
        zero_division : Literal["warn", 0, 1], default="warn"
            Value to return when there is a zero division. If set to "warn", this
            acts as 0, but warnings are also raised.

    Returns
    -------
        float or numpy.ndarray
            The multiclass F-beta score. If ``average`` is ``None``, a numpy array
            of shape (num_classes,) is returned.

    Raises
    ------
        ValueError
            ``average`` is not one of ``micro``, ``macro``, ``weighted``, or ``None``,
            or ``beta`` is less than 0.

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import multiclass_fbeta_score
        >>> target = [0, 1, 2, 0]
        >>> preds = [0, 2, 1, 0]
        >>> multiclass_fbeta_score(target, preds, beta=0.5, num_classes=3)
        array([1., 0., 0.])

    """
    _check_beta(beta=beta)
    _check_average_arg(average=average)

    target, preds = _multiclass_stat_scores_format(
        target, preds, num_classes=num_classes, top_k=top_k
    )

    # pylint: disable=invalid-name
    tp, fp, _, fn = _multiclass_stat_scores_update(
        target=target, preds=preds, num_classes=num_classes, classwise=True
    )

    return _fbeta_reduce(
        tp=tp,
        fp=fp,
        fn=fn,
        beta=beta,
        average=average,
        zero_division=zero_division,
    )


def multilabel_fbeta_score(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    beta: float,
    num_labels: int,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    average: Literal["micro", "macro", "weighted", None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
):
    """Compute the F-beta score for multilabel data.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Predictions as returned by a classifier.
        beta : float
            Weight of precision in harmonic mean.
        num_labels : int
            Number of labels for the task.
        threshold : float, default=0.5
            Threshold for deciding the positive class.
        top_k : int, optional
            If given, and predictions are probabilities/logits, the score will
            be computed only for the top k classes. Otherwise, ``top_k`` will be
            set to 1.
        average : Literal["micro", "macro", "weighted", None], default=None
            If ``None``, return the score for each label. Otherwise,
            use one of the following options to compute the average score:
                - ``micro``: Calculate metric globally from the total count of true
                  positives, false positives and false negatives.
                - ``macro``: Calculate metric for each label, and find their
                  unweighted mean. This does not take label imbalance into account.
                - ``weighted``: Calculate metric for each label, and find their
                  average weighted by the support (the number of true instances
                  for each label). This alters "macro" to account for label
                  imbalance.
        zero_division : Literal["warn", 0, 1], default="warn"
            Value to return when there is a zero division. If set to "warn", this
            acts as 0, but warnings are also raised.

    Returns
    -------
        float or numpy.ndarray
            The multilabel F-beta score. If ``average`` is ``None``, a numpy array
            of shape (num_labels,) is returned.

    Raises
    ------
        ValueError
            ``average`` is not one of ``micro``, ``macro``, ``weighted``, or ``None``,
            or ``beta`` is less than 0.

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import multilabel_fbeta_score
        >>> target = [[0, 1], [1, 1]]
        >>> preds = [[0.1, 0.9], [0.8, 0.2]]
        >>> multilabel_fbeta_score(target, preds, beta=0.5, num_labels=2)
        array([1.        , 0.83333333])

    """
    _check_beta(beta=beta)
    _check_average_arg(average=average)

    target, preds = _multilabel_stat_scores_format(
        target, preds, num_labels=num_labels, threshold=threshold, top_k=top_k
    )

    # pylint: disable=invalid-name
    tp, fp, _, fn = _multilabel_stat_scores_update(
        target=target, preds=preds, num_labels=num_labels, labelwise=True
    )

    return _fbeta_reduce(
        tp=tp,
        fp=fp,
        fn=fn,
        beta=beta,
        average=average,
        zero_division=zero_division,
    )


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
    average: Literal["micro", "macro", "weighted", None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> Union[float, np.ndarray]:
    """Compute the F-beta score for binary, multiclass, or multilabel data.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Estimated targets as returned by a classifier.
        beta : float
            Weight of precision in harmonic mean.
        task : Literal["binary", "multiclass", "multilabel"]
            Type of classification task.
        pos_label : int, default=1
            Label to consider as positive for binary classification tasks.
        num_classes : int
            Number of classes for the task. Required if ``task`` is ``"multiclass"``.
        threshold : float, default=0.5
            Threshold for deciding the positive class. Only used if ``task`` is
            ``"binary"`` or ``"multilabel"``.
        top_k : int, optional
            If given, and predictions are probabilities/logits, the precision will
            be computed only for the top k classes. Otherwise, ``top_k`` will be
            set to 1. Only used if ``task`` is ``"multiclass"`` or ``"multilabel"``.
        num_labels : int
            Number of labels for the task. Required if ``task`` is ``"multilabel"``.
        average : Literal["micro", "macro", "weighted", None], default=None
            If ``None``, return the score for each label/class. Otherwise,
            use one of the following options to compute the average score:
                - ``micro``: Calculate metrics globally by counting the total true
                  positives, false positives and false negatives.
                - ``macro``: Calculate metrics for each class/label, and find their
                  unweighted mean. This does not take label/class imbalance into
                  account.
                - ``weighted``: Calculate metrics for each label/class, and find
                  their average weighted by support (the number of true instances
                  for each label/class). This alters ``macro`` to account for
                  label/class imbalance.
        zero_division : Literal["warn", 0, 1], default="warn"
            Value to return when there is a zero division. If set to "warn", this
            acts as 0, but warnings are also raised.

    Returns
    -------
        score: float or numpy.ndarray
            The F-beta score. If ``average`` is not ``None`` and ``task`` is not
            ``binary``, a numpy array of shape (num_classes,) is returned.

    Raises
    ------
        ValueError
            If ``task`` is not one of ``binary``, ``multiclass``, or
            ``multilabel``.

    Examples (binary)
    -----------------
        >>> from cyclops.evaluation.metrics.functional import fbeta_score
        >>> target = [0, 1, 1, 0]
        >>> preds = [0.1, 0.8, 0.4, 0.3]
        >>> fbeta_score(target, preds, beta=0.5, task="binary")
        0.8333333333333334

    Examples (multiclass)
    ---------------------
        >>> from cyclops.evaluation.metrics.functional import fbeta_score
        >>> target = [0, 1, 2, 2]
        >>> preds = [1 2, 2, 0]
        >>> fbeta_score(target, preds, beta=0.5, task="multiclass", num_classes=3)
        array([0.83333333, 0.        , 0.55555556])

    Examples (multilabel)
    ---------------------
        >>> from cyclops.evaluation.metrics.functional import fbeta_score
        >>> target = [[0, 1], [1, 1]]
        >>> preds = [[0.1, 0.9], [0.8, 0.2]]
        >>> fbeta_score(target, preds, beta=0.5, task="multilabel", num_labels=2)
        array([1.        , 0.83333333])

    """
    if task == "binary":
        score = binary_fbeta_score(
            target,
            preds,
            beta,
            pos_label=pos_label,
            threshold=threshold,
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
            zero_division=zero_division,
        )
    else:
        raise ValueError(
            f"Task {task} is not supported, expected one of 'binary', 'multiclass'"
            " or 'multilabel'"
        )

    return score


def binary_f1_score(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    pos_label: int = 1,
    threshold: float = 0.5,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> float:
    """Compute the F1 score for binary classification tasks.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Predictions as returned by a classifier.
        pos_label: int, default=1
            The label of the positive class.
        threshold : float, default=0.5
            Threshold value for binarizing predictions in form of logits or
            probability scores.
        zero_division : Literal["warn", 0, 1], default="warn"
            Value to return when there is a zero division. If set to "warn", this
            acts as 0, but warnings are also raised.

    Returns
    -------
        float
            The F1 score.

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import binary_f1_score
        >>> target = [0, 1, 1, 0]
        >>> preds = [0.1, 0.8, 0.4, 0.3]
        >>> binary_f1_score(target, preds)
        0.6666666666666666

    """
    return binary_fbeta_score(
        target,
        preds,
        beta=1.0,
        pos_label=pos_label,
        threshold=threshold,
        zero_division=zero_division,
    )


def multiclass_f1_score(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    num_classes: int,
    top_k: Optional[int] = None,
    average: Literal["micro", "macro", "weighted", None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> Union[float, np.ndarray]:
    """Compute the F1 score for multiclass classification tasks.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Predictions as returned by a classifier.
        num_classes : int
            Number of classes in the dataset.
        top_k : int, optional
            If given, and predictions are probabilities/logits, the precision will
            be computed only for the top k classes. Otherwise, ``top_k`` will be
            set to 1.
        average : Literal["micro", "macro", "weighted", None], default=None
           If ``None``, return the score for each class. Otherwise, use one of
           the following options to compute the average score:
                - ``micro``: Calculate metric globally from the total count of true
                  positives, false positives and false negatives.
                - ``macro``: Calculate metric for each class, and find their
                  unweighted mean. This does not take class imbalance into account.
                - ``weighted``: Calculate metric for each class, and find their
                  average weighted by the support (the number of true instances
                  for each class). This alters "macro" to account for class
                  imbalance. It can result in an F-score that is not between
                  precision and recall.
        zero_division : Literal["warn", 0, 1], default="warn"
            Value to return when there is a zero division. If set to "warn", this
            acts as 0, but warnings are also raised.

    Returns
    -------
        float or numpy.ndarray
            The F1 score. If ``average`` is ``None``, a numpy.ndarray of shape
            (``num_classes``,) is returned.

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import multiclass_f1_score
        >>> target = [0, 1, 2, 0]
        >>> preds = [1, 1, 1, 0]
        >>> multiclass_f1_score(target, preds, num_classes=3)
        array([0.66666667, 0.5       , 0.        ])

    """
    return multiclass_fbeta_score(
        target,
        preds,
        beta=1.0,
        num_classes=num_classes,
        top_k=top_k,
        average=average,
        zero_division=zero_division,
    )


def multilabel_f1_score(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    num_labels: int,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    average: Literal["micro", "macro", "weighted", None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
):
    """Compute the F1 score for multilabel classification tasks.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Predictions as returned by a classifier.
        num_labels : int
            Number of labels for the task.
        threshold : float, default=0.5
            Threshold for deciding the positive class.
        top_k : int, optional
            If given, and predictions are probabilities/logits, the precision will
            be computed only for the top k classes. Otherwise, ``top_k`` will be
            set to 1.
        average : Literal["micro", "macro", "weighted", None], default=None
            If ``None``, return the score for each label. Otherwise, use one of
            the following options to compute the average score:
                - ``micro``: Calculate metric globally from the total count of true
                  positives, false positives and false negatives.
                - ``macro``: Calculate metric for each label, and find their
                  unweighted mean. This does not take label imbalance into account.
                - ``weighted``: Calculate metric for each label, and find their
                  average weighted by the support (the number of true instances
                  for each label). This alters "macro" to account for label
                  imbalance.
        zero_division : Literal["warn", 0, 1], default="warn"
            Value to return when there is a zero division. If set to "warn", this
            acts as 0, but warnings are also raised.

    Returns
    -------
        float or numpy.ndarray
            The F1 score. If ``average`` is ``None``, a numpy.ndarray of shape
            (``num_labels``,) is returned.

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import multilabel_f1_score
        >>> target = [[0, 1, 1], [1, 0, 0]]
        >>> preds = [[0.1, 0.9, 0.8], [0.05, 0.1, 0.2]]
        >>> multilabel_f1_score(target, preds, num_labels=3)
        array([0., 1., 1.])

    """
    return multilabel_fbeta_score(
        target,
        preds,
        beta=1.0,
        num_labels=num_labels,
        threshold=threshold,
        top_k=top_k,
        average=average,
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
    average: Literal["micro", "macro", "weighted", None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> Union[float, np.ndarray]:
    """Compute the F1 score for multiclass data.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Estimated targets as returned by a classifier.
        task : Literal["binary", "multiclass", "multilabel"]
            Type of classification task.
        pos_label : int, default=1
            Label to consider as positive for binary classification tasks.
        num_classes : int, default=None
            Number of classes for the task. Required if ``task`` is ``"multiclass"``.
        threshold : float, default=0.5
            Threshold for deciding the positive class. Only used if ``task`` is
            ``"binary"`` or ``"multilabel"``.
        top_k : int, optional
            If given, and predictions are probabilities/logits, the precision will
            be computed only for the top k classes. Otherwise, ``top_k`` will be
            set to 1. Only used if ``task`` is ``"multiclass"`` or ``"multilabel"``.
        num_labels : int, default=None
            Number of labels for the task. Required if ``task`` is ``"multilabel"``.
        average : Literal["micro", "macro", "weighted", None], default=None
            If ``None``, return the score for each label/class. Otherwise,
            use one of the following options to compute the average score:
                - ``micro``: Calculate metrics globally by counting the total true
                  positives, false positives and false negatives.
                - ``macro``: Calculate metrics for each class/label, and find their
                  unweighted mean. This does not take label/class imbalance into
                  account.
                - ``weighted``: Calculate metrics for each label/class, and find
                  their average weighted by support (the number of true instances
                  for each label/class). This alters ``macro`` to account for
                  label/class imbalance.
        zero_division : Literal["warn", 0, 1], default="warn"
            Value to return when there is a zero division. If set to "warn", this
            acts as 0, but warnings are also raised.


    Returns
    -------
        float or numpy.ndarray
            The F1 score. If ``average`` is ``None`` and ``task`` is not ``binary``,
            a numpy.ndarray of shape (``num_classes`` or ``num_labels``,) is returned.

    Examples (binary)
    -----------------
        >>> from cyclops.evaluation.metrics.functional import f1_score
        >>> target = [0, 1, 0, 1]
        >>> preds = [0.1, 0.9, 0.8, 0.2]
        >>> f1_score(target, preds, task="binary")
        0.5

    Examples (multiclass)
    ---------------------
        >>> from cyclops.evaluation.metrics.functional import f1_score
        >>> target = [0, 1, 2, 0]
        >>> preds = [[0.05, 0.95, 0], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6], [0.9, 0.1, 0]]
        >>> f1_score(target, preds, task="multiclass", num_classes=3)
        array([0.66666667, 0.8       , 0.        ])

    Examples (multilabel)
    ---------------------
        >>> from cyclops.evaluation.metrics.functional import f1_score
        >>> target = [[0, 1, 1], [1, 0, 0]]
        >>> preds = [[0.1, 0.9, 0.8], [0.05, 0.1, 0.2]]
        >>> f1_score(target, preds, task="multilabel", num_labels=3)
        array([0., 1., 1.])

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
        average=average,
        zero_division=zero_division,
    )
