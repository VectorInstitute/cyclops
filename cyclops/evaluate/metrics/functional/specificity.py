"""Functions to compute the specificity metric."""

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


def _specificity_reduce(  # pylint: disable=too-many-arguments, invalid-name
    tp: Union[np.int_, np.ndarray],
    fp: Union[np.int_, np.ndarray],
    tn: Union[np.int_, np.ndarray],
    fn: Union[np.int_, np.ndarray],
    average: Literal["micro", "macro", "weighted", None],
    zero_division: Literal["warn", 0, 1] = "warn",
) -> Union[float, np.ndarray]:
    """Reduce specificity.

    Parameters
    ----------
        tp : numpy.ndarray or int
            True positives.
        fp : numpy.ndarray or int
            False positives.
        tn : numpy.ndarray or int
            True negatives.
        fn : numpy.ndarray or int
            False negatives.
        average : Literal["micro", "macro", "weighted", None], default=None
            If None, return the specificity for each class, otherwise return the
            average specificity. Average options are:
                - ``micro``: Calculate metrics globally by counting the total
                  true positives, false negatives, false positives and true negatives.
                - ``macro``: Calculate metrics for each label, and find their
                  unweighted mean. This does not take label imbalance into account.
                - ``weighted``: Calculate metrics for each label, and find their
                  average, weighted by support (the number of true instances for
                  each label).
        zero_division : Literal["warn", 0, 1], default="warn"
            Sets the value to return when there is a zero division. If set to ``warn``,
            this acts as 0, but warnings are also raised.

    Returns
    -------
        specificity : float or numpy.ndarray (if average is None).

    """
    numerator = tn
    denominator = tn + fp

    if average == "micro":
        numerator = np.array(np.sum(numerator))
        denominator = np.array(np.sum(denominator))

    score = _prf_divide(
        np.array(numerator) if np.isscalar(tp) else numerator,
        np.array(denominator) if np.isscalar(tp) else denominator,
        metric="specificity",
        modifier="score",
        average=average,
        warn_for=("specificity",),
        zero_division=zero_division,
    )

    if average == "weighted":
        weights = tp + fn
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
        result = _get_value_if_singleton_array(score)

    return result


def binary_specificity(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    pos_label: int = 1,
    threshold: float = 0.5,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> float:
    """Compute specificity for binary classification tasks.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Estimated targets (predictions) as returned by a classifier.
        pos_label : int, default=1
            The label to use for the positive class.
        threshold : float, default=0.5
            The threshold to use for converting the predictions to binary
            values. Logits will be converted to probabilities using the sigmoid
            function.

    Returns
    -------
        float
            The specificity score.

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import binary_specificity
        >>> target = [0, 1, 1, 0, 1]
        >>> preds = [0.1, 0.9, 0.8, 0.5, 0.4]
        >>> binary_specificity(target, preds)
        0.5

    """
    _binary_stat_scores_args_check(threshold=threshold, pos_label=pos_label)

    target, preds = _binary_stat_scores_format(target, preds, threshold)

    # pylint: disable=invalid-name # for tn, fp, fn, tp
    tp, fp, tn, fn = _binary_stat_scores_update(target, preds, pos_label=pos_label)

    return _specificity_reduce(
        tp,
        fp,
        tn,
        fn,
        average=None,
        zero_division=zero_division,
    )


def multiclass_specificity(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    num_classes: int,
    top_k: Optional[int] = None,
    average: Literal["micro", "macro", "weighted", None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> Union[float, np.ndarray]:
    """Compute specificity for multiclass classification tasks.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Estimated targets (predictions) as returned by a classifier.
        num_classes : int
            The number of classes in the dataset.
        top_k : int, optional
            Number of highest probability or logit score predictions considered
            to find the correct label. Only works when ``preds`` contain
            probabilities/logits.
        average : Literal["micro", "macro", "weighted", None], default=None
            If None, return the specificity for each class, otherwise return the
            average specificity. Average options are:
                - ``micro``: Calculate metrics globally by counting the total true
                  positives, false negatives, false positives and true negatives.
                - ``macro``: Calculate metrics for each class, and find their unweighted
                  mean. This does not take class imbalance into account.
                - ``weighted``: Calculate metrics for each class, and find their
                  average, weighted by support (the number of true instances for each
                  label).
        zero_division : Literal["warn", 0, 1], default="warn"
            Sets the value to return when there is a zero division. If set to ``warn``,
            this acts as 0, but warnings are also raised.

    Returns
    -------
        float or numpy.ndarray
            The specificity score. If ``average`` is None, a numpy.ndarray of
            shape (``num_classes``,) is returned.

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import multiclass_specificity
        >>> target = [0, 1, 2, 0, 1, 2]
        >>> preds = [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.2, 0.75],
        ...          [0.35, 0.5, 0.15], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]
        >>> multiclass_specificity(target, preds, num_classes=3)
        array([1.  , 0.75, 1.  ])

    """
    _check_average_arg(average)

    target, preds = _multiclass_stat_scores_format(
        target, preds, num_classes=num_classes, top_k=top_k
    )

    # pylint: disable=invalid-name
    tp, fp, tn, fn = _multiclass_stat_scores_update(
        target,
        preds,
        num_classes,
        classwise=True,
    )

    return _specificity_reduce(
        tp,
        fp,
        tn,
        fn,
        average=average,
        zero_division=zero_division,
    )


def multilabel_specificity(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    num_labels: int,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    average: Literal["micro", "macro", "weighted", None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> Union[float, np.ndarray]:
    """Compute specificity for multilabel classification tasks.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Estimated targets (predictions) as returned by a classifier.
        num_labels : int
            The number of labels in the dataset.
        threshold : float, default=0.5
            The threshold value for converting probability or logit scores to
            binary. A sigmoid function is first applied to logits to convert them
            to probabilities.
        top_k : int, optional
            Number of highest probability or logit score predictions considered
            to find the correct label. Only works when ``preds`` contains
            probabilities/logits.
        average : Literal["micro", "macro", "weighted", None], default=None
            If None, return the specificity for each class, otherwise return the
            average specificity. Average options are:
                - ``micro``: Calculate metrics globally by counting the total
                  true positives, false negatives, false positives and true
                  negatives.
                - ``macro``: Calculate metrics for each label, and find their
                  unweighted mean. This does not take label imbalance into account.
                - ``weighted``: Calculate metrics for each label, and find their
                  average, weighted by support (the number of true instances for
                  each label).
        zero_division : Literal["warn", 0, 1], default="warn"
            Sets the value to return when there is a zero division. If set to ``warn``,
            this acts as 0, but warnings are also raised.

    Returns
    -------
        float or numpy.ndarray
            The specificity score. If ``average`` is None, a numpy.ndarray of
            shape (``num_labels``,) is returned.

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import multilabel_specificity
        >>> target = [[0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0]]
        >>> preds = [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.2, 0.75],
        ...          [0.35, 0.5, 0.15], [0.05, 0.9, 0.05]]
        >>> multilabel_specificity(target, preds, num_labels=3)
        array([0.5, 0., 0.5])

    """
    _check_average_arg(average)

    target, preds = _multilabel_stat_scores_format(
        target, preds, num_labels=num_labels, threshold=threshold, top_k=top_k
    )

    # pylint: disable=invalid-name
    tp, fp, tn, fn = _multilabel_stat_scores_update(
        target,
        preds,
        num_labels,
        labelwise=True,
    )

    return _specificity_reduce(
        tp,
        fp,
        tn,
        fn,
        average=average,
        zero_division=zero_division,
    )


def specificity(  # pylint: disable=too-many-arguments
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
    """Compute specificity score for different classification tasks.

    The specificity is the ratio of true negatives to the sum of true negatives and
    false positives. It is also the recall of the negative class.

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
                  positives, false positives, false negatives and true negatives.
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
        score : float or numpy.ndarray
            The specificity score. If ``average`` is ``None`` and ``task`` is not
            ``binary``, a numpy.ndarray of shape (``num_classes`` or ``num_labels``,)
            is returned.

    Raises
    ------
        ValueError
            If ``task`` is not one of ``binary``, ``multiclass``, or ``multilabel``.

    Examples (binary)
    -----------------
        >>> from cyclops.evaluation.metrics.functional import specificity
        >>> target = [0, 1, 1, 0, 1]
        >>> preds = [0.9, 0.05, 0.05, 0.35, 0.05]
        >>> specificity(target, preds, task="binary")
        0.5

    Examples (multiclass)
    ---------------------
        >>> from cyclops.evaluation.metrics.functional import specificity
        >>> target = [0, 1, 2, 0, 1]
        >>> preds = [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.2, 0.75],
        ...          [0.35, 0.5, 0.15], [0.05, 0.9, 0.05]]
        >>> specificity(target, preds, task="multiclass", num_classes=3)
        array([0.5, 0., 0.5])

    Examples (multilabel)
    ---------------------
        >>> from cyclops.evaluation.metrics.functional import specificity
        >>> target = [[0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0]]
        >>> preds = [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.2, 0.75],
        ...          [0.35, 0.5, 0.15], [0.05, 0.9, 0.05]]
        >>> specificity(target, preds, task="multilabel", num_labels=3)
        array([0.5, 0., 0.5])

    """
    if task == "binary":
        score = binary_specificity(
            target,
            preds,
            pos_label=pos_label,
            threshold=threshold,
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
            zero_division=zero_division,
        )
    else:
        raise ValueError(
            f"Task {task} is not supported, expected one of 'binary', 'multiclass'"
            " or 'multilabel'"
        )

    return score
