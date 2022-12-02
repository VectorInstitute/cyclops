"""Functions for computing accuracy scores for classification tasks."""

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


def _accuracy_reduce(  # pylint: disable=too-many-arguments
    tp: Union[np.ndarray, np.int_],
    fp: Union[np.ndarray, np.int_],
    tn: Union[np.ndarray, np.int_],
    fn: Union[np.ndarray, np.int_],
    task_type: Literal["binary", "multiclass", "multilabel"],
    average: Literal["micro", "macro", "weighted", None],
    zero_division: Literal["warn", 0, 1] = "warn",
) -> Union[np.ndarray, float]:
    """Compute accuracy score per class or sample and apply average.

    Parameters
    ----------
        tp : numpy.ndarray or int
            The number of true positives.
        fp : numpy.ndarray or int
            The number of false positives.
        tn : numpy.ndarray or int
            The number of true negatives.
        fn : numpy.ndarray or int
            The number of false negatives.
        task_type : Literal["binary", "multiclass", "multilabel"]
            The type of task for the input data. One of 'binary', 'multiclass'
            or 'multilabel'.
        average : Literal["micro", "macro", "weighted", None]
            The type of averaging to apply to the accuracy scores. One of
            'micro', 'macro', 'weighted' or None.
        zero_division : Literal["warn", 0, 1]
            Sets the value to return when there is a zero division. If set to "warn",
            this acts as 0, but warnings are also raised.

    Returns
    -------
        accuracy : numpy.ndarray or float
            The average accuracy score if 'average' is not None, otherwise the
            accuracy score per class.

    """
    # pylint: disable=invalid-name
    if average == "micro":
        tp = np.array(np.sum(tp))
        fn = np.array(np.sum(fn))
        numerator = tp
        denominator = tp + fn
        if task_type == "multilabel":
            fp = np.array(np.sum(fp))
            tn = np.array(np.sum(tn))
            numerator = tp + tn
            denominator = tp + fp + fn + tn
    else:
        if task_type in ["binary", "multilabel"]:
            numerator = tp + tn
            denominator = tp + fp + fn + tn
        else:
            numerator = tp
            denominator = tp + fn

    score = _prf_divide(
        np.array(numerator) if np.isscalar(numerator) else numerator,
        np.array(denominator) if np.isscalar(denominator) else denominator,
        metric="accuracy",
        modifier="true",
        average=average,
        warn_for=("accuracy",),
        zero_division=zero_division,
    )

    if average in ["macro", "weighted"]:
        weights = None
        if average == "weighted":
            weights = tp + fn

        if weights is not None and np.sum(weights) == 0:
            return (
                np.zeros_like(score)
                if zero_division in ["warn", 0]
                else np.ones_like(score)
            )

        return np.average(score, weights=weights)

    return _get_value_if_singleton_array(score)


def binary_accuracy(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    pos_label: int = 1,
    threshold: float = 0.5,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> float:
    """Compute accuracy score for binary classification tasks.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Estimated targets (predictions) as returned by a classifier.
        pos_label : int, default=1
            The label of the positive class. Can be 0 or 1.
        threshold : float, default=0.5
            The threshold value for converting probability or logit scores to
            binary. A sigmoid function is first applied to logits to convert them
            to probabilities.
        zero_division : Literal["warn", 0, 1], default="warn"
            Sets the value to return when there is a zero division. If set to ``warn``,
            this acts as 0, but warnings are also raised.

    Returns
    -------
        float
            The accuracy score.

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import binary_accuracy
        >>> target = [0, 1, 0, 1]
        >>> preds = [0, 1, 1, 1]
        >>> binary_accuracy(target, preds)
        0.75
        >>> target = [0, 1, 0, 1]
        >>> preds = [0.1, 0.9, 0.8, 0.4]
        >>> binary_accuracy(target, preds, threshold=0.5)
        0.5

    """
    _binary_stat_scores_args_check(threshold=threshold, pos_label=pos_label)

    target, preds = _binary_stat_scores_format(target, preds, threshold)

    # pylint: disable=invalid-name
    tp, fp, tn, fn = _binary_stat_scores_update(target, preds, pos_label=pos_label)
    return _accuracy_reduce(
        tp, fp, tn, fn, task_type="binary", average=None, zero_division=zero_division
    )


def multiclass_accuracy(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    num_classes: int,
    top_k: Optional[int] = None,
    average: Literal["micro", "macro", "weighted", None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> Union[float, np.ndarray]:
    """Compute the accuracy score for multiclass classification problems.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Estimated targets (predictions) as returned by a classifier.
        num_classes : int
            Number of classes in the dataset.
        top_k : int, default=None
            Number of highest probability predictions or logits to consider when
            computing the accuracy score.
        average : Literal["micro", "macro", "weighted", None], default=None
            If not None, this determines the type of averaging performed on the data:
                - ``micro``: Calculate metrics globally by counting the total
                  true positives, false negatives, false positives and true
                  negatives.
                - ``macro``: Calculate metrics for each class, and find their
                  unweighted mean. This does not take class imbalance into account.
                - ``weighted``: Calculate metrics for each class, and find their
                  average, weighted by support (the number of true instances for
                  each class). This alters ``macro`` to account for class imbalance.
        zero_division : Literal["warn", 0, 1], default="warn"
            Sets the value to return when there is a zero division. If set to ``warn``,
            this acts as 0, but warnings are also raised.

    Returns
    -------
        float or numpy.ndarray
            The average accuracy score as a float if ``average`` is not None,
            otherwise a numpy array of accuracy scores per class/label.

    Raises
    ------
        ValueError
            If ``average`` is not one of ``micro``, ``macro``, ``weighted`` or ``None``.

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import multiclass_accuracy
        >>> target = [0, 1, 2, 2, 2]
        >>> preds = [0, 0, 2, 2, 1]
        >>> multiclass_accuracy(target, preds, num_classes=3)
        array([1.        , 0.        , 0.66666667])
        >>> multiclass_accuracy(target, preds, num_classes=3, average="micro")
        0.6
        >>> multiclass_accuracy(target, preds, num_classes=3, average="macro")
        0.5555555555555555
        >>> multiclass_accuracy(target, preds, num_classes=3, average="weighted")
        0.6

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

    return _accuracy_reduce(
        tp,
        fp,
        tn,
        fn,
        task_type="multiclass",
        average=average,
        zero_division=zero_division,
    )


def multilabel_accuracy(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    num_labels: int,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    average: Literal["micro", "macro", "weighted", None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> Union[float, np.ndarray]:
    """Compute the accuracy score for multilabel-indicator targets.

    Parameters
    ----------
        target : array-like of shape (num_samples, num_labels)
            Ground truth (correct) target values.
        preds : array-like of shape (num_samples, num_labels)
            Estimated targets as returned by a classifier.
        num_labels : int
            Number of labels in the multilabel classification task.
        threshold : float, default=0.5
            Threshold value for binarizing the output of the classifier.
        top_k : int, optional, default=None
            The number of highest probability or logit predictions considered
            to find the correct label. Only works when ``preds`` contains
            probabilities/logits.
        average : Literal['micro', 'macro', 'weighted', None], default=None
            If None, return the accuracy score per label, otherwise this determines
            the type of averaging performed on the data:
                - ``micro``: Calculate metrics globally by counting the total
                  true positives, false negatives, true negatives and false positives.
                - ``macro``: Calculate metrics for each label, and find their unweighted
                  mean. This does not take label imbalance into account.
                - ``weighted``: Calculate metrics for each label, and find their
                  average, weighted by support (the number of true instances for
                  each label).
        zero_division : Literal['warn', 0, 1], default="warn"
            Sets the value to return when there is a zero division. If set to ``warn``,
            this acts as 0, but warnings are also raised.

    Returns
    -------
        float or numpy.ndarray
            The average accuracy score as a flot if ``average`` is not None,
            otherwise a numpy array of accuracy scores per label.

    Raises
    ------
        ValueError
            If ``average`` is not one of ``micro``, ``macro``, ``weighted``,
            or ``None``.

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import multilabel_accuracy
        >>> target = [[0, 1, 1], [1, 0, 0]]
        >>> preds = [[0, 1, 0], [1, 0, 1]]
        >>> multilabel_accuracy(target, preds, num_labels=3, average=None)
        array([1., 1., 0.])
        >>> multilabel_accuracy(target, preds, num_labels=3, average="micro")
        0.6666666666666666

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

    return _accuracy_reduce(
        tp,
        fp,
        tn,
        fn,
        task_type="multilabel",
        average=average,
        zero_division=zero_division,
    )


def accuracy(  # pylint: disable=too-many-arguments
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
    """Compute accuracy score for different classification tasks.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Estimated targets (predictions) as returned by a classifier.
        task : Literal["binary", "multiclass", "multilabel"]
            The type of task for the input data. One of 'binary', 'multiclass'
            or 'multilabel'.
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
            If ``None``, return the recall score for each label/class. Otherwise,
            use one of the following options to compute the average score:
                - ``micro``: Calculate metrics globally by counting the total true
                  positives. false positives, true negatives and false negatives.
                - ``macro``: Calculate metrics for each class/label, and find their
                  unweighted mean. This does not take label imbalance into account.
                - ``weighted``: Calculate metrics for each label/class, and find
                  their average weighted by support (the number of true instances
                  for each label/class). This alters ``macro`` to account for
                  label/class imbalance.
        zero_division : Literal["warn", 0, 1], default="warn"
            Sets the value to return when there is a zero division. If set to ``warn``,
            this acts as 0, but warnings are also raised.

    Returns
    -------
        accuracy_score : float or numpy.ndarray
            The average accuracy score as a float if ``average`` is not None,
            otherwise a numpy array of accuracy scores per class/label.

    Raises
    ------
        ValueError
            If ``task`` is not one of ``binary``, ``multiclass`` or ``multilabel``.
        AssertionError
            If ``task`` is ``multiclass`` and ``num_classes`` is not provided or is
            less than 0.
        AssertionError
            If ``task`` is ``multilabel`` and ``num_labels`` is not provided or is
            less than 0.

    Examples (binary)
    -----------------
        >>> from cyclops.evaluation.metrics.functional import accuracy
        >>> target = [0, 1, 0, 1]
        >>> preds = [0, 1, 1, 1]
        >>> accuracy(target, preds, task="binary")
        0.75

    Examples (multiclass)
    ---------------------
        >>> target = [0, 1, 2, 2, 2]
        >>> preds = [0, 0, 2, 2, 1]
        >>> accuracy(target, preds, task="multiclass", num_classes=3, average="micro")
        0.6

    Examples (multilabel)
        >>> target = [[0, 1, 1], [1, 0, 0]]
        >>> preds = [[0, 1, 0], [1, 0, 1]]
        >>> accuracy(target, preds, task="multilabel", num_labels=3, average="mcro")
        0.6666666666666666

    """
    if task == "binary":
        accuracy_score = binary_accuracy(
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
        accuracy_score = multiclass_accuracy(
            target,
            preds,
            num_classes=num_classes,
            top_k=top_k,
            average=average,  # type: ignore
            zero_division=zero_division,
        )
    elif task == "multilabel":
        assert (
            isinstance(num_labels, int) and num_labels > 0
        ), "Number of labels must be specified for multilabel classification."
        accuracy_score = multilabel_accuracy(
            target,
            preds,
            num_labels=num_labels,
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
    return accuracy_score
