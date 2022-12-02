"""Functions for computing precision and recall scores on different input types."""

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


def _precision_recall_reduce(  # pylint: disable=invalid-name, too-many-arguments
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    metric: Literal["precision", "recall"],
    average: Literal["micro", "macro", "weighted", None],
    zero_division: Literal["warn", 0, 1] = "warn",
) -> Union[np.ndarray, float]:
    """Compute precision or recall scores and apply specified average.

    Parameters
    ----------
        tp : numpy.ndarray
            True positives.
        fp : numpy.ndarray
            False positives.
        fn : numpy.ndarray
            False negatives.
        metric : Literal["precision", "recall"]
            Metric to compute.
        average : Literal["micro", "macro", "weighted", None]
            Average to apply. If None, return scores for each class.
        zero_division : Literal["warn", 0, 1]
            Value to return when there are no true positives or true negatives.
            If set to "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
        scores : numpy.ndarray or float
            Precision or recall scores. If ``average`` is None, return scores for
            each class as a numpy.ndarray. Otherwise, return the average as a
            float.

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
    else:
        weights = None

    if average is not None and score.ndim != 0 and len(score) > 1:
        result = np.average(score, weights=weights)
    else:
        result = _get_value_if_singleton_array(score)

    return result


def binary_precision(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    pos_label: int = 1,
    threshold: float = 0.5,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> float:
    """Compute precision score for binary classification.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Predictions as returned by a classifier.
        pos_label : int, default=1
            The label of the positive class.
        threshold : float, default=0.5
            Threshold for deciding the positive class.
        zero_division : Literal["warn", 0, 1], default="warn"
            Value to return when there is a zero division. If set to "warn", this
            acts as 0, but warnings are also raised.

    Returns
    -------
        float
            Precision score.

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import binary_precision
        >>> target = [0, 1, 0, 1]
        >>> preds = [0, 1, 1, 1]
        >>> binary_precision(target, preds)
        0.6666666666666666
        >>> target = [[0, 1, 0, 1], [0, 0, 1, 1]]
        >>> preds = [[0.1, 0.9, 0.8, 0.2], [0.2, 0.3, 0.6, 0.1]]
        >>> binary_precision(target, preds)
        0.6666666666666666

    """
    _binary_stat_scores_args_check(threshold=threshold, pos_label=pos_label)

    target, preds = _binary_stat_scores_format(target, preds, threshold)

    # pylint: disable=invalid-name
    tp, fp, _, fn = _binary_stat_scores_update(target, preds, pos_label=pos_label)

    return _precision_recall_reduce(
        tp,
        fp,
        fn,
        metric="precision",
        average=None,
        zero_division=zero_division,
    )


def multiclass_precision(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    num_classes: int,
    top_k: Optional[int] = None,
    average: Literal["micro", "macro", "weighted", None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> Union[float, np.ndarray]:
    """Compute precision score for multiclass classification tasks.

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
           If ``None``, return the precision score for each class. Otherwise,
           use one of the following options to compute the average precision score:
                - ``micro``: Calculate metric globally from the total count of true
                  positives and false positives.
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
        precision : float or numpy.ndarray
            Precision score. If ``average`` is None, return a numpy.ndarray of
            precision scores for each class.

    Raises
    ------
        ValueError
            If ``average`` is not one of ``micro``, ``macro``, ``weighted``
            or ``None``.

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import multiclass_precision
        >>> target = [0, 1, 2, 0]
        >>> preds = [0, 2, 1, 0]
        >>> multiclass_precision(target, preds, num_classes=3)
        array([1., 0., 0.])

    """
    _check_average_arg(average)

    target, preds = _multiclass_stat_scores_format(
        target, preds, num_classes=num_classes, top_k=top_k
    )

    # pylint: disable=invalid-name
    tp, fp, _, fn = _multiclass_stat_scores_update(
        target,
        preds,
        num_classes,
        classwise=True,
    )

    return _precision_recall_reduce(
        tp,
        fp,
        fn,
        metric="precision",
        average=average,
        zero_division=zero_division,
    )


def multilabel_precision(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    num_labels: int,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    average: Literal["micro", "macro", "weighted", None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
):
    """Compute precision score for multilabel classification tasks.

    The input is expected to be an array-like of shape (N, L), where N is the
    number of samples and L is the number of labels. The input is expected to
    be a binary array-like, where 1 indicates the presence of a label and 0
    indicates its absence.

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
        average : Literal["micro", "macro", "weighted", None], default=None
            If ``None``, return the precision score for each label. Otherwise,
            use one of the following options to compute the average precision score:
                - ``micro``: Calculate metric globally from the total count of true
                  positives and false positives.
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
        precision: float or numpy.ndarray
            Precision score. If ``average`` is None, return a numpy.ndarray of
            precision scores for each label.

    Raises
    ------
        ValueError
            If average is not one of ``micro``, ``macro``, ``weighted``,
            or ``None``.

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import multilabel_precision
        >>> target = [[0, 1], [1, 1]]
        >>> preds = [[0.1, 0.9], [0.2, 0.8]]
        >>> multilabel_precision(target, preds, num_labels=2)
        array([0., 1. ])

    """
    _check_average_arg(average)

    target, preds = _multilabel_stat_scores_format(
        target, preds, num_labels=num_labels, threshold=threshold, top_k=top_k
    )

    # pylint: disable=invalid-name
    tp, fp, _, fn = _multilabel_stat_scores_update(
        target,
        preds,
        num_labels,
        labelwise=True,
    )

    return _precision_recall_reduce(
        tp,
        fp,
        fn,
        metric="precision",
        average=average,
        zero_division=zero_division,
    )


def precision(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    task: Literal["binary", "multiclass", "multilabel"],
    pos_label: int = 1,
    num_classes: Optional[int] = None,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    num_labels: Optional[int] = None,
    average: Literal["micro", "macro", "weighted", None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> Union[float, np.ndarray]:
    """Compute precision score for different classification tasks.

    Precision is the ratio of correctly predicted positive observations to the
    total predicted positive observations.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Predictions as returned by a classifier.
        task : Literal["binary", "multiclass", "multilabel"]
            Task type.
        pos_label : int
            Label of the positive class. Only used for binary classification.
        num_classes : Optional[int]
            Number of classes. Only used for multiclass classification.
        threshold : float
            Threshold for positive class predictions. Default is 0.5.
        top_k : Optional[int]
            Number of highest probability or logits predictions to consider when
            computing multiclass or multilabel metrics. Default is None.
        num_labels : Optional[int]
            Number of labels. Only used for multilabel classification.
        average : Literal["micro", "macro", "weighted", None]
            Average to apply. If None, return scores for each class. Default is
            None.
            One of:
                - ``micro``: Calculate metrics globally by counting the total true
                  positives and and false positives.
                - ``macro``: Calculate metrics for each label/class, and find their
                  unweighted mean. This does not take label imbalance into account.
                - ``weighted``: Calculate metrics for each label, and find their
                  average weighted by support (the number of true instances for
                  each label). This alters ``macro`` to account for label imbalance.
        zero_division : Literal["warn", 0, 1]
            Value to return when there are no true positives or true negatives.
            If set to ``warn``, this acts as 0, but warnings are also raised.

    Returns
    -------
        precision_score : numpy.ndarray or float
            Precision score. If ``average`` is not None or task is ``binary``,
            return a float. Otherwise, return a numpy.ndarray of precision scores
            for each class/label.

    Raises
    ------
        ValueError
            If task is not one of ``binary``, ``multiclass`` or ``multilabel``.

    Examples (binary)
    -----------------
        >>> from cyclops.evaluation.metrics.functional import precision
        >>> target = [0, 1, 1, 0]
        >>> preds = [0.1, 0.9, 0.8, 0.3]
        >>> precision(target, preds, task="binary")
        1.

    Examples (multiclass)
    ---------------------
        >>> from cyclops.evaluation.metrics.functional import precision
        >>> target = [0, 1, 2, 0, 1, 2]
        >>> preds = [[0.1, 0.6, 0.3], [0.05, 0.95, 0], [0.1, 0.8, 0.1],
        ...         [0.5, 0.3, 0.2],  [0.2, 0.5, 0.3], [0.2, 0.2, 0.6]]
        >>> precision(target, preds, task="multiclass", num_classes=3,
        ...     average="macro")
        0.8333333333333334

    Examples (multilabel)
    ---------------------
        >>> from cyclops.evaluation.metrics.functional import precision
        >>> target = [[0, 1], [1, 1]]
        >>> preds = [[0.1, 0.9], [0.2, 0.8]]
        >>> precision(target, preds, task="multilabel", num_labels=2,
        ...     average="macro")
        0.5

    """
    if task == "binary":
        precision_score = binary_precision(
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
        precision_score = multiclass_precision(
            target,
            preds,
            num_classes=num_classes,
            average=average,  # type: ignore
            top_k=top_k,
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
            zero_division=zero_division,
        )
    else:
        raise ValueError(
            f"Task '{task}' not supported, expected 'binary', 'multiclass' or "
            f"'multilabel'."
        )

    return precision_score


def binary_recall(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    pos_label: int = 1,
    threshold: float = 0.5,
    zero_division: Literal["warn", 0, 1] = "warn",
):
    """Compute recall score for binary classification.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Predictions as returned by a classifier.
        pos_label : int, default=1
            Label of the positive class.
        threshold : float, default=0.5
            Threshold for deciding the positive class.
        zero_division : Literal["warn", 0, 1], default="warn"
            Value to return when there is a zero division. If set to "warn", this
            acts as 0, but warnings are also raised.

    Returns
    -------
        float
            Recall score.

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import binary_recall
        >>> target = [0, 1, 0, 1]
        >>> preds = [0, 1, 1, 0]
        >>> binary_recall(target, preds)
        0.5

    """
    target, preds = _binary_stat_scores_format(target, preds, threshold)

    # pylint: disable=invalid-name
    tp, fp, _, fn = _binary_stat_scores_update(target, preds, pos_label=pos_label)

    return _precision_recall_reduce(
        tp,
        fp,
        fn,
        metric="recall",
        average=None,
        zero_division=zero_division,
    )


def multiclass_recall(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    num_classes: int,
    top_k: Optional[int] = None,
    average: Literal["micro", "macro", "weighted", None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
):
    """Compute recall score for multiclass classification.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Predictions as returned by a classifier.
        num_classes : int
            Number of classes.
        top_k : Optional[int]
            If given, and predictions are probabilities/logits, the recall will
            be computed only for the top k classes. Otherwise, ``top_k`` will be
            set to 1.
        average : Literal["micro", "macro", "weighted", None]
            Average to apply. If None, return scores for each class. Default is
            None. One of:
                - ``micro``: Calculate metrics globally by counting the total true
                  positives and false negatives.
                - ``macro``: Calculate metrics for each label, and find their
                  unweighted mean. This does not take label imbalance into
                  account.
                - ``weighted``: Calculate metrics for each label, and find their
                  average weighted by support (the number of true instances
                  for each label). This alters "macro" to account for label
                  imbalance.
        zero_division : Literal["warn", 0, 1]
            Value to return when there are no true positives or true negatives.
            If set to ``warn``, this acts as 0, but warnings are also raised.

    Returns
    -------
        float or numpy.ndarray
            Recall score. If ``average`` is None, return a numpy.ndarray of
            recall scores for each class.


    Raises
    ------
        ValueError
            If ``average`` is not one of ``micro``, ``macro``, ``weighted``
            or ``None``.

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import multiclass_recall
        >>> target = [0, 1, 2, 0, 1, 2]
        >>> preds = [[0.4, 0.1, 0.5], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6],
        ...     [0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.2, 0.2, 0.6]]
        >>> multiclass_recall(target, preds, num_classes=3, average="macro")
        0.8333333333333334

    """
    _check_average_arg(average)

    target, preds = _multiclass_stat_scores_format(
        target, preds, num_classes=num_classes, top_k=top_k
    )

    # pylint: disable=invalid-name
    tp, fp, _, fn = _multiclass_stat_scores_update(
        target,
        preds,
        num_classes,
        classwise=True,
    )

    return _precision_recall_reduce(
        tp,
        fp,
        fn,
        metric="recall",
        average=average,
        zero_division=zero_division,
    )


def multilabel_recall(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    num_labels: int,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    average: Literal["micro", "macro", "weighted", None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
):
    """Compute recall score for multilabel classification tasks.

    The input is expected to be an array-like of shape (N, L), where N is the
    number of samples and L is the number of labels. The input is expected to
    be a binary array-like, where 1 indicates the presence of a label and 0
    indicates its absence.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Predictions as returned by a classifier.
        num_labels : int
            Number of labels in the dataset.
        threshold : float, default=0.5
            Threshold for deciding the positive class.
        average : Literal["micro", "macro", "weighted", None], default=None
            If ``None``, return the recall score for each class. Otherwise,
            use one of the following options to compute the average score:
                - ``micro``: Calculate metric globally from the total count of true
                    positives and false negatives.
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
            Recall score. If ``average`` is None, return a numpy.ndarray of
            recall scores for each label.

    Raises
    ------
        ValueError
            If ``average`` is not one of ``micro``, ``macro``, ``weighted``
            or ``None``.


    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import multilabel_recall
        >>> target = [1, 1, 2, 0, 2, 2]
        >>> preds = [1, 2, 2, 0, 2, 0]
        >>> multilabel_recall(target, preds, num_classes=3)
        array([1.        , 0.5       , 0.66666667])

    """
    _check_average_arg(average)

    target, preds = _multilabel_stat_scores_format(
        target, preds, num_labels=num_labels, threshold=threshold, top_k=top_k
    )

    # pylint: disable=invalid-name
    tp, fp, _, fn = _multilabel_stat_scores_update(
        target,
        preds,
        num_labels,
        labelwise=True,
    )

    return _precision_recall_reduce(
        tp,
        fp,
        fn,
        metric="recall",
        average=average,
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
    average: Literal["micro", "macro", "weighted", None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> Union[float, np.ndarray]:
    """Compute recall score for different classification tasks.

    Recall is the ratio tp / (tp + fn) where tp is the number of true positives
    and fn the number of false negatives. The recall is intuitively the ability
    of the classifier to find all the positive samples.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Predictions as returned by a classifier.
        task : Literal["binary", "multiclass", "multilabel"]
            Task type.
        pos_label : int
            Label of the positive class. Only used for binary classification.
        num_classes : Optional[int]
            Number of classes. Only used for multiclass classification.
        threshold : float, default=0.5
            Threshold for positive class predictions.
        top_k : Optional[int]
            Number of highest probability or logits predictions to consider when
            computing multiclass or multilabel metrics. Default is None.
        num_labels : Optional[int]
            Number of labels. Only used for multilabel classification.
        average : Literal["micro", "macro", "weighted", None]
            Average to apply. If None, return scores for each class. Default is
            None.
            One of:
                - ``micro``: Calculate metrics globally by counting the total true
                  positives and false negatives.
                - ``macro``: Calculate metrics for each label, and find their
                  unweighted mean. This does not take label imbalance into account.
                - ``weighted``: Calculate metrics for each label, and find their
                  average weighted by support (the number of true instances for
                  each label). This alters ``macro`` to account for label imbalance.
        zero_division : Literal["warn", 0, 1]
            Value to return when there are no true positives or true negatives.
            If set to ``warn``, this acts as 0, but warnings are also raised.

    Returns
    -------
        recall_score : float or numpy.ndarray
            Recall score. If ``average`` is not None or ``task`` is ``binary``,
            return a float. Otherwise, return a numpy.ndarray of recall scores
            for each class/label.

    Raises
    ------
        ValueError
            If ``task`` is not one of ``binary``, ``multiclass`` or ``multilabel``.

    Examples (binary)
    -----------------
        >>> from cyclops.evaluation.metrics.functional import recall
        >>> target = [0, 1, 1, 0, 1]
        >>> preds = [0.4, 0.2, 0.0, 0.6, 0.9]
        >>> recall(target, preds, task="binary")
        0.3333333333333333

    Examples (multiclass)
    ---------------------
        >>> from cyclops.evaluation.metrics.functional import recall
        >>> target = [1, 1, 2, 0, 2, 2]
        >>> preds = [1, 2, 2, 0, 2, 0]
        >>> recall(target, preds, task="multiclass", num_classes=3)
        array([1.        , 0.5       , 0.66666667])

    Examples (multilabel)
    ---------------------
        >>> from cyclops.evaluation.metrics.functional import recall
        >>> target = [[1, 0, 1], [0, 1, 0]]
        >>> preds = [[0.4, 0.2, 0.0], [0.6, 0.9, 0.1]]
        >>> recall(target, preds, task="multilabel", num_labels=3)
        array([0., 1., 0.])

    """
    if task == "binary":
        recall_score = binary_recall(
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
        recall_score = multiclass_recall(
            target,
            preds,
            num_classes=num_classes,
            average=average,  # type: ignore
            top_k=top_k,
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
            zero_division=zero_division,
        )
    else:
        raise ValueError(
            f"Task '{task}' not supported, expected 'binary', 'multiclass' or "
            f"'multilabel'."
        )

    return recall_score
