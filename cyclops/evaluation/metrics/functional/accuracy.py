"""Functions for computing accuracy scores for classification tasks."""
from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics._classification import _prf_divide

from cyclops.evaluation.metrics.functional.stat_scores import (
    _binary_stat_scores_update,
    _multiclass_stat_scores_update,
    _multilabel_stat_scores_update,
)


def _accuracy_reduce(  # pylint: disable=too-many-arguments
    tp: Union[np.ndarray, int],
    fp: Union[np.ndarray, int],
    tn: Union[np.ndarray, int],
    fn: Union[np.ndarray, int],
    task_type: Literal["binary", "multiclass", "multilabel"],
    average: Literal["micro", "macro", "weighted", "samples", None],
    sample_weight: Optional[ArrayLike] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> Union[np.ndarray, float]:
    """Compute accuracy score per class or sample and apply average.

    Parameters
    ----------
        tp : np.ndarray or int
            The number of true positives.
        fp : np.ndarray or int
            The number of false positives.
        tn : np.ndarray or int
            The number of true negatives.
        fn : np.ndarray or int
            The number of false negatives.
        task_type : Literal["binary", "multiclass", "multilabel"]
            The type of task for the input data. One of 'binary', 'multiclass'
            or 'multilabel'.
        average : Literal["micro", "macro", "weighted", "samples", None]
            The type of averaging to apply to the accuracy scores. One of
            'micro', 'macro', 'weighted', 'samples' or None.
        sample_weight : ArrayLike, default=None
            The weight to apply to each sample if averaging is ``samples``.
        zero_division : Literal["warn", 0, 1]
            Sets the value to return when there is a zero division. If set to "warn",
            this acts as 0, but warnings are also raised.

    Returns
    -------
        accuracy : Union[np.ndarray, float]
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

    if average in ["macro", "weighted", "samples"]:
        weights = None
        if average == "weighted":
            weights = tp + fn
        elif average == "samples":
            weights = sample_weight

        if weights is not None and np.sum(weights) == 0:
            return (
                np.zeros_like(score)
                if zero_division in ["warn", 0]
                else np.ones_like(score)
            )

        return np.average(score, weights=weights)

    return score


def accuracy(  # pylint: disable=too-many-arguments
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
            The class to report. Can be 0 or 1.
        num_classes : int, optional
            The number of classes in the dataset. Required for multiclass tasks.
        threshold : float, default=0.5
            The threshold value for converting probability or logit scores to
            binary. A sigmoid function is first applied to logits to convert them
            to probabilities.
        top_k : int, default=None
            The number of top predictions to consider when computing accuracy.
            Required for multiclass and multilabel tasks.
        num_labels : int, optional
            The number of labels in the dataset. Required for multilabel tasks.
        average : Literal["micro", "macro", "weighted", "samples", None], default=None
            The type of averaging to apply to the accuracy scores. One of:
                - ``micro``: Calculate metrics globally by counting the total
                  true positives, false negatives and false positives.
                - ``macro``: Calculate metrics for each class, and find their
                  unweighted mean. This does not take class imbalance into account.
                - ``weighted``: Calculate metrics for each class, and find their
                  average, weighted by support (the number of true instances for
                  each class). This alters ``macro`` to account for class imbalance.
                - ``samples``: Calculate metrics for each instance, and find their
                  average (only meaningful for multilabel classification).
                - ``None``: The scores for each class or label are returned.
        sample_weight : ArrayLike, default=None
            Sample weights.
        zero_division : Literal["warn", 0, 1], default="warn"
            Sets the value to return when there is a zero division. If set to ``warn``,
            this acts as 0, but warnings are also raised.

    Returns
    -------
        accuracy_score : Union[float, np.ndarray]
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

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import accuracy
        >>> target = [0, 1, 0, 1]
        >>> preds = [0, 1, 1, 1]
        >>> accuracy(target, preds, task="binary")
        0.75

        >>> target = [0, 1, 2, 2, 2]
        >>> preds = [0, 0, 2, 2, 1]
        >>> accuracy(target, preds, task="multiclass", num_classes=3, average="micro")
        0.6

        >>> target = np.array([[0, 1, 1], [1, 0, 0]])
        >>> preds = np.array([[0, 1, 0], [1, 0, 1]])
        >>> accuracy(target, preds, task="multilabel", num_labels=3, average="mcro")
        0.6666666666666666

    """
    if task == "binary":
        accuracy_score = binary_accuracy(
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
        accuracy_score = multiclass_accuracy(
            target,
            preds,
            num_classes=num_classes,
            top_k=top_k,
            sample_weight=sample_weight,
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
            sample_weight=sample_weight,
            average=average,
            zero_division=zero_division,
        )
    else:
        raise ValueError(
            f"Task {task} is not supported, expected one of 'binary', 'multiclass'"
            " or 'multilabel'"
        )
    return accuracy_score


def binary_accuracy(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    pos_label: int = 1,
    threshold: float = 0.5,
    sample_weight: Optional[ArrayLike] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> float:
    """Compute binary accuracy score.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Estimated targets (predictions) as returned by a classifier.
        pos_label : int, default=1
            The class to report. Can be 0 or 1.
        threshold : float, default=0.5
            The threshold value for converting probability or logit scores to
            binary. A sigmoid function is first applied to logits to convert them
            to probabilities.
        sample_weight : ArrayLike, default=None
            Sample weights.
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
    # pylint: disable=invalid-name
    tp, fp, tn, fn = _binary_stat_scores_update(
        target,
        preds,
        pos_label=pos_label,
        threshold=threshold,
        sample_weight=sample_weight,
    )
    return _accuracy_reduce(
        tp, fp, tn, fn, task_type="binary", average=None, zero_division=zero_division
    )


def multiclass_accuracy(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    num_classes: int,
    top_k: Optional[int] = None,
    sample_weight: Optional[ArrayLike] = None,
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
        sample_weight : ArrayLike, default=None
            Sample weights.
        average : Literal["micro", "macro", "weighted", None], default=None
            If not None, this determines the type of averaging performed on the data:
                - ``micro``: Calculate metrics globally by counting the total
                  true positives, false negatives and false positives.
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
        float or np.ndarray
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
    if average not in ["micro", "macro", "weighted", None]:
        raise ValueError(
            f"Argument average has to be one of 'micro', 'macro', 'weighted', "
            f"or None, got {average}."
        )

    # pylint: disable=invalid-name
    tp, fp, tn, fn = _multiclass_stat_scores_update(
        target,
        preds,
        num_classes,
        top_k=top_k,
        classwise=True,
        sample_weight=sample_weight,
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
    sample_weight: Optional[ArrayLike] = None,
    average: Literal["micro", "macro", "samples", "weighted", None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> Union[float, np.ndarray]:
    """Compute the accuracy score for multilabel-indicator input.

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
        sample_weight : array-like of shape (num_samples,), default=None
            Sample weights.
        average : Literal['micro', 'macro', 'samples', 'weighted', None], default=None
            If None, return the accuracy score per label, otherwise this determines
            the type of averaging performed on the data:
                - ``micro``: Calculate metrics globally by counting the total
                  true positives, false negatives and false positives.
                - ``macro``: Calculate metrics for each label, and find their unweighted
                  mean. This does not take label imbalance into account.
                - ``weighted``: Calculate metrics for each label, and find their
                  average, weighted by support (the number of true instances for
                  each label).
                - ``samples``: Calculate metrics for each instance, and find their
                  average.
        zero_division : Literal['warn', 0, 1], default="warn"
            Sets the value to return when there is a zero division. If set to ``warn``,
            this acts as 0, but warnings are also raised.

    Returns
    -------
        float or np.ndarray
            The average accuracy score as a flot if ``average`` is not None,
            otherwise a numpy array of accuracy scores per label.

    Raises
    ------
        ValueError
            If ``average`` is not one of ``micro``, ``macro``, ``weighted``,
            ``samples``, or ``None``.

    Examples
    --------
        >>> import numpy as np
        >>> from cyclops.evaluation.metrics.functional import multilabel_accuracy
        >>> target = np.array([[0, 1, 1], [1, 0, 0]])
        >>> preds = np.array([[0, 1, 0], [1, 0, 1]])
        >>> multilabel_accuracy(target, preds, num_labels=3, average=None)
        array([1., 1., 0.])

        >>> multilabel_accuracy(target, preds, num_labels=3, average="micro")
        0.6666666666666666

    """
    if average not in ["micro", "macro", "samples", "weighted", None]:
        raise ValueError(
            f"Argument `average` has to be one of 'micro', 'macro', 'samples', "
            f"'weighted', or None, got `{average}.`"
        )

    # pylint: disable=invalid-name
    tp, fp, tn, fn = _multilabel_stat_scores_update(
        target,
        preds,
        num_labels,
        threshold=threshold,
        top_k=top_k,
        reduce="samples" if average == "samples" else "macro",
        sample_weight=sample_weight,
    )

    return _accuracy_reduce(
        tp,
        fp,
        tn,
        fn,
        task_type="multilabel",
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )
