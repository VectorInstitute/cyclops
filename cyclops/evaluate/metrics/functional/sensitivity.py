"""Functions for computing sensitivity scores on different input types."""

from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike

from cyclops.evaluate.metrics.functional.precision_recall import (
    binary_recall,
    multiclass_recall,
    multilabel_recall,
    recall,
)


def binary_sensitivity(
    target: ArrayLike,
    preds: ArrayLike,
    pos_label: int = 1,
    threshold: float = 0.5,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> float:
    """Compute sensitvity score for binary classification problems.

    Sensitivity is the recall of the positive class in a binary classification
    problem.

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
            sensitivity score.

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import binary_sensitivity
        >>> target = [0, 1, 0, 1]
        >>> preds = [0, 1, 1, 0]
        >>> binary_sensitivity(target, preds)
        0.5

    """
    return binary_recall(
        target,
        preds,
        pos_label=pos_label,
        threshold=threshold,
        zero_division=zero_division,
    )


def multiclass_sensitivity(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    num_classes: int,
    top_k: Optional[int] = None,
    average: Literal["micro", "macro", "weighted", None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
):
    """Compute sensitivity score for multiclass classification problems.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Predictions as returned by a classifier.
        num_classes : int
            Total number of classes in the dataset.
        top_k : Optional[int]
            If given, and predictions are probabilities/logits, the sensitivity will
            be computed only for the top k classes. Otherwise, ``top_k`` will be
            set to 1.
        average : Literal["micro", "macro", "weighted", None], default=None
            Average to apply. If None, return scores for each class. Otherwise,
            use one of the following options to compute the average score:
                - ``micro``: Calculate metrics globally by counting the total true
                  positives and false negatives.
                - ``macro``: Calculate metrics for each label, and find their
                  unweighted mean. This does not take label imbalance into
                  account.
                - ``weighted``: Calculate metrics for each label, and find their
                  average weighted by support (the number of true instances
                  for each label). This alters "macro" to account for label
                  imbalance.
        zero_division : Literal["warn", 0, 1], default="warn"
            Value to return when there are no true positives or true negatives.
            If set to ``warn``, this acts as 0, but warnings are also raised.

    Returns
    -------
        float or numpy.ndarray
            Sensitivity score. If ``average`` is None, return a numpy.ndarray of
            sensitivity scores for each class.


    Raises
    ------
        ValueError
            If ``average`` is not one of ``micro``, ``macro``, ``weighted``
            or ``None``.

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import multiclass_sensitivity
        >>> target = [0, 1, 2, 0, 1, 2]
        >>> preds = [[0.4, 0.1, 0.5], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6],
        ...     [0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.2, 0.2, 0.6]]
        >>> multiclass_sensitivity(target, preds, num_classes=3, average="macro")
        0.8333333333333334

    """
    return multiclass_recall(
        target,
        preds,
        num_classes,
        top_k=top_k,
        average=average,
        zero_division=zero_division,
    )


def multilabel_sensitivity(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    num_labels: int,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    average: Literal["micro", "macro", "weighted", None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
):
    """Compute sensitivity score for multilabel classification tasks.

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
            If ``None``, return the sensitivity score for each class. Otherwise,
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
            Sensitivity score. If ``average`` is None, return a numpy.ndarray of
            sensitivity scores for each label.

    Raises
    ------
        ValueError
            If ``average`` is not one of ``micro``, ``macro``, ``weighted``
            or ``None``.


    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import multilabel_sensitivity
        >>> target = [1, 1, 2, 0, 2, 2]
        >>> preds = [1, 2, 2, 0, 2, 0]
        >>> multilabel_sensitivity(target, preds, num_classes=3)
        array([1.        , 0.5       , 0.66666667])

    """
    return multilabel_recall(
        target,
        preds,
        num_labels,
        threshold=threshold,
        top_k=top_k,
        average=average,
        zero_division=zero_division,
    )


def sensitivity(  # pylint: disable=too-many-arguments
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
    """Compute sensitivity score for different classification tasks.

    Sensitivity is the ratio tp / (tp + fn) where tp is the number of true positives
    and fn the number of false negatives. The sensitivity is intuitively the ability
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
        average : Literal["micro", "macro", "weighted", None], default=None
            Average to apply. If None, return scores for each class/label. Otherwise,
            use one of the following options to compute the average score:
                - ``micro``: Calculate metrics globally by counting the total true
                  positives and false negatives.
                - ``macro``: Calculate metrics for each class/label, and find their
                  unweighted mean. This does not take class/label imbalance into
                  account.
                - ``weighted``: Calculate metrics for each class/label, and find
                  their average weighted by support (the number of true instances
                  for each class/label). This alters ``macro`` to account for
                  class/label imbalance.
        zero_division : Literal["warn", 0, 1], default="warn"
            Value to return when there are no true positives or true negatives.
            If set to ``warn``, this acts as 0, but warnings are also raised.

    Returns
    -------
        float or numpy.ndarray
            Sensitivity score. If ``average`` is not None or ``task`` is ``binary``,
            return a float. Otherwise, return a numpy.ndarray of sensitivity scores
            for each class/label.

    Raises
    ------
        ValueError
            If ``task`` is not one of ``binary``, ``multiclass`` or ``multilabel``.

    Examples (binary)
    -----------------
        >>> from cyclops.evaluation.metrics.functional import sensitivity
        >>> target = [0, 1, 1, 0, 1]
        >>> preds = [0.4, 0.2, 0.0, 0.6, 0.9]
        >>> sensitivity(target, preds, task="binary")
        0.3333333333333333

    Examples (multiclass)
    ---------------------
        >>> from cyclops.evaluation.metrics.functional import sensitivity
        >>> target = [1, 1, 2, 0, 2, 2]
        >>> preds = [1, 2, 2, 0, 2, 0]
        >>> sensitivity(target, preds, task="multiclass", num_classes=3)
        array([1.        , 0.5       , 0.66666667])

    Examples (multilabel)
    ---------------------
        >>> from cyclops.evaluation.metrics.functional import sensitivity
        >>> target = [[1, 0, 1], [0, 1, 0]]
        >>> preds = [[0.4, 0.2, 0.0], [0.6, 0.9, 0.1]]
        >>> sensitivity(target, preds, task="multilabel", num_labels=3)
        array([0., 1., 0.])

    """
    return recall(
        target,
        preds,
        task,
        pos_label=pos_label,
        num_classes=num_classes,
        threshold=threshold,
        top_k=top_k,
        num_labels=num_labels,
        average=average,
        zero_division=zero_division,
    )
