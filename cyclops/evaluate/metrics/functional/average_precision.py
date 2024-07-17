"""Functions for computing average precision (AUPRC) for classification tasks."""

from typing import Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from cyclops.evaluate.metrics.functional.precision_recall_curve import (
    _binary_precision_recall_curve_compute,
    _binary_precision_recall_curve_format,
    _binary_precision_recall_curve_update,
)


def _binary_average_precision_compute(
    state: Union[
        Tuple[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        npt.NDArray[np.int_],
    ],
    thresholds: Optional[npt.NDArray[np.float_]],
    pos_label: Optional[int] = None,
) -> float:
    """Compute average precision for binary classification task.

    Parameters
    ----------
    state : Tuple or numpy.ndarray
        State from which the precision-recall curve can be computed. Can be
        either a tuple of (target, preds) or a multi-threshold confusion matrix.
    thresholds : numpy.ndarray
        Thresholds used for computing the precision and recall scores. If not None,
        must be a 1D numpy array of floats in the [0, 1] range and monotonically
        increasing.
    pos_label : int
        The label of the positive class.

    Returns
    -------
    float
        The average precision score.

    Raises
    ------
    ValueError
        If ``thresholds`` is None.

    """
    precision, recall, _ = _binary_precision_recall_curve_compute(
        state,
        thresholds,
        pos_label,
    )
    return -np.sum(np.diff(recall) * np.array(precision)[:-1])  # type: ignore


def binary_average_precision(
    target: npt.ArrayLike,
    preds: npt.ArrayLike,
    thresholds: Optional[npt.NDArray[np.float_]],
    pos_label: int = 1,
) -> float:
    """Compute average precision for binary classification task.

    Parameters
    ----------
    target : npt.ArrayLike
        Target values.
    preds : npt.ArrayLike
        Predicted values.
    thresholds : numpy.ndarray
        Thresholds used for computing the precision and recall scores. If not None,
        must be a 1D numpy array of floats in the [0, 1] range and monotonically
        increasing.
    pos_label : int
        The label of the positive class.

    Returns
    -------
    float
        The average precision score.

    Raises
    ------
    ValueError
        If ``thresholds`` is None.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.functional import binary_average_precision
    >>> target = [0, 1, 1, 0]
    >>> preds = [0, 0.5, 0.7, 0.8]
    >>> binary_average_precision(target, preds, thresholds=None)
    0.5833333333333333

    """
    target, preds = _binary_precision_recall_curve_format(target, preds, pos_label)
    state = _binary_precision_recall_curve_update(target, preds)
    return _binary_average_precision_compute(state, thresholds, pos_label)


def average_precision(
    target: npt.ArrayLike,
    preds: npt.ArrayLike,
    task: Literal["binary", "multiclass", "multilabel"],
    thresholds: Optional[npt.NDArray[np.float_]] = None,
    pos_label: int = 1,
    num_classes: Optional[int] = None,
    top_k: Optional[int] = None,
    num_labels: Optional[int] = None,
    average: Literal["micro", "macro", "weighted", None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> Union[npt.NDArray[np.float_], float]:
    """Compute average precision for classification tasks.

    Parameters
    ----------
    target : npt.ArrayLike
        Target values.
    preds : npt.ArrayLike
        Predicted values.
    task : {"binary", "multiclass", "multilabel"}
        The task type.
    pos_label : int
        The label of the positive class.
    num_classes : int
        The number of classes.
    threshold : float
        The threshold for converting probabilities to binary predictions.
    top_k : int
        The number of top predictions to consider for multilabel classification.
    num_labels : int
        The number of labels for multilabel classification.
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
    float or numpy.ndarray
        The average precision score(s).

    Raises
    ------
    ValueError
        If ``task`` is not one of ``binary``, ``multiclass`` or ``multilabel``.

    """
    if task == "binary":
        return binary_average_precision(target, preds, thresholds, pos_label)
    if task == "multiclass":
        raise NotImplementedError("Multiclass average precision is not implemented.")
    if task == "multilabel":
        raise NotImplementedError("Multilabel average precision is not implemented.")

    raise ValueError(
        "Expected argument `task` to be either 'binary', 'multiclass' or "
        f"'multilabel', but got {task}",
    )
