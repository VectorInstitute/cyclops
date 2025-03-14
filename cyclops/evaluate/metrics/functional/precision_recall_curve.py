"""Functions for computing the precision-recall curve for different input types."""

from typing import Any, List, Literal, NamedTuple, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy as sp
from sklearn.metrics._ranking import _binary_clf_curve
from sklearn.preprocessing import label_binarize

from cyclops.evaluate.metrics.utils import (
    _check_thresholds,
    common_input_checks_and_format,
    sigmoid,
)


class PRCurve(NamedTuple):
    """Named tuple with Precision-Recall curve (Precision, Recall and thresholds)."""

    precision: Union[npt.NDArray[np.float_], List[npt.NDArray[np.float_]]]
    recall: Union[npt.NDArray[np.float_], List[npt.NDArray[np.float_]]]
    thresholds: Union[npt.NDArray[np.float_], List[npt.NDArray[np.float_]]]


def _format_thresholds(
    thresholds: Optional[Union[int, List[float], npt.NDArray[np.float_]]] = None,
) -> Optional[npt.NDArray[np.float_]]:
    """Format thresholds to be a 1D numpy array of floats."""
    if isinstance(thresholds, int):
        thresholds = np.linspace(0, 1, thresholds)
    elif isinstance(thresholds, list):
        thresholds = np.array(thresholds)

    return thresholds


def _ovr_multi_threshold_confusion_matrix(
    target: npt.NDArray[np.int_],
    preds: npt.NDArray[np.int_],
    num_classes: int,
    num_thresholds: int,
) -> npt.NDArray[np.int_]:
    """Compute multi-threshold confusion matrix for one-vs-rest classification."""
    pred_sum = np.count_nonzero(preds, axis=0)
    target_sum = np.count_nonzero(target, axis=0)

    tp = np.count_nonzero(np.multiply(preds, target), axis=0)
    fp = pred_sum - tp
    fn = target_sum - tp
    tn = target.shape[0] - tp - fp - fn

    return np.array([tn, fp, fn, tp]).T.reshape(num_thresholds, num_classes, 2, 2)


def _precision_recall_curve_compute_from_confmat(
    confmat: npt.NDArray[np.int_],
    thresholds: npt.NDArray[np.float_],
) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """Compute precision-recall curve from a multi-threshold confusion matrix."""
    tps = confmat[..., 1, 1]
    fps = confmat[..., 0, 1]
    fns = confmat[..., 1, 0]

    precision = np.divide(
        tps,
        tps + fps,
        out=np.zeros_like(tps, dtype=np.float64),
        where=(tps + fps) != 0,
    )
    recall = np.divide(
        tps,
        tps + fns,
        out=np.zeros_like(tps, dtype=np.float64),
        where=(tps + fns) != 0,
    )

    sort_idx = np.argsort(thresholds)
    thresholds = thresholds[sort_idx]
    precision = precision[sort_idx]
    recall = recall[sort_idx]  # in descending order

    return precision, recall, thresholds


def _binary_precision_recall_curve_format(
    target: npt.ArrayLike,
    preds: npt.ArrayLike,
    pos_label: int,
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Check and format binary precision-recall curve input/data.

    Parameters
    ----------
    target : npt.ArrayLike
        Ground truth (correct) target values.
    preds : npt.ArrayLike
        Estimated probabilities or non-thresholded output of decision function.
        A sigmoid function is applied if ``preds`` are not in [0, 1].
    pos_label : int
        Label of the positive class.

    Returns
    -------
    target : numpy.ndarray
        Ground truth (correct) target values as a numpy array.
    preds : numpy.ndarray
        Estimated probabilities or non-thresholded output of decision function
        as a numpy array.

    Raises
    ------
    ValueError
        If ``target`` is not binary, with only 1 and 0 as values; If ``target`` and
        ``preds`` are not of the same shape; If ``preds`` is not continuous.

    """
    target, preds, type_target, type_preds = common_input_checks_and_format(
        target,
        preds,
    )

    if pos_label not in [0, 1]:
        raise ValueError(f"Positive label must be 0 or 1, got {pos_label}.")

    if type_preds == "continuous-multioutput":
        assert preds.shape[-1] == 2, (
            "The argument `preds` must either be a 1D array or a 2D array with "
            f"exactly 2 columns, got an array with shape: {preds.shape}."
        )
        preds = preds[
            ...,
            pos_label,
        ]  # keep only the probabilities for the positive class
        type_preds = "continuous"

    if preds.shape != target.shape:
        raise ValueError(
            "The arguments `preds` and `target` should have the same shape. "
            f"Got {preds.shape} and {target.shape}.",
        )

    if type_target != "binary" or type_preds != "continuous":
        raise ValueError(
            "Expected argument `target` to be binary and `preds` to be an array of"
            f" floats with probability/logit scores, got {type_target} and"
            f" {type_preds} respectively.",
        )

    if not np.all(np.isin(target, [0, 1])):
        raise ValueError(
            "Expected argument `target` to be an array of 0s and 1s, but got "
            f"array with values {np.unique(target)}",
        )

    if not np.all(np.logical_and(preds >= 0.0, preds <= 1.0)):
        preds = sigmoid(preds)

    return target, preds


def _binary_precision_recall_curve_update(
    target: npt.NDArray[Any],
    preds: npt.NDArray[Any],
    thresholds: Optional[npt.NDArray[np.float_]] = None,
) -> Union[Tuple[npt.NDArray[np.int_], npt.NDArray[np.float_]], npt.NDArray[np.int_]]:
    """Compute the state from which the precision-recall curve can be computed.

    Parameters
    ----------
    target : numpy.ndarray
        Binary target values.
    preds : numpy.ndarray
        Predicted probabilities.
    thresholds : Optional[numpy.ndarray]
        Thresholds used for computing the precision and recall scores.

    Returns
    -------
    (target, preds): Tuple[numpy.ndarray, numpy.ndarray]
        Target and predicted probabilities, if ``thresholds`` is None.
    confmat : numpy.ndarray
        Multi-threshold confusion matrix, if ``thresholds`` is not None.

    """
    if thresholds is None:
        return target, preds

    # compute multi-threshold confusion matrix
    len_t = len(thresholds)
    preds_t = (
        np.expand_dims(preds, axis=-1) >= np.expand_dims(thresholds, axis=0)
    ).astype(np.int64)

    tp = np.sum((target == preds_t.T) & (target == 1), axis=1)
    fp = np.sum((target != preds_t.T) & (target == 0), axis=1)
    tn = np.sum((target == preds_t.T) & (target == 0), axis=1)
    fn = np.sum((target != preds_t.T) & (target == 1), axis=1)

    confmat: npt.NDArray[np.int_] = np.stack([tn, fp, fn, tp], axis=1).reshape(
        len_t,
        2,
        2,
    )

    return confmat


def _binary_precision_recall_curve_compute(
    state: Union[
        Tuple[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        npt.NDArray[np.int_],
    ],
    thresholds: Optional[npt.NDArray[np.float_]],
    pos_label: Optional[int] = None,
) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """Compute precision-recall curve from a state.

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
    precision : numpy.ndarray
        Precision scores such that element i is the precision of predictions
        with score >= thresholds[i].
    recall : numpy.ndarray
        Recall scores in descending order.
    thresholds : numpy.ndarray
        Thresholds used for computing the precision and recall scores.

    Raises
    ------
    ValueError
        If ``thresholds`` is None.

    """
    if isinstance(state, np.ndarray):
        precision, recall, thresholds = _precision_recall_curve_compute_from_confmat(
            state,
            thresholds,  # type: ignore[arg-type]
        )
    else:
        fps, tps, thresholds = _binary_clf_curve(
            state[0],
            state[1],
            pos_label=pos_label,
            sample_weight=None,
        )

        precision = np.divide(
            tps,
            tps + fps,
            out=np.zeros_like(tps, dtype=np.float64),
            where=(tps + fps) != 0,
        )
        recall = np.divide(
            tps,
            tps[-1],
            out=np.zeros_like(tps, dtype=np.float64),
            where=tps[-1] != 0,
        )

        # stop when full recall attained
        # and reverse the outputs so recall is decreasing
        last_ind = tps.searchsorted(tps[-1], side="right")
        sliced = slice(last_ind, None, -1)

        precision = np.hstack((precision[sliced], 1))
        recall = np.hstack((recall[sliced], 0))
        thresholds = thresholds[sliced]  # type: ignore[index]

    return precision, recall, thresholds


def binary_precision_recall_curve(
    target: npt.ArrayLike,
    preds: npt.ArrayLike,
    thresholds: Optional[Union[int, List[float], npt.NDArray[np.float_]]] = None,
    pos_label: int = 1,
) -> PRCurve:
    """Compute precision-recall curve for binary input.

    Parameters
    ----------
    target : npt.ArrayLike
        Binary target values.
    preds : npt.ArrayLike
        Predicted probabilities or output of a decision function. If ``preds``
        are logits, they will be transformed to probabilities via the sigmoid
        function.
    thresholds : int or list of floats or numpy.ndarray of floats, default=None
        Thresholds used for computing the precision and recall scores.
        If int, then the number of thresholds to use.
        If list or numpy.ndarray, then the thresholds to use.
        If None, then the thresholds are automatically determined by the
        unique values in ``preds``.
    pos_label : int
        The label of the positive class.

    Returns
    -------
    PRCurve
       A named tuple containing the precision (element i is the precision of predictions
       with score >= thresholds[i]), recall (scores in descending order)
       and thresholds used to compute the precision-recall curve.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.functional import binary_precision_recall_curve
    >>> target = [0, 0, 1, 1]
    >>> preds = [0.1, 0.4, 0.35, 0.8]
    >>> precision, recall, thresholds = binary_precision_recall_curve(
    ...     target, preds, thresholds=5
    ... )
    >>> precision
    array([0.5       , 0.66666667, 1.        , 1.        , 0.        ])
    >>> recall
    array([1. , 1. , 0.5, 0.5, 0. ])
    >>> thresholds
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])

    """
    _check_thresholds(thresholds)

    target, preds = _binary_precision_recall_curve_format(
        target,
        preds,
        pos_label=pos_label,
    )
    thresholds = _format_thresholds(thresholds)

    state = _binary_precision_recall_curve_update(target, preds, thresholds)
    precision_, recall_, thresholds_ = _binary_precision_recall_curve_compute(
        state,
        thresholds,
        pos_label=pos_label,
    )

    return PRCurve(precision_, recall_, thresholds_)


def _multiclass_precision_recall_curve_format(
    target: npt.ArrayLike,
    preds: npt.ArrayLike,
    num_classes: int,
) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.float_]]:
    """Check and format the input for the multiclass precision-recall curve.

    Parameters
    ----------
    target : npt.ArrayLike
        The target values.
    preds : npt.ArrayLike
        The predicted probabilities or output of a decision function. If
        ``preds`` is not in the [0, 1] range, it will be transformed into this
        range via the softmax function.
    num_classes : int
        Number of classes.

    Returns
    -------
    target : numpy.ndarray
        The target values as a numpy array.
    preds : numpy.ndarray
        The predicted probabilities as a numpy array.
    thresholds : numpy.ndarray
        Thresholds used for computing the precision and recall scores as a numpy array.

    Raises
    ------
    ValueError
        If ``target`` is not a 1D array of integers or contains values outside the
        range [0, num_classes) or does not have one more dimension than ``preds``;
        if ``preds`` is not a 2D array of floats or does not have the same number
        of classes as ``num_classes``; if ``preds and ``target`` do not have the
        same number of samples.

    """
    formatted = common_input_checks_and_format(target, preds)
    target_: npt.NDArray[np.int_] = formatted[0]
    preds_: npt.NDArray[np.float_] = formatted[1]
    type_target: str = formatted[2]
    type_preds: str = formatted[3]

    if preds_.ndim != target_.ndim + 1:
        raise ValueError(
            "Expected argument `preds` to have one more dimension than argument "
            f"`target`, but got {preds_.ndim} and {target_.ndim} respectively",
        )

    if type_target not in ["binary", "multiclass"]:
        raise ValueError(
            "Expected argument `target` to be an array of integers with "
            f"shape (N,) but got {type_target}",
        )

    if type_target == "binary" and not num_classes > 2:
        raise ValueError(
            "Expected `target` to be a multiclass target, but got a binary target",
        )

    if type_preds != "continuous-multioutput":
        raise ValueError(
            "Expected argument `preds` to be `preds` to be an array of floats"
            f" with probability/logit scores but got {type_preds}",
        )

    if preds_.shape[-1] != num_classes:
        raise ValueError(
            "Expected argument `preds` to have the same number of classes as "
            f"argument `num_classes`, but got {preds_.shape[-1]} and {num_classes} "
            "respectively",
        )

    if preds_.shape[0] != target_.shape[0]:
        raise ValueError(
            "Expected argument `preds` to have the same number of samples as "
            f"argument `target`, but got {preds_.shape[0]} and {target_.shape[0]} "
            "respectively",
        )

    num_implied_classes = len(np.unique(target_))
    if num_implied_classes > num_classes:
        raise ValueError(
            "Detected more unique values in `target` than `num_classes`. Expected only "
            f"{num_classes} but found {num_implied_classes} in `target`.",
        )

    if not np.all(np.logical_and(preds_ >= 0.0, preds_ <= 1.0)):
        preds_ = sp.special.softmax(preds_, axis=1)  # logit to probability

    if not np.allclose(1, preds_.sum(axis=1)):
        raise ValueError(
            "``preds`` need to be probabilities for multiclass problems"
            " i.e. they should sum up to 1.0 over classes",
        )

    return target_, preds_


def _multiclass_precision_recall_curve_update(
    target: npt.NDArray[np.int_],
    preds: npt.NDArray[np.float_],
    num_classes: int,
    thresholds: Optional[npt.NDArray[np.float_]] = None,
) -> Union[Tuple[npt.NDArray[np.int_], npt.NDArray[np.float_]], npt.NDArray[np.int_]]:
    """Update the state of the multiclass precision-recall curve.

    Parameters
    ----------
    target : numpy.ndarray
        Binary target values.
    preds : numpy.ndarray
        Predicted probabilities.
    num_classes : int
        Number of classes.
    thresholds : numpy.ndarray, default=None
        Thresholds used for computing the precision and recall scores.

    Returns
    -------
    (target, preds) : Tuple[numpy.ndarray, numpy.ndarray]
        The target and predicted probabilities, if ``thresholds`` is None.
    state : numpy.ndarray
        The state of the multiclass precision-recall curve,  if ``thresholds``
        is not None.

    """
    if thresholds is None:
        return target, preds

    # one-vs-all multi-threshold confusion matrix
    len_t = len(thresholds)
    preds_t = (
        np.expand_dims(preds, axis=-1)
        >= np.expand_dims(np.expand_dims(thresholds, axis=0), axis=0)
    ).astype(np.int64)

    target_t = np.expand_dims(
        label_binarize(target, classes=np.arange(num_classes)),
        axis=-1,
    )

    return _ovr_multi_threshold_confusion_matrix(
        target_t,
        preds_t,
        num_classes=num_classes,
        num_thresholds=len_t,
    )


def _multiclass_precision_recall_curve_compute(
    state: Union[
        Tuple[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        npt.NDArray[np.int_],
    ],
    thresholds: npt.NDArray[np.float_],
    num_classes: int,
) -> Union[
    Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]],
    Tuple[
        List[npt.NDArray[np.float_]],
        List[npt.NDArray[np.float_]],
        List[npt.NDArray[np.float_]],
    ],
]:
    """Compute the multiclass precision-recall curve.

    Parameters
    ----------
    state : numpy.ndarray
        The state of the multiclass precision-recall curve. If ``thresholds`` is
        None, then ``state`` is a tuple of the target and predicted probabilities.
        Otherwise, ``state`` is the one-vs-all multi-threshold confusion matrix.
    thresholds : numpy.ndarray
        Thresholds used for computing the precision and recall scores.
    num_classes : int
        Number of classes.

    Returns
    -------
    precision : numpy.ndarray or list of numpy.ndarray
        Precision scores where element i is the precision score corresponding to the
        threshold i. If state is a tuple of the target and predicted probabilities,
        then precision is a list of arrays, where each array corresponds to the
        precision scores for a class.
    recall : numpy.ndarray or list of numpy.ndarray
        Recall scores where element `i` is the recall score corresponding to the
        threshold  `i`. If state is a tuple of the target and predicted probabilities,
        then recall is a list of arrays, where each array corresponds to the recall
        scores for a class.
    thresholds : numpy.ndarray or list of numpy.ndarray
        Thresholds used for computing the precision and recall scores.

    """
    if isinstance(state, np.ndarray):
        precision, recall, thresholds = _precision_recall_curve_compute_from_confmat(
            state,
            thresholds,
        )

        precision = np.hstack((precision.T, np.ones((num_classes, 1))))
        recall = np.hstack((recall.T, np.zeros((num_classes, 1))))

        return precision, recall, thresholds

    precision_list, recall_list, thresholds_list = [], [], []
    for i in range(num_classes):
        (
            precision_i,
            recall_i,
            thresholds_i,
        ) = _binary_precision_recall_curve_compute(
            (state[0], state[1][:, i]),
            thresholds=None,
            pos_label=i,
        )

        precision_list.append(precision_i)
        recall_list.append(recall_i)
        thresholds_list.append(thresholds_i)

    return precision_list, recall_list, thresholds_list


def multiclass_precision_recall_curve(
    target: npt.ArrayLike,
    preds: npt.ArrayLike,
    num_classes: int,
    thresholds: Optional[Union[int, List[float], npt.NDArray[np.float_]]] = None,
) -> PRCurve:
    """Compute the precision-recall curve for multiclass problems.

    Parameters
    ----------
    target : ArrayLike
        Ground truth (correct) target values.
    preds : ArrayLike
        Estimated probabilities or decision function. If ``preds`` is a logit, it
        will be converted to a probability using the softmax function.
    num_classes : int
        The number of classes in the dataset.
    thresholds : Union[int, List[float], numpy.ndarray], default=None
        Thresholds used for computing the precision and recall scores.
        If int, then the number of thresholds to use.
        If list or array, then the thresholds to use.
        If None, then the thresholds are automatically determined by the
        unique values in ``preds``.

    Returns
    -------
    PRcurve
        A named tuple containing the precision, recall, and thresholds.
        Precision and recall are arrays where element i is the precision and
        recall score corresponding to threshold i. If state is a tuple of the
        target and predicted probabilities, then precision and recall are lists
        of arrays, where each array corresponds to the precision and recall
        scores for a class.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.functional import (
    ...     multiclass_precision_recall_curve,
    ... )
    >>> target = [0, 1, 2, 2]
    >>> preds = [[0.1, 0.6, 0.3], [0.05, 0.95, 0], [0.5, 0.3, 0.2], [0.3, 0.4, 0.3]]
    >>> precision, recall, thresholds = multiclass_precision_recall_curve(
    ...     target, preds, num_classes=3, thresholds=5
    ... )
    >>> precision
    array([[0.25, 0.  , 0.  , 0.  , 0.  , 1.  ],
           [0.25, 0.25, 0.5 , 1.  , 0.  , 1.  ],
           [0.5 , 0.5 , 0.  , 0.  , 0.  , 1.  ]])
    >>> recall
    array([[1. , 0. , 0. , 0. , 0. , 0. ],
           [1. , 1. , 1. , 1. , 0. , 0. ],
           [1. , 0.5, 0. , 0. , 0. , 0. ]])
    >>> thresholds
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])

    """
    _check_thresholds(thresholds)

    target, preds = _multiclass_precision_recall_curve_format(
        target,
        preds,
        num_classes=num_classes,
    )

    thresholds = _format_thresholds(thresholds)

    state = _multiclass_precision_recall_curve_update(
        target,
        preds,
        num_classes=num_classes,
        thresholds=thresholds,
    )

    precision_, recall_, thresholds_ = _multiclass_precision_recall_curve_compute(
        state,
        thresholds,  # type: ignore
        num_classes,
    )
    return PRCurve(precision_, recall_, thresholds_)


def _multilabel_precision_recall_curve_format(
    target: npt.ArrayLike,
    preds: npt.ArrayLike,
    num_labels: int,
) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.float_]]:
    """Check and format the multilabel precision-recall curve input/data.

    Parameters
    ----------
    target : npt.ArrayLike
        The target values.
    preds : npt.ArrayLike
        Predicted probabilities or output of a decision function. If the
        values are not in [0, 1], then they are converted into probabilities
        by applying the sigmoid function.
    num_labels : int
        The number of labels in the dataset.
    thresholds : int, list of floats or numpy.ndarray of floats, default=None
        Thresholds used for computing the precision and recall scores.
        If int, then the number of thresholds to use.
        If list or array, then the thresholds to use.
        If None, then the thresholds are automatically determined by the
        unique values in ``preds``.

    Returns
    -------
    target : numpy.ndarray
        The target values as a numpy array.
    preds : numpy.ndarray
        The predicted probabilities as a numpy array.

    Raises
    ------
    ValueError
        If ``target`` is not in multilabel-indicator format.
    ValueError
        If ``preds`` does not contain float values.
    ValueError
        If ``num_labels`` does not match up with the number of columns in ``preds``.
    ValueError
        If the number of columns in ``preds`` is not the same as the number of
        columns in ``target``.

    """
    target, preds, type_target, type_preds = common_input_checks_and_format(
        target,
        preds,
    )

    # allow single-sample inputs
    if type_preds in ["continuous", "binary"] and type_target == "binary":
        preds = np.expand_dims(preds, axis=0)
        type_preds = (
            "continuous-multioutput"
            if type_preds == "continuous"
            else "multilabel-indicator"
        )
    if type_target == "binary":
        target = np.expand_dims(target, axis=0)
        type_target = "multilabel-indicator"

    # validate input types
    if type_target != "multilabel-indicator":
        raise ValueError(
            "Expected argument `target` to be a multilabel indicator array, but got "
            f"{type_target}",
        )

    if type_preds != "continuous-multioutput":
        raise ValueError(
            "Expected argument `preds` to be an array of floats with"
            f" probabilities/logit scores, but got {type_preds}",
        )

    if num_labels != preds.shape[1]:
        raise ValueError(
            "Expected `num_labels` to be equal to the number of columns in `preds`, "
            f"but got {num_labels} and {preds.shape[1]}",
        )

    if target.shape[1] != preds.shape[1]:
        raise ValueError(
            "Number of columns in `target` and `preds` must be the same."
            f"Got {target.shape[1]} and {preds.shape[1]}.",
        )

    if not np.all(np.logical_and(preds >= 0.0, preds <= 1.0)):
        preds = sigmoid(preds)

    return target, preds


def _multilabel_precision_recall_curve_update(
    target: npt.NDArray[np.int_],
    preds: npt.NDArray[np.float_],
    num_labels: int,
    thresholds: Optional[npt.NDArray[np.float_]] = None,
) -> Union[Tuple[npt.NDArray[np.int_], npt.NDArray[np.float_]], npt.NDArray[np.int_]]:
    """Update the multilabel precision-recall curve state.

    Parameters
    ----------
    target : numpy.ndarray
        The target values.
    preds : numpy.ndarray
        Predicted probabilities or output of a decision function.
    num_labels : int
        The number of labels in the dataset.
    thresholds : numpy.ndarray
        Thresholds used for computing the precision and recall scores.

    Returns
    -------
    (target, preds) : Tuple[numpy.ndarray, numpy.ndarray]
        The target and predicted values, if ``thresholds`` is None.
    state : numpy.ndarray
        One-vs-rest multi-threshold confusion matrix, if ``thresholds`` is not None.

    """
    if thresholds is None:
        return target, preds

    # one-vs-all multi-threshold confusion matrix
    len_t = len(thresholds)
    preds_t = (
        np.expand_dims(preds, axis=-1) >= np.expand_dims(thresholds, axis=0)
    ).astype(np.int_)

    target_t = np.expand_dims(target, axis=-1)

    return _ovr_multi_threshold_confusion_matrix(target_t, preds_t, num_labels, len_t)


def _multilabel_precision_recall_curve_compute(
    state: Union[
        Tuple[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        npt.NDArray[np.int_],
    ],
    thresholds: npt.NDArray[np.float_],
    num_labels: int,
) -> Union[
    Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]],
    Tuple[
        List[npt.NDArray[np.float_]],
        List[npt.NDArray[np.float_]],
        List[npt.NDArray[np.float_]],
    ],
]:
    """Compute the precision-recall curve for multilabel data.

    Parameters
    ----------
    state : Tuple[numpy.ndarray, numpy.ndarray] or numpy.ndarray
        The target and predicted values, if ``thresholds`` is None. Otherwise,
        the one-vs-rest multi-threshold confusion matrix.
    thresholds : numpy.ndarray
        Thresholds used for computing the precision and recall scores.
    num_labels : int
        Number of labels.

    Returns
    -------
    precision : numpy.ndarray or List[numpy.ndarray]
        Precision values for each label.
    recall : numpy.ndarray or List[numpy.ndarray]
        Recall values for each label.
    thresholds : numpy.ndarray or List[numpy.ndarray]
        If ``thresholds`` is None, then thresholds is a list of arrays, one for
        each label. Otherwise, thresholds is a single array with shape
        (len(``thresholds``,).

    """
    if isinstance(state, np.ndarray):
        precision, recall, thresholds = _precision_recall_curve_compute_from_confmat(
            state,
            thresholds,
        )

        precision = np.hstack((precision.T, np.ones((num_labels, 1))))
        recall = np.hstack((recall.T, np.zeros((num_labels, 1))))

        return precision, recall, thresholds

    precision_list, recall_list, thresholds_list = [], [], []
    for i in range(num_labels):
        target = state[0][:, i]
        preds = state[1][:, i]
        (
            precision_i,
            recall_i,
            thresholds_i,
        ) = _binary_precision_recall_curve_compute(
            (target, preds),
            thresholds=None,
            pos_label=1,
        )

        precision_list.append(precision_i)
        recall_list.append(recall_i)
        thresholds_list.append(thresholds_i)

    return precision_list, recall_list, thresholds_list


def multilabel_precision_recall_curve(
    target: npt.ArrayLike,
    preds: npt.ArrayLike,
    num_labels: int,
    thresholds: Optional[Union[int, List[float], npt.NDArray[np.float_]]] = None,
) -> PRCurve:
    """Compute the precision-recall curve for multilabel input.

    Parameters
    ----------
    target : npt.ArrayLike
        The target values.
    preds : npt.ArrayLike
        Predicted probabilities or output of a decision function. If the
        values are not in [0, 1], then they are converted into that range
        by applying the sigmoid function.
    num_labels : int
        The number of labels in the dataset.
    thresholds : numpy.ndarray
        Thresholds used for computing the precision and recall scores.
        If int, then the number of thresholds to use.
        If list of floats, then the thresholds to use.
        If None, then the thresholds are computed automatically from the unique
        values in ``preds``.

    Returns
    -------
    PRCurve
        A named tuple with the following:
        - ``precision``: numpy.ndarray or List[numpy.ndarray].
        Precision values for each label. If ``thresholds`` is None, then
        precision is a list of arrays, one for each label. Otherwise,
        precision is a single array with shape
        (``num_labels``, len(``thresholds``)).
        - ``recall``: numpy.ndarray or List[numpy.ndarray].
        Recall values for each label. If ``thresholds`` is None, then
        recall is a list of arrays, one for each label. Otherwise,
        recall is a single array with shape (``num_labels``, len(``thresholds``)).
        - ``thresholds``: numpy.ndarray or List[numpy.ndarray].
        If ``thresholds`` is None, then thresholds is a list of arrays, one for
        each label. Otherwise, thresholds is a single array with shape
        (len(``thresholds``,).

    Examples
    --------
    >>> from cyclops.evaluate.metrics.functional import (
    ...     multilabel_precision_recall_curve,
    ... )
    >>> target = [[1, 1, 0], [0, 1, 0]]
    >>> preds = [[0.1, 0.9, 0.8], [0.05, 0.95, 0.35]]
    >>> precision, recall, thresholds = multilabel_precision_recall_curve(
    ...     target, preds, num_labels=3, thresholds=5
    ... )
    >>> precision
    array([[0.5, 0. , 0. , 0. , 0. , 1. ],
           [1. , 1. , 1. , 1. , 0. , 1. ],
           [0. , 0. , 0. , 0. , 0. , 1. ]])
    >>> recall
    array([[1., 0., 0., 0., 0., 0.],
           [1., 1., 1., 1., 0., 0.],
           [0., 0., 0., 0., 0., 0.]])
    >>> thresholds
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])

    """
    _check_thresholds(thresholds)

    target, preds = _multilabel_precision_recall_curve_format(
        target,
        preds,
        num_labels=num_labels,
    )

    thresholds = _format_thresholds(thresholds)

    state = _multilabel_precision_recall_curve_update(
        target,
        preds,
        num_labels=num_labels,
        thresholds=thresholds,
    )

    precision_, recall_, thresholds_ = _multilabel_precision_recall_curve_compute(
        state,
        thresholds,  # type: ignore
        num_labels,
    )
    return PRCurve(precision_, recall_, thresholds_)


def precision_recall_curve(
    target: npt.ArrayLike,
    preds: npt.ArrayLike,
    task: Literal["binary", "multiclass", "multilabel"],
    thresholds: Optional[Union[int, List[float], npt.NDArray[np.float_]]] = None,
    pos_label: int = 1,
    num_classes: Optional[int] = None,
    num_labels: Optional[int] = None,
) -> PRCurve:
    """Compute the precision-recall curve for different tasks/input types.

    Parameters
    ----------
    target : npt.ArrayLike
        Ground truth (correct) target values.
    preds : npt.ArrayLike
        Estimated probabilities or non-thresholded output of decision function.
    task : Literal["binary", "multiclass", "multilabel"]
        The task for which the precision-recall curve is computed.
    thresholds : int or list of floats or numpy.ndarray of floats, default=None
        Thresholds used for computing the precision and recall scores. If int,
        then the number of thresholds to use. If list or array, then the
        thresholds to use. If None, then the thresholds are automatically
        determined by the sunique values in ``preds``
    pos_label : int, default=1
        The label of the positive class.
    num_classes : int, optional
        The number of classes in the dataset. Required if ``task`` is ``"multiclass"``.
    num_labels : int, optional
        The number of labels in the dataset. Required if ``task`` is ``"multilabel"``.

    Returns
    -------
    PRCurve
        A named tuple with the following:
        - ``precision``: numpy.ndarray or List[numpy.ndarray].
        The precision scores where ``precision[i]`` is the precision score for
        ``scores >= thresholds[i]``. If ``task`` is 'multiclass' or 'multilabel',
        then ``precision`` is a list of numpy arrays, where ``precision[i]`` is the
        precision scores for class or label ``i``.
        - ``recall``: numpy.ndarray or List[numpy.ndarray].
        The recall scores where ``recall[i]`` is the recall score for ``scores >=
        thresholds[i]``. If ``task`` is 'multiclass' or 'multilaabel', then
        ``recall`` is a list of numpy arrays, where ``recall[i]`` is the recall
        scores for class or label ``i``.
        - ``thresholds``: numpy.ndarray or List[numpy.ndarray].
        Thresholds used for computing the precision and recall scores.

    Raises
    ------
    ValueError
        If ``task`` is not one of 'binary', 'multiclass' or 'multilabel'.
    AssertionError
        If ``task`` is ``multiclass`` and ``num_classes`` is not provided.
    AssertionError
        If ``task`` is ``multilabel`` and ``num_labels`` is not provided.

    Examples
    --------
    >>> # (binary)
    >>> from cyclops.evaluate.metrics.functional import precision_recall_curve
    >>> target = [0, 0, 1, 1]
    >>> preds = [0.1, 0.4, 0.35, 0.8]
    >>> precision, recall, thresholds = precision_recall_curve(target, preds, "binary")
    >>> precision
    array([0.5       , 0.66666667, 0.5       , 1.        , 1.        ])
    >>> recall
    array([1. , 1. , 0.5, 0.5, 0. ])
    >>> thresholds
    array([0.1 , 0.35, 0.4 , 0.8 ])

    >>> # (multiclass)
    >>> from cyclops.evaluate.metrics.functional import precision_recall_curve
    >>> target = [0, 1, 2, 2]
    >>> preds = [[0.1, 0.6, 0.3], [0.05, 0.95, 0], [0.5, 0.3, 0.2], [0.3, 0.4, 0.3]]
    >>> precision, recall, thresholds = precision_recall_curve(
    ...     target, preds, task="multiclass", num_classes=3
    ... )
    >>> [prec.tolist() for prec in precision]
    [[0.25, 0.3333333333333333, 0.0, 0.0, 1.0], [0.25, 0.3333333333333333, 0.5, 1.0, 1.0], [0.5, 0.6666666666666666, 0.5, 1.0]]
    >>> [rec.tolist() for rec in recall]
    [[1.0, 1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.5, 0.0]]
    >>> thresholds
    [array([0.05, 0.1 , 0.3 , 0.5 ]), array([0.3 , 0.4 , 0.6 , 0.95]), array([0. , 0.2, 0.3])]

    >>> # (multilabel)
    >>> from cyclops.evaluate.metrics.functional import precision_recall_curve
    >>> target = [[1, 1, 0], [0, 1, 0]]
    >>> preds = [[0.1, 0.9, 0.8], [0.05, 0.95, 0.35]]
    >>> precision, recall, thresholds = precision_recall_curve(
    ...     target, preds, "multilabel", num_labels=3
    ... )
    >>> precision
    [array([0.5, 1. , 1. ]), array([1., 1., 1.]), array([0., 0., 1.])]
    >>> recall
    [array([1., 1., 0.]), array([1. , 0.5, 0. ]), array([0., 0., 0.])]
    >>> thresholds
    [array([0.05, 0.1 ]), array([0.9 , 0.95]), array([0.35, 0.8 ])]

    """  # noqa: W505
    if task == "binary":
        return binary_precision_recall_curve(
            target,
            preds,
            thresholds=thresholds,
            pos_label=pos_label,
        )
    if task == "multiclass":
        assert isinstance(num_classes, int) and num_classes > 0, (
            "Number of classes must be a positive integer."
        )

        return multiclass_precision_recall_curve(
            target,
            preds,
            num_classes=num_classes,
            thresholds=thresholds,
        )
    if task == "multilabel":
        assert isinstance(num_labels, int) and num_labels > 0, (
            "Number of labels must be a positive integer."
        )

        return multilabel_precision_recall_curve(
            target,
            preds,
            num_labels=num_labels,
            thresholds=thresholds,
        )

    raise ValueError(
        "Expected argument `task` to be either 'binary', 'multiclass' or "
        f"'multilabel', but got {task}",
    )
