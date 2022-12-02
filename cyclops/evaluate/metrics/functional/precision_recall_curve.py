"""Functions for computing the precision-recall curve for different input types."""

from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import scipy as sp
from numpy.typing import ArrayLike
from sklearn.metrics._ranking import _binary_clf_curve
from sklearn.preprocessing import label_binarize

from cyclops.evaluate.metrics.utils import (
    _check_thresholds,
    common_input_checks_and_format,
    sigmoid,
)


def _format_thresholds(
    thresholds: Union[int, List[float], np.ndarray] = None
) -> Optional[np.ndarray]:
    """Format thresholds to be a 1D numpy array of floats."""
    if isinstance(thresholds, int):
        thresholds = np.linspace(0, 1, thresholds)
    elif isinstance(thresholds, list):
        thresholds = np.array(thresholds)

    return thresholds


def _ovr_multi_threshold_confusion_matrix(
    target: np.ndarray, preds: np.ndarray, num_classes: int, num_thresholds: int
) -> np.ndarray:
    """Compute multi-threshold confusion matrix for one-vs-rest classification."""
    pred_sum = np.count_nonzero(preds, axis=0)
    target_sum = np.count_nonzero(target, axis=0)

    # pylint: disable=invalid-name
    tp = np.count_nonzero(np.multiply(preds, target), axis=0)
    fp = pred_sum - tp
    fn = target_sum - tp
    tn = target.shape[0] - tp - fp - fn

    # pylint: disable=too-many-function-args
    confmat = np.array([tn, fp, fn, tp]).T.reshape(num_thresholds, num_classes, 2, 2)

    return confmat


def _precision_recall_curve_compute_from_confmat(
    confmat: np.ndarray, thresholds: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    target: ArrayLike,
    preds: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """Check and format binary precision-recall curve input/data.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Estimated probabilities or non-thresholded output of decision function.
            A sigmoid function is applied if ``preds`` are not in [0, 1].

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
            If ``target`` and ``preds`` are not of the same shape.
        ValueError
            If ``target`` is not binary or ``preds`` is not continuous.
        ValueError
            If ``target`` does not contain only 0s and 1s.

    """
    target, preds, type_target, type_preds = common_input_checks_and_format(
        target, preds
    )

    if preds.shape != target.shape:
        raise ValueError(
            "The arguments `preds` and `target` should have the same shape. "
            f"Got {preds.shape} and {target.shape}."
        )

    if type_target != "binary" or type_preds != "continuous":
        raise ValueError(
            "Expected argument `target` to be binary and `preds` to be an array of"
            f" floats with probability/logit scores, got {type_target} and"
            f" {type_preds} respectively."
        )

    if not np.all(np.isin(target, [0, 1])):
        raise ValueError(
            "Expected argument `target` to be an array of 0s and 1s, but got "
            f"array with values {np.unique(target)}"
        )

    if not np.all(np.logical_and(preds >= 0.0, preds <= 1.0)):
        preds = sigmoid(preds)

    return target, preds


def _binary_precision_recall_curve_update(
    target: np.ndarray, preds: np.ndarray, thresholds: Optional[np.ndarray]
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
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

    # pylint: disable=invalid-name
    tp = np.sum((target == preds_t.T) & (target == 1), axis=1)
    fp = np.sum((target != preds_t.T) & (target == 0), axis=1)
    tn = np.sum((target == preds_t.T) & (target == 0), axis=1)
    fn = np.sum((target != preds_t.T) & (target == 1), axis=1)

    confmat = np.stack([tn, fp, fn, tp], axis=1).reshape(len_t, 2, 2)

    return confmat


def _binary_precision_recall_curve_compute(
    state: Union[Tuple, np.ndarray],
    thresholds: np.ndarray,
    pos_label: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    """
    if isinstance(state, np.ndarray):
        precision, recall, thresholds = _precision_recall_curve_compute_from_confmat(
            state, thresholds
        )
    else:
        fps, tps, thresholds = _binary_clf_curve(
            state[0], state[1], pos_label=pos_label, sample_weight=None
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

        # pylint: disable=invalid-name
        # stop when full recall attained
        # and reverse the outputs so recall is decreasing
        last_ind = tps.searchsorted(tps[-1])
        sl = slice(last_ind, None, -1)

        precision = np.hstack((precision[sl], 1))
        recall = np.hstack((recall[sl], 0))
        thresholds = thresholds[sl]

    return precision, recall, thresholds


def binary_precision_recall_curve(
    target: ArrayLike,
    preds: ArrayLike,
    thresholds: Union[int, List[float], np.ndarray] = None,
    pos_label: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision-recall curve for binary input.

    Parameters
    ----------
        target : ArrayLike
            Binary target values.
        preds : ArrayLike
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
        precision : numpy.ndarray
            Precision scores such that element i is the precision of predictions
            with score >= thresholds[i].
        recall : numpy.ndarray
            Recall scores in descending order.
        thresholds : numpy.ndarray
            Thresholds used for computing the precision and recall scores.

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import (
        ...     binary_precision_recall_curve
        ... )
        >>> target = [0, 0, 1, 1]
        >>> preds = [0.1, 0.4, 0.35, 0.8]
        >>> precision, recall, thresholds = binary_precision_recall_curve(target,
        ...     preds, thresholds=5
        ... )
        >>> precision
        array([0.5, 0.66666667, 1., 1., 0.]
        >>> recall
        array([1., 1., 0.5, 0.5, 0.])
        >>> thresholds
        array([0.1, 0.25 , 0.5, 0.75 , 1.])

    """
    _check_thresholds(thresholds)

    target, preds = _binary_precision_recall_curve_format(target, preds)
    thresholds = _format_thresholds(thresholds)

    state = _binary_precision_recall_curve_update(target, preds, thresholds)

    return _binary_precision_recall_curve_compute(
        state, thresholds, pos_label=pos_label
    )


def _multiclass_precision_recall_curve_format(
    target: ArrayLike,
    preds: ArrayLike,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Check and format the input for the multiclass precision-recall curve.

    Parameters
    ----------
        target : ArrayLike
            The target values.
        preds : ArrayLike
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
            Thresholds used for computing the precision and recall scores as a
            numpy array.

    Raises
    ------
        ValueError
            If ``target`` does not have one more dimension than ``preds``.
        ValueError
            If ``preds`` is not a 2D array of floats.
        ValueError
            If ``target`` is not a 1D array of integers.
        ValueError
            If ``preds`` does not have the same number of classes as ``num_classes``.
        ValueError
            If ``target`` and ``preds`` have different number of samples.
        ValueError
            If ``target`` contains values outside of the range [0, num_classes).

    """
    target, preds, type_target, type_preds = common_input_checks_and_format(
        target, preds
    )

    if preds.ndim != target.ndim + 1:
        raise ValueError(
            "Expected argument `preds` to have one more dimension than argument "
            f"`target`, but got {preds.ndim} and {target.ndim} respectively"
        )

    if type_target not in ["binary", "multiclass"]:
        raise ValueError(
            "Expected argument `target` to be an array of integers with "
            f"shape (N,) but got {type_target}"
        )

    if type_target == "binary" and not num_classes > 2:
        raise ValueError(
            "Expected `target` to be a multiclass target, but got a binary target"
        )

    if type_preds != "continuous-multioutput":
        raise ValueError(
            "Expected argument `preds` to be `preds` to be an array of floats"
            f" with probability/logit scores but got {type_preds}"
        )

    if preds.shape[-1] != num_classes:
        raise ValueError(
            "Expected argument `preds` to have the same number of classes as "
            f"argument `num_classes`, but got {preds.shape[-1]} and {num_classes} "
            "respectively"
        )

    if preds.shape[0] != target.shape[0]:
        raise ValueError(
            "Expected argument `preds` to have the same number of samples as "
            f"argument `target`, but got {preds.shape[0]} and {target.shape[0]} "
            "respectively"
        )

    num_implied_classes = len(np.unique(target))
    if num_implied_classes > num_classes:
        raise ValueError(
            "Detected more unique values in `target` than `num_classes`. Expected only "
            f"{num_classes} but found {num_implied_classes} in `target`."
        )

    if not np.all(np.logical_and(preds >= 0.0, preds <= 1.0)):
        preds = sp.special.softmax(preds, axis=1)  # logit to probability

    if not np.allclose(1, preds.sum(axis=1)):
        raise ValueError(
            "``preds`` need to be probabilities for multiclass problems"
            " i.e. they should sum up to 1.0 over classes"
        )

    return target, preds


def _multiclass_precision_recall_curve_update(
    target: np.ndarray,
    preds: np.ndarray,
    num_classes: int,
    thresholds: np.ndarray = None,
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
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
        label_binarize(target, classes=np.arange(num_classes)), axis=-1
    )

    state = _ovr_multi_threshold_confusion_matrix(
        target_t, preds_t, num_classes=num_classes, num_thresholds=len_t
    )

    return state


def _multiclass_precision_recall_curve_compute(
    state: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    thresholds: np.ndarray,
    num_classes: int,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]],
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
            Precision scores where element i is the precision score corresponding
            to the threshold i. If state is a tuple of the target and predicted
            probabilities, then precision is a list of arrays, where each array
            corresponds to the precision scores for a class.
        recall : numpy.ndarray or list of numpy.ndarray
            Recall scores where element i is the recall score corresponding to
            the threshold i. If state is a tuple of the target and predicted
            probabilities, then recall is a list of arrays, where each array
            corresponds to the recall scores for a class.
        thresholds : numpy.ndarray or list of numpy.ndarray
            Thresholds used for computing the precision and recall scores.

    """
    if isinstance(state, np.ndarray):
        precision, recall, thresholds = _precision_recall_curve_compute_from_confmat(
            state, thresholds
        )

        precision = np.hstack((precision.T, np.ones((num_classes, 1))))
        recall = np.hstack((recall.T, np.zeros((num_classes, 1))))
    else:
        precision, recall, thresholds = [], [], []
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

            precision.append(precision_i)
            recall.append(recall_i)
            thresholds.append(thresholds_i)

    return precision, recall, thresholds


def multiclass_precision_recall_curve(
    target: ArrayLike,
    preds: ArrayLike,
    num_classes: int,
    thresholds: Union[int, List[float], np.ndarray] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]],
]:
    """Compute the precision-recall curve for multiclass problems.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Estimated probabilities or decision function. If ``preds`` is a
            logit, it will be converted to a probability using the softmax
            function.
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
        precision : numpy.ndarray or list of numpy.ndarray
            Precision scores where element i is the precision score corresponding
            to the threshold i. If state is a tuple of the target and predicted
            probabilities, then precision is a list of arrays, where each array
            corresponds to the precision scores for a class.
        recall : numpy.ndarray or list of numpy.ndarray
            Recall scores where element i is the recall score corresponding to
            the threshold i. If state is a tuple of the target and predicted
            probabilities, then recall is a list of arrays, where each array
            corresponds to the recall scores for a class.
        thresholds : numpy.ndarray or list of numpy.ndarray
            Thresholds used for computing the precision and recall scores.

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import (
        ...     multiclass_precision_recall_curve
        ... )
        >>> target = [0, 1, 2, 2]
        >>> preds = [[0.1, 0.6, 0.3], [0.05, 0.95, 0], [0.5, 0.3, 0.2], [0.3, 0.4, 0.3]]
        >>> precision, recall, thresholds = multiclass_precision_recall_curve(target,
        ...     preds, num_classes=3, thresholds=5)
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
        target, preds, num_classes=num_classes
    )

    thresholds = _format_thresholds(thresholds)

    state = _multiclass_precision_recall_curve_update(
        target, preds, num_classes=num_classes, thresholds=thresholds
    )

    return _multiclass_precision_recall_curve_compute(state, thresholds, num_classes)


def _multilabel_precision_recall_curve_format(
    target: ArrayLike,
    preds: ArrayLike,
    num_labels: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Check and format the multilabel precision-recall curve input/data.

    Parameters
    ----------
        target : ArrayLike
            The target values.
        preds : ArrayLike
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
        thresholds : numpy.ndarray
            Thresholds used for computing the precision and recall scores as a
            numpy array, if ``thresholds`` is not None.

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
        target, preds
    )

    if type_target != "multilabel-indicator":
        raise ValueError(
            "Expected argument `target` to be a multilabel indicator array, but got "
            f"{type_target}"
        )

    if type_preds != "continuous-multioutput":
        raise ValueError(
            "Expected argument `preds` to be an array of floats with"
            f" probabilities/logit scores, but got {type_preds}"
        )

    if num_labels != preds.shape[1]:
        raise ValueError(
            "Expected `num_labels` to be equal to the number of columns in `preds`, "
            f"but got {num_labels} and {preds.shape[1]}"
        )

    if target.shape[1] != preds.shape[1]:
        raise ValueError(
            "Number of columns in `target` and `preds` must be the same."
            f"Got {target.shape[1]} and {preds.shape[1]}."
        )

    if not np.all(np.logical_and(preds >= 0.0, preds <= 1.0)):
        preds = sigmoid(preds)

    return target, preds


def _multilabel_precision_recall_curve_update(
    target: np.ndarray, preds: np.ndarray, num_labels: int, thresholds: np.ndarray
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
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

    state = _ovr_multi_threshold_confusion_matrix(target_t, preds_t, num_labels, len_t)

    return state


def _multilabel_precision_recall_curve_compute(
    state: Union[Tuple[np.ndarray, np.ndarray], np.ndarray],
    thresholds: np.ndarray,
    num_labels: int,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]],
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
            state, thresholds
        )

        precision = np.hstack((precision.T, np.ones((num_labels, 1))))
        recall = np.hstack((recall.T, np.zeros((num_labels, 1))))
    else:
        precision, recall, thresholds = [], [], []
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

            precision.append(precision_i)
            recall.append(recall_i)
            thresholds.append(thresholds_i)

    return precision, recall, thresholds


def multilabel_precision_recall_curve(
    target: ArrayLike,
    preds: ArrayLike,
    num_labels: int,
    thresholds: Union[int, List[float], np.ndarray] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]],
]:
    """Compute the precision-recall curve for multilabel input.

    Parameters
    ----------
        target : ArrayLike
            The target values.
        preds : ArrayLike
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
        precision : numpy.ndarray or List[numpy.ndarray]
            Precision values for each label. If ``thresholds`` is None, then
            precision is a list of arrays, one for each label. Otherwise,
            precision is a single array with shape
            (``num_labels``, len(``thresholds``)).
        recall : numpy.ndarray or List[numpy.ndarray]
            Recall values for each label. If ``thresholds`` is None, then
            recall is a list of arrays, one for each label. Otherwise,
            recall is a single array with shape (``num_labels``, len(``thresholds``)).
        thresholds : numpy.ndarray or List[numpy.ndarray]
            If ``thresholds`` is None, then thresholds is a list of arrays, one for
            each label. Otherwise, thresholds is a single array with shape
            (len(``thresholds``,).

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import (
        ...     multilabel_precision_recall_curve)
        >>> target = [[1, 1, 0], [0, 1, 0]]
        >>> preds = [[0.1, 0.9, 0.8], [0.05, 0.95, 0.35]]
        >>> precision, recall, thresholds = multilabel_precision_recall_curve(
        ...     target, preds, num_labels=3, thresholds=5)
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
        target, preds, num_labels=num_labels
    )

    thresholds = _format_thresholds(thresholds)

    state = _multilabel_precision_recall_curve_update(
        target, preds, num_labels=num_labels, thresholds=thresholds
    )

    return _multilabel_precision_recall_curve_compute(state, thresholds, num_labels)


def precision_recall_curve(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    task: Literal["binary", "multiclass", "multilabel"],
    thresholds: Union[int, List[float], np.ndarray] = None,
    pos_label: int = 1,
    num_classes: int = None,
    num_labels: int = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]],
]:
    """Compute the precision-recall curve for different tasks/input types.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
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
            The number of classes in the dataset. Required if ``task`` is
            ``"multiclass"``.
        num_labels : int, optional
            The number of labels in the dataset. Required if ``task`` is
            ``"multilabel"``.

    Returns
    -------
        precision : numpy.ndarray
            The precision scores where ``precision[i]`` is the precision score for
            ``scores >= thresholds[i]``. If ``task`` is 'multiclass' or 'multilaabel',
            then ``precision`` is a list of numpy arrays, where ``precision[i]`` is the
            precision scores for class or label ``i``.
        recall : numpy.ndarray
            The recall scores where ``recall[i]`` is the recall score for ``scores >=
            thresholds[i]``. If ``task`` is 'multiclass' or 'multilaabel', then
            ``recall`` is a list of numpy arrays, where ``recall[i]`` is the recall
            scores for class or label ``i``.
        thresholds : numpy.ndarray
            Thresholds used for computing the precision and recall scores.

    Raises
    ------
        ValueError
            If ``task`` is not one of 'binary', 'multiclass' or 'multilabel'.
        AssertionError
            If ``task`` is ``multiclass`` and ``num_classes`` is not provided.
        AssertionError
            If ``task`` is ``multilabel`` and ``num_labels`` is not provided.

    Example (binary)
    ----------------
        >>> from cyclops.evaluation.metrics.functional import precision_recall_curve
        >>> target = [0, 0, 1, 1]
        >>> preds = [0.1, 0.4, 0.35, 0.8]
        >>> precision, recall, thresholds = precision_recall_curve(target, preds,
        ...     "binary")
        >>> precision
        array([0.66666667, 0.5, 1., 1.])
        >>> recall
        array([1. , 0.5, 0.5, 0. ])
        >>> thresholds
        array([0.35, 0.4 , 0.8 ])

    Example (multiclass)
    --------------------
        >>> from cyclops.evaluation.metrics.functional import precision_recall_curve
        >>> target = [0, 1, 2, 2]
        >>> preds = [[0.1, 0.6, 0.3], [0.05, 0.95, 0], [0.5, 0.3, 0.2], [0.3, 0.4, 0.3]]
        >>> precision, recall, thresholds = precision_recall_curve(
        ...     target, preds, task="multiclass", num_classes=3)
        >>> precision
        [array([0.33333333, 0.        , 0.        , 1.        ]),
        array([1., 1.]),
        array([0.66666667, 0.5       , 1.        ])]
        >>> recall
        [array([1., 0., 0., 0.]), array([1., 0.]), array([1. , 0.5, 0. ])]
        >>> thresholds
        [array([0.1, 0.3, 0.5]), array([0.95]), array([0.2, 0.3])]

    Example (multilabel)
    --------------------
        >>> from cyclops.evaluation.metrics.functional import precision_recall_curve
        >>> target = [[1, 1, 0], [0, 1, 0]]
        >>> preds = [[0.1, 0.9, 0.8], [0.05, 0.95, 0.35]]
        >>> precision, recall, thresholds = precision_recall_curve(target, preds,
        ...     "multilabel", num_labels=3)
        >>> precision
        [array([1., 1.]), array([1., 1., 1.]), array([0., 1.])]
        >>> recall
        [array([1., 0.]), array([1. , 0.5, 0. ]), array([0., 0.])]
        >>> thresholds
        [array([0.1]), array([0.9 , 0.95]), array([0.8])]

    """
    if task == "binary":
        precision, recall, thresholds = binary_precision_recall_curve(
            target, preds, thresholds=thresholds, pos_label=pos_label
        )
    elif task == "multiclass":
        assert (
            isinstance(num_classes, int) and num_classes > 0
        ), "Number of classes must be a positive integer."

        precision, recall, thresholds = multiclass_precision_recall_curve(
            target, preds, num_classes=num_classes, thresholds=thresholds
        )
    elif task == "multilabel":
        assert (
            isinstance(num_labels, int) and num_labels > 0
        ), "Number of labels must be a positive integer."

        precision, recall, thresholds = multilabel_precision_recall_curve(
            target, preds, num_labels=num_labels, thresholds=thresholds
        )
    else:
        raise ValueError(
            "Expected argument `task` to be either 'binary', 'multiclass' or "
            f"'multilabel', but got {task}"
        )

    return precision, recall, thresholds
