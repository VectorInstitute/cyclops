"""Functions for computing the area under the ROC curve (AUROC)."""

import warnings
from typing import List, Literal, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import auc

from cyclops.evaluate.metrics.functional.precision_recall_curve import (
    _binary_precision_recall_curve_format,
    _binary_precision_recall_curve_update,
    _format_thresholds,
    _multiclass_precision_recall_curve_format,
    _multiclass_precision_recall_curve_update,
    _multilabel_precision_recall_curve_format,
    _multilabel_precision_recall_curve_update,
)
from cyclops.evaluate.metrics.functional.roc import (
    _binary_roc_compute,
    _multiclass_roc_compute,
    _multilabel_roc_compute,
)
from cyclops.evaluate.metrics.utils import _check_thresholds


def _reduce_auroc(
    fpr: Union[np.ndarray, List[np.ndarray]],
    tpr: Union[np.ndarray, List[np.ndarray]],
    average: Literal["macro", "weighted"] = None,
    weights: np.ndarray = None,
) -> Union[float, np.ndarray]:
    """Compute the area under the ROC curve and apply ``average`` method.

    Parameters
    ----------
        fpr : numpy.ndarray or list of numpy.ndarray
            False positive rate.
        tpr : numpy.ndarray or list of numpy.ndarray
            True positive rate.
        average : Literal["macro", "weighted"], default=None
            If not None, apply the method to compute the average area under the
            ROC curve.
        weights : numpy.ndarray, default=None
            Sample weights.

    Returns
    -------
        auroc : float or numpy.ndarray
            Area under the ROC curve. If ``average`` is not None, ``auroc`` is a
            numpy array.

    Raises
    ------
        ValueError
            If ``average`` is not one of ``macro`` or ``weighted`` or if
            ``average`` is ``weighted`` and ``weights`` is None.

    """
    result = [
        auc(x, y) for x, y in zip(fpr, tpr)
    ]  # without the loop: np.trapz(tpr, fpr, axis=1) * direction
    result = np.stack(result)

    if average is not None:
        if np.isnan(result).any():
            warnings.warn(
                "Average precision score for one or more classes was `nan`."
                f" Ignoring these classes in {average}-average",
                UserWarning,
            )
        idx = ~np.isnan(result)

        if average == "macro":
            result = result[idx].mean()
        elif average == "weighted" and weights is not None:
            weights = np.divide(
                weights[idx],
                weights[idx].sum(),
                out=np.zeros_like(weights, dtype=np.float64),
                where=weights[idx].sum() != 0,
            )
            result = (result[idx] * weights).sum()
        else:
            raise ValueError(
                "Received an incompatible combinations of inputs to make reduction."
            )

    return result


def _binary_auroc_compute(
    state: Union[Tuple[np.ndarray, np.ndarray], np.ndarray],
    thresholds: np.ndarray = None,
    max_fpr: float = None,
    pos_label: int = 1,
) -> float:
    """Compute the area under the ROC curve for binary classification tasks.

    Parameters
    ----------
        state : Union[Tuple[numpy.ndarray, numpy.ndarray], numpy.ndarray]
            If ``state`` is a tuple, then it must be a tuple of two numpy arrays
            ``(target, preds)``. If ``state`` is a numpy array, then it is a multi-
            threshold confusion matrix.
        thresholds : numpy.ndarray, default=None
            Thresholds used for computing binarizing the predictions. If None,
            then the thresholds are automatically determined by the unique values
            in ``preds``.
        max_fpr : float, default=None
            The maximum value of the false positive rate. If ``None``, the
            false positive rate is set to the complement of the true positive
            rate.
        pos_label : int, default=1
            The label of the positive class.

    Returns
    -------
        auroc : float
            Area under the ROC curve.

    """
    fpr, tpr, _ = _binary_roc_compute(state, thresholds=thresholds, pos_label=pos_label)

    if max_fpr is None or max_fpr == 1:
        return auc(x=fpr, y=tpr)

    # Add a single point at max_fpr by linear interpolation
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    partial_auc = auc(fpr, tpr)

    # McClish correction: standardize result to be 0.5 if non-discriminant
    # and 1 if maximal
    min_area = 0.5 * max_fpr**2
    max_area = max_fpr

    return 0.5 * (1 + (partial_auc - min_area) / (max_area - min_area))


def binary_auroc(
    target: ArrayLike,
    preds: ArrayLike,
    max_fpr: float = None,
    thresholds: Union[int, List[float], np.ndarray] = None,
) -> float:
    """Compute the area under the ROC curve for binary classification tasks.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Estimated probabilities or decision function. If the values in ``preds``
            are not in the range [0, 1], then they will be transformed to this range
            via a sigmoid function.
        max_fpr : float, default=None
            The maximum value of the false positive rate. If not None, then
            the partial AUCROC in the range [0, max_fpr] is returned.
        thresholds : Union[int, List[float], numpy.ndarray], default=None
            Thresholds used for binarizing the values of ``preds``.
            If int, then the number of thresholds to use.
            If list or array, then the thresholds to use.
            If None, then the thresholds are automatically determined by the
            unique values in ``preds``.

    Returns
    -------
        auroc : float
            Area under the ROC curve.

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import binary_auroc
        >>> target = [1, 0, 0, 1]
        >>> preds = [0.1, 0.9, 0.4, 0.6]
        >>> binary_auroc(target, preds, thresholds=5)
        0.25

    """
    _check_thresholds(thresholds)

    if max_fpr is not None:
        if not isinstance(max_fpr, (int, float)):
            raise ValueError(
                "Expected argument ``max_fpr`` to be a float or integer, but got"
                f" {max_fpr}"
            )
        if max_fpr <= 0 or max_fpr > 1:
            raise ValueError(
                "Expected argument ``max_fpr`` to be in the range (0, 1], but got"
                f" {max_fpr}"
            )

    target, preds = _binary_precision_recall_curve_format(target, preds)
    thresholds = _format_thresholds(thresholds)

    state = _binary_precision_recall_curve_update(target, preds, thresholds)

    return _binary_auroc_compute(state, thresholds=thresholds, max_fpr=max_fpr)


def _multiclass_auroc_compute(
    state: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    num_classes: int,
    thresholds: np.ndarray = None,
    average: Literal["macro", "weighted"] = None,
) -> Union[float, np.ndarray]:
    """Compute the area under the ROC curve for multiclass classification tasks.

    Parameters
    ----------
        state : Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]
            If ``state`` is a numpy array, then it is a multi-threshold confusion
            matrix. If ``state`` is a tuple, then it must be a tuple of two numpy
            arrays ``(target, preds)``.
        num_classes : int
            Number of classes.
        thresholds : numpy.ndarray, default=None
            Thresholds used for computing binarizing the predictions. If None,
            then the thresholds are automatically determined by the unique values
            in ``preds``.
        average : Literal["macro", "weighted"], default=None
            If ``None``, then the scores for each class are returned. Otherwise,
            this determines the type of averaging performed on the scores.

    Returns
    -------
        auroc : Union[float, numpy.ndarray]
            Area under the ROC curve. If ``average`` is ``None``, then a numpy array
            of shape (num_classes,) is returned, otherwise a float is returned.

    """
    fpr, tpr, _ = _multiclass_roc_compute(state, num_classes, thresholds=thresholds)
    return _reduce_auroc(
        fpr,
        tpr,
        average=average,
        weights=np.bincount(state[0], minlength=num_classes)
        if thresholds is None
        else state[1][:, 1, :].sum(-1),
    )


def multiclass_auroc(
    target: ArrayLike,
    preds: ArrayLike,
    num_classes: int,
    thresholds: Union[int, List[float], np.ndarray] = None,
    average: Literal["macro", "weighted"] = None,
) -> Union[float, np.ndarray]:
    """Compute the area under the ROC curve for multiclass classification tasks.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Estimated probabilities or decision function. If the values in ``preds``
            are not in the range [0, 1], then they will be transformed to this range
            via a softmax function.
        num_classes : int
            Number of classes.
        thresholds : Union[int, List[float], numpy.ndarray], default=None
            Thresholds used for binarizing the values of ``preds``.
            If int, then the number of thresholds to use.
            If list or array, then the thresholds to use.
            If None, then the thresholds are automatically determined by the
            unique values in ``preds``.
        average : Literal["macro", "weighted"], default=None
            If ``None``, then the scores for each class are returned. Otherwise,
            this determines the type of averaging performed on the scores. One of
            - `macro`: Calculate metrics for each class, and find their unweighted
              mean. This does not take class imbalance into account.
            - `weighted`: Calculate metrics for each class, and find their average,
              weighted by support (the number of true instances for each class).

    Returns
    -------
        auroc : Union[float, numpy.ndarray]
            Area under the ROC curve. If ``average`` is ``None``, then a numpy array
            of shape (num_classes,) is returned, otherwise a float is returned.

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import multiclass_auroc
        >>> target = [1, 0, 2, 0]
        >>> preds = [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05],
        ...         [0.05, 0.05, 0.9], [0.9, 0.05, 0.05]]
        >>> multiclass_auroc(target, preds, num_classes=3, thresholds=5,
        ...     average=None
        ... )
        array([0.5       , 0.33333333, 1.        ])

    """
    _check_thresholds(thresholds)

    if average is not None and average not in ("macro", "weighted"):
        raise ValueError(
            "Expected argument `average` to be one of ('macro', 'weighted', None),"
            f" but got {average}"
        )

    target, preds = _multiclass_precision_recall_curve_format(
        target, preds, num_classes=num_classes
    )
    thresholds = _format_thresholds(thresholds)

    state = _multiclass_precision_recall_curve_update(
        target, preds, num_classes=num_classes, thresholds=thresholds
    )

    return _multiclass_auroc_compute(
        state, num_classes, thresholds=thresholds, average=average
    )


def _multilabel_auroc_compute(
    state: Union[Tuple[np.ndarray, np.ndarray], np.ndarray],
    num_labels: int,
    thresholds: np.ndarray = None,
    average: Literal["micro", "macro", "weighted"] = None,
) -> Union[float, np.ndarray]:
    """Compute the area under the ROC curve for multilabel classification tasks.

    Parameters
    ----------
        state : Union[Tuple[numpy.ndarray, numpy.ndarray], numpy.ndarray]
            If ``state`` is a numpy array, then it is a multi-threshold confusion
            matrix. If ``state`` is a tuple, then it must be a tuple of two numpy
            arrays ``(target, preds)``.
        num_labels : int
            Number of labels.
        thresholds : numpy.ndarray, default=None
            Thresholds used for computing binarizing the predictions. If None,
            then the thresholds are automatically determined by the unique values
            in ``preds``.
        average : Literal["micro", "macro", "weighted"], default=None
            If ``None``, then the scores for each label are returned. Otherwise,
            this determines the type of averaging performed on the scores. One of
            - `micro`: Calculate metrics globally.
            - `macro`: Calculate metrics for each label, and find their unweighted
              mean. This does not take label imbalance into account.
            - `weighted`: Calculate metrics for each label, and find their average,
              weighted by support (the number of true instances for each label).

    Returns
    -------
        float or numpy.ndarray
            Area under the ROC curve. If ``average`` is ``None``, then a numpy array
            of shape (num_labels,) is returned, otherwise a float is returned.

    """
    if average == "micro":
        if isinstance(state, np.ndarray) and thresholds is not None:
            return _binary_auroc_compute(state.sum(1), thresholds, max_fpr=None)

        target = state[0].flatten()
        preds = state[1].flatten()
        return _binary_auroc_compute((target, preds), thresholds, max_fpr=None)
    fpr, tpr, _ = _multilabel_roc_compute(state, num_labels, thresholds=thresholds)
    return _reduce_auroc(
        fpr,
        tpr,
        average=average,
        weights=(state[0] == 1).sum(axis=0).astype(np.float64)
        if thresholds is None
        else state[1][:, 1, :].sum(axis=-1),
    )


def multilabel_auroc(
    target: ArrayLike,
    preds: ArrayLike,
    num_labels: int,
    thresholds: Union[int, List[float], np.ndarray] = None,
    average: Literal["micro", "macro", "weighted"] = None,
) -> Union[float, np.ndarray]:
    """Compute the area under the ROC curve for multilabel classification tasks.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Estimated probabilities or decision function. If the values in ``preds``
            are not in the range [0, 1], then they will be transformed to this range
            via a softmax function.
        num_labels : int
            Number of labels.
        thresholds : Union[int, List[float], numpy.ndarray], default=None
            Thresholds used for binarizing the values of ``preds``.
            If int, then the number of thresholds to use.
            If list or array, then the thresholds to use.
            If None, then the thresholds are automatically determined by the
            unique values in ``preds``.
        average : Literal["micro", "macro", "weighted"], default=None
            If ``None``, then the scores for each label are returned. Otherwise,
            this determines the type of averaging performed on the scores. One of
            - `micro`: Calculate metrics globally by counting the total true
                positives, false negatives and false positives.
            - `macro`: Calculate metrics for each label, and find their unweighted
                mean. This does not take label imbalance into account.
            - `weighted``: Calculate metrics for each label, and find their average,
                weighted by support (the number of true instances for each label).

    Returns
    -------
        float or numpy.ndarray
            Area under the ROC curve. If ``average`` is ``None``, then a numpy array
            of shape (num_labels,) is returned, otherwise a float is returned.

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import multilabel_auroc
        >>> target = [[0, 1, 0], [0, 1, 1], [1, 0, 1]]
        >>> preds = [[0.1, 0.9, 0.8], [0.05, 0.1, 0.9], [0.8, 0.2, 0.3]]
        >>> multilabel_auroc(target, preds, num_labels=3, thresholds=5,
        ...     average=None)
        array([1.  , 0.75, 0.25])

    """
    _check_thresholds(thresholds)

    if average is not None and average not in ("micro", "macro", "weighted"):
        raise ValueError(
            "Expected argument `average` to be one of ('micro', 'macro', 'weighted'"
            f" , None), but got {average}"
        )

    target, preds = _multilabel_precision_recall_curve_format(
        target, preds, num_labels=num_labels
    )
    thresholds = _format_thresholds(thresholds)

    state = _multilabel_precision_recall_curve_update(
        target, preds, num_labels=num_labels, thresholds=thresholds
    )

    return _multilabel_auroc_compute(
        state, num_labels, thresholds=thresholds, average=average
    )


def auroc(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    task: Literal["binary", "multiclass", "multilabel"],
    max_fpr: float = None,
    thresholds: Union[int, List[float], np.ndarray] = None,
    num_classes: int = None,
    num_labels: int = None,
    average: Literal["micro", "macro", "weighted"] = None,
) -> Union[float, np.ndarray]:
    """Compute the area under the ROC curve for different tasks.

        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Estimated probabilities or decision function. If ``preds`` is not in
            the range [0, 1], a sigmoid function is applied to transform it to
            the range [0, 1].
        task : Literal["binary", "multiclass", "multilabel"]
            Task type. One of ``binary``, ``multiclass``, ``multilabel``.
        max_fpr : float, default=None
            The maximum value of the false positive rate. If not None, the
            a partial AUC in the range [0, max_fpr] is returned. Only used for
            binary classification.
        thresholds : int or list of floats or numpy.ndarray of floats, default=None
            Thresholds used for binarizing the values of ``preds``.
            If int, then the number of thresholds to use.
            If list or array, then the thresholds to use.
            If None, then the thresholds are automatically determined by the
            unique values in ``preds``.
        num_classes : int, default=None
            Number of classes. This parameter is required for the ``multiclass``
            task.
        num_labels : int, default=None
            Number of labels. This parameter is required for the ``multilabel``
            task.
        average : Literal["micro", "macro", "weighted"], default=None
            If not None, apply the method to compute the average area under the
            ROC curve. Only applicable for the ``multiclass`` and ``multilabel``
            tasks. One of:
                - ``micro``: Calculate metrics globally by counting the total true
                  positives, false negatives and false positives.
                - ``macro``: Calculate metrics for each label, and find their unweighted
                  mean. This does not take label imbalance into account.
                - ``weighted``: Calculate metrics for each label, and find their
                  average, weighted by support (accounting for label imbalance).

    Returns
    -------
        auroc_score : float or numpy.ndarray
            Area under the ROC curve. If ``average`` is None or task is ``binary``,
            ``auroc`` is a float. Otherwise, ``auroc`` is a numpy array.

    Examples (binary)
    -----------------
        >>> from cyclops.evaluation.metrics.functional import auroc
        >>> target = [0, 1, 0, 1]
        >>> preds = [0.1, 0.35, 0.4, 0.8]
        >>> auroc(target, preds, task="binary")
        0.75

    Examples (multiclass)
    ---------------------
        >>> from cyclops.evaluation.metrics.functional import auroc
        >>> target = [0, 1, 2, 0, 1, 2]
        >>> preds = [[0.1, 0.6, 0.3], [0.05, 0.95, 0], [0.5, 0.3, 0.2],
        ...          [0.1, 0.6, 0.3], [0.05, 0.95, 0], [0.5, 0.3, 0.2]]
        >>> auroc(target, preds, task="multiclass", num_classes=3, average=None)
        array([0.5, 1. , 0.5])

    Examples (multilabel)
    ---------------------
        >>> from cyclops.evaluation.metrics.functional import auroc
        >>> target = [[0, 1], [1, 1], [0, 0], [1, 0]]
        >>> preds = [[0.1, 0.9], [0.8, 0.2], [0.4, 0.6], [0.2, 0.8]]
        >>> auroc(target, preds, task="multilabel", num_labels=2, average=None)
        array([0.25, 0.5 ])

    """
    if task == "binary":
        auroc_score = binary_auroc(
            target, preds, max_fpr=max_fpr, thresholds=thresholds
        )
    elif task == "multiclass":
        assert (
            isinstance(num_classes, int) and num_classes > 0
        ), "Number of classes must be a positive integer."
        auroc_score = multiclass_auroc(
            target,
            preds,
            num_classes,
            thresholds=thresholds,
            average=average,  # type: ignore
        )
    elif task == "multilabel":
        assert (
            isinstance(num_labels, int) and num_labels > 0
        ), "Number of labels must be a positive integer."
        auroc_score = multilabel_auroc(
            target, preds, num_labels, thresholds=thresholds, average=average
        )
    else:
        raise ValueError(
            "Expected argument `task` to be either 'binary', 'multiclass' or "
            f"'multilabel', but got {task}"
        )

    return auroc_score
