"""Functional API for the matthews correlation coefficient (MCC) metric."""

from typing import Optional, Tuple, Union

import array_api_compat as apc

from cyclops.evaluate.metrics.experimental.functional.confusion_matrix import (
    _binary_confusion_matrix_compute,
    _binary_confusion_matrix_format_arrays,
    _binary_confusion_matrix_update_state,
    _binary_confusion_matrix_validate_args,
    _binary_confusion_matrix_validate_arrays,
    _multiclass_confusion_matrix_format_arrays,
    _multiclass_confusion_matrix_update_state,
    _multiclass_confusion_matrix_validate_args,
    _multiclass_confusion_matrix_validate_arrays,
    _multilabel_confusion_matrix_compute,
    _multilabel_confusion_matrix_format_arrays,
    _multilabel_confusion_matrix_update_state,
    _multilabel_confusion_matrix_validate_args,
    _multilabel_confusion_matrix_validate_arrays,
)
from cyclops.evaluate.metrics.experimental.utils.types import Array


def _mcc_reduce(confmat: Array) -> Array:
    """Reduce an un-normalized confusion matrix into the matthews corrcoef."""
    xp = apc.array_namespace(confmat)

    # convert multilabel into binary
    confmat = xp.sum(confmat, axis=0) if confmat.ndim == 3 else confmat

    if int(apc.size(confmat) or 0) == 4:  # binary case
        # convert tp, tn, fp, fn to float32 for type promotion rules to work
        tn, fp, fn, tp = xp.reshape(xp.astype(confmat, xp.float32, copy=False), (-1,))
        if tp + tn != 0 and fp + fn == 0:
            return xp.asarray(1.0, dtype=confmat.dtype, device=apc.device(confmat))  # type: ignore[no-any-return]

        if tp + tn == 0 and fp + fn != 0:
            return xp.asarray(-1.0, dtype=confmat.dtype, device=apc.device(confmat))  # type: ignore[no-any-return]

    tk = xp.sum(confmat, axis=-1, dtype=xp.float64)  # tn + fp and tp + fn
    pk = xp.sum(confmat, axis=-2, dtype=xp.float64)  # tn + fn and tp + fp
    c = xp.astype(xp.linalg.trace(confmat), xp.float64, copy=False)  # tn and tp
    s = xp.sum(confmat, dtype=xp.float64)  # tn + tp + fn + fp

    cov_ytyp = c * s - sum(tk * pk)
    cov_ypyp = s**2 - sum(pk * pk)
    cov_ytyt = s**2 - sum(tk * tk)

    numerator = cov_ytyp
    denom = cov_ypyp * cov_ytyt

    eps = xp.asarray(
        xp.finfo(xp.float32).eps,
        dtype=xp.float32,
        device=apc.device(confmat),
    )

    def _is_close_to_zero(a: Array) -> Array:
        """Check if an array is close to zero."""
        return xp.all(xp.abs(a) <= eps)  # type: ignore[no-any-return]

    if _is_close_to_zero(denom) and int(apc.size(confmat) or 0) == 4:
        if tp == 0 or tn == 0:
            a = tp + tn

        if fp == 0 or fn == 0:
            b = fp + fn

        numerator = xp.sqrt(eps) * (a - b)
        denom = (tp + fp + eps) * (tp + fn + eps) * (tn + fp + eps) * (tn + fn + eps)
    elif _is_close_to_zero(denom):
        return xp.asarray(0.0, dtype=confmat.dtype, device=apc.device(confmat))  # type: ignore[no-any-return]
    return numerator / xp.sqrt(denom)  # type: ignore[no-any-return]


def binary_mcc(
    target: Array,
    preds: Array,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
) -> Array:
    """Compute the matthews correlation coefficient for binary classification.

    Parameters
    ----------
    target : Array
        An array object that is compatible with the Python array API standard
        and contains the ground truth labels. The expected shape of the array
        is `(N, ...)`, where `N` is the number of samples.
    preds : Array
        An array object that is compatible with the Python array API standard and
        contains the predictions of a binary classifier. the expected shape of the
        array is `(N, ...)` where `N` is the number of samples. If `preds` contains
        floating point values that are not in the range `[0, 1]`, a sigmoid function
        will be applied to each value before thresholding.
    threshold : float, default=0.5
        The threshold to use when converting probabilities to binary predictions.
    ignore_index : int, optional, default=None
        Specifies a target value that is ignored and does not contribute to the
        metric. If `None`, ignore nothing.

    Returns
    -------
    Array
        The matthews correlation coefficient.

    Raises
    ------
    ValueError
        If `target` and `preds` have different shapes.
    ValueError
        If `target` and `preds` are not array-API-compatible.
    ValueError
        If `target` or `preds` are empty.
    ValueError
        If `target` or `preds` are not numeric arrays.
    ValueError
        If `threshold` is not a float in the [0,1] range.
    ValueError
        If `normalize` is not one of `'pred'`, `'true'`, `'all'`, `'none'`, or `None`.
    ValueError
        If `ignore_index` is not `None` or an integer.

    Examples
    --------
    >>> import numpy.array_api as anp
    >>> from cyclops.evaluate.metrics.experimental.functional import binary_mcc
    >>> target = anp.asarray([0, 1, 0, 1, 0, 1])
    >>> preds = anp.asarray([0, 0, 1, 1, 0, 1])
    >>> binary_mcc(target, preds)
    Array(0.33333333, dtype=float64)
    >>> target = anp.asarray([0, 1, 0, 1, 0, 1])
    >>> preds = anp.asarray([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])
    >>> binary_mcc(target, preds)
    Array(0.33333333, dtype=float64)

    """
    _binary_confusion_matrix_validate_args(
        threshold=threshold,
        normalize=None,
        ignore_index=ignore_index,
    )
    xp = _binary_confusion_matrix_validate_arrays(target, preds, ignore_index)

    target, preds = _binary_confusion_matrix_format_arrays(
        target,
        preds,
        threshold,
        ignore_index,
        xp=xp,
    )
    tn, fp, fn, tp = _binary_confusion_matrix_update_state(target, preds, xp=xp)

    confmat = _binary_confusion_matrix_compute(tn, fp, fn, tp, normalize=None)
    return _mcc_reduce(confmat)


def multiclass_mcc(
    target: Array,
    preds: Array,
    num_classes: int,
    ignore_index: Optional[Union[int, Tuple[int]]] = None,
) -> Array:
    """Compute the matthews correlation coefficient for multiclass classification.

    Parameters
    ----------
    target : Array
        The target array of shape `(N, ...)`, where `N` is the number of samples.
    preds : Array
        The prediction array with shape `(N, ...)`, for integer inputs, or
        `(N, C, ...)`, for float inputs, where `N` is the number of samples and
        `C` is the number of classes.
    num_classes : int
        The number of classes.
    ignore_index : int, Tuple[int], optional, default=None
        Specifies a target value(s) that is ignored and does not contribute to the
        metric. If `None`, ignore nothing.

    Returns
    -------
    Array
        The matthews correlation coefficient.

    Raises
    ------
    ValueError
        If `target` and `preds` are not array-API-compatible.
    ValueError
        If `target` or `preds` are empty.
    ValueError
        If `target` or `preds` are not numeric arrays.
    ValueError
        If `num_classes` is not an integer larger than 1.
    ValueError
        If `normalize` is not one of `'pred'`, `'true'`, `'all'`, `'none'`, or `None`.
    ValueError
        If `ignore_index` is not `None`, an integer or a tuple of integers.
    ValueError
        If `preds` contains floats but `target` does not have one dimension less than
        `preds`.
    ValueError
        If the second dimension of `preds` is not equal to `num_classes`.
    ValueError
        If when `target` has one dimension less than `preds`, the shape of `preds` is
        not `(N, C, ...)` while the shape of `target` is `(N, ...)`.
    ValueError
        If when `target` and `preds` have the same number of dimensions, they
        do not have the same shape.
    RuntimeError
        If `target` contains values that are not in the range [0, `num_classes`).

    Examples
    --------
    >>> import numpy.array_api as anp
    >>> from cyclops.evaluate.metrics.experimental.functional import multiclass_mcc
    >>> target = anp.asarray([2, 1, 0, 0])
    >>> preds = anp.asarray([2, 1, 0, 1])
    >>> multiclass_mcc(target, preds, num_classes=3)
    Array(0.7, dtype=float64)
    >>> target = anp.asarray([2, 1, 0, 0])
    >>> preds = anp.asarray(
    ...     [
    ...         [0.16, 0.26, 0.58],
    ...         [0.22, 0.61, 0.17],
    ...         [0.71, 0.09, 0.20],
    ...         [0.05, 0.82, 0.13],
    ...     ]
    ... )
    >>> multiclass_mcc(target, preds, num_classes=3)
    Array(0.7, dtype=float64)

    """
    _multiclass_confusion_matrix_validate_args(
        num_classes,
        normalize=None,
        ignore_index=ignore_index,
    )
    xp = _multiclass_confusion_matrix_validate_arrays(
        target,
        preds,
        num_classes,
        ignore_index=ignore_index,
    )

    target, preds = _multiclass_confusion_matrix_format_arrays(
        target,
        preds,
        ignore_index=ignore_index,
        xp=xp,
    )
    confmat = _multiclass_confusion_matrix_update_state(
        target,
        preds,
        num_classes,
        xp=xp,
    )
    return _mcc_reduce(confmat)


def multilabel_mcc(
    target: Array,
    preds: Array,
    num_labels: int,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
) -> Array:
    """Compute the matthews correlation coefficient for multilabel classification.

    Parameters
    ----------
    target : Array
        The target array of shape `(N, L, ...)`, where `N` is the number of samples
        and `L` is the number of labels.
    preds : Array
        The prediction array of shape `(N, L, ...)`, where `N` is the number of
        samples and `L` is the number of labels. If `preds` contains floats that
        are not in the range [0,1], they will be converted to probabilities using
        the sigmoid function.
    num_labels : int
        The number of labels.
    threshold : float, default=0.5
        The threshold to use for binarizing the predictions.
    ignore_index : int, optional, default=None
        Specifies a target value that is ignored and does not contribute to the
        metric. If `None`, ignore nothing.

    Returns
    -------
    Array
        The matthews correlation coefficient.

    Raises
    ------
    ValueError
        If `target` and `preds` are not array-API-compatible.
    ValueError
        If `target` or `preds` are empty.
    ValueError
        If `target` or `preds` are not numeric arrays.
    ValueError
        If `threshold` is not a float in the [0,1] range.
    ValueError
        If `normalize` is not one of `'pred'`, `'true'`, `'all'`, `'none'`, or `None`.
    ValueError
        If `ignore_index` is not `None` or a non-negative integer.
    ValueError
        If `num_labels` is not an integer larger than 1.
    ValueError
        If `target` and `preds` do not have the same shape.
    ValueError
        If the second dimension of `preds` is not equal to `num_labels`.
    RuntimeError
        If `target` contains values that are not in the range [0, 1].

    Examples
    --------
    >>> import numpy.array_api as anp
    >>> from cyclops.evaluate.metrics.experimental.functional import multilabel_mcc
    >>> target = anp.asarray([[0, 1, 0], [1, 0, 1]])
    >>> preds = anp.asarray([[0, 0, 1], [1, 0, 1]])
    >>> multilabel_mcc(target, preds, num_labels=3)
    Array(0.33333333, dtype=float64)
    >>> target = anp.asarray([[0, 1, 0], [1, 0, 1]])
    >>> preds = anp.asarray([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
    >>> multilabel_mcc(target, preds, num_labels=3)
    Array(0.33333333, dtype=float64)

    """
    _multilabel_confusion_matrix_validate_args(
        num_labels,
        threshold=threshold,
        normalize=None,
        ignore_index=ignore_index,
    )
    xp = _multilabel_confusion_matrix_validate_arrays(
        target,
        preds,
        num_labels,
        ignore_index=ignore_index,
    )

    target, preds = _multilabel_confusion_matrix_format_arrays(
        target,
        preds,
        threshold=threshold,
        ignore_index=ignore_index,
        xp=xp,
    )
    tn, fp, fn, tp = _multilabel_confusion_matrix_update_state(target, preds, xp=xp)

    confmat = _multilabel_confusion_matrix_compute(
        tn,
        fp,
        fn,
        tp,
        num_labels,
        normalize=None,
    )
    return _mcc_reduce(confmat)
