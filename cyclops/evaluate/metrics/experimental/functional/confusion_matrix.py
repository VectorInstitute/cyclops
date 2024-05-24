"""Functions for computing the confusion matrix for classification tasks."""

# mypy: disable-error-code="no-any-return"
import warnings
from types import ModuleType
from typing import Literal, Optional, Tuple, Union

import array_api_compat as apc

from cyclops.evaluate.metrics.experimental.utils.ops import (
    bincount,
    clone,
    flatten,
    remove_ignore_index,
    safe_divide,
    sigmoid,
    squeeze_all,
    to_int,
)
from cyclops.evaluate.metrics.experimental.utils.types import Array
from cyclops.evaluate.metrics.experimental.utils.validation import (
    _basic_input_array_checks,
    _check_same_shape,
    is_floating_point,
)


def _normalize_confusion_matrix(
    confmat: Array,
    normalize: Optional[str] = None,
    *,
    xp: ModuleType,
) -> Array:
    """Normalize the confusion matrix."""
    if normalize in ["true", "pred", "all"]:
        confmat = xp.astype(confmat, xp.float32, copy=False)

    if normalize == "pred":
        return safe_divide(confmat, xp.sum(confmat, axis=-2, keepdims=True))
    if normalize == "true":
        return safe_divide(confmat, xp.sum(confmat, axis=-1, keepdims=True))
    if normalize == "all":
        return safe_divide(confmat, xp.sum(confmat, axis=(-1, -2), keepdims=True))

    nan_elements = int(0 or apc.size(confmat[xp.isnan(confmat)]))
    if nan_elements:
        confmat[xp.isnan(confmat)] = 0
        warnings.warn(
            f"Encountered {nan_elements} NaN elements in the confusion matrix. "
            "These elements were replaced with 0.",
            category=RuntimeWarning,
            stacklevel=1,
        )

    return confmat


def _binary_confusion_matrix_validate_args(
    threshold: float = 0.5,
    normalize: Optional[str] = None,
    ignore_index: Optional[int] = None,
) -> None:
    """Validate the arguments of the `binary_confusion_matrix` method."""
    if not (isinstance(threshold, float) and (0.0 <= threshold <= 1.0)):
        raise ValueError(
            "Expected argument `threshold` to be a float in the [0,1] range, "
            f"but got {threshold}.",
        )

    allowed_normalize = ("true", "pred", "all", "none", None)
    if normalize not in allowed_normalize:
        raise ValueError(
            f"Expected argument `normalize` to be one of {allowed_normalize}, "
            f"but got {normalize}",
        )

    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(
            "Expected argument `ignore_index` to either be `None`, an integer, "
            f" but got {ignore_index}",
        )


def _binary_confusion_matrix_validate_arrays(
    target: Array,
    preds: Array,
    ignore_index: Optional[int] = None,
) -> ModuleType:
    """Validate the inputs of the `binary_confusion_matrix` method."""
    _basic_input_array_checks(target, preds)
    _check_same_shape(target, preds)

    xp = apc.array_namespace(target, preds)

    unique_values = xp.unique_values(target)
    if ignore_index is None:
        check = xp.any((unique_values != 0) & (unique_values != 1))
    else:
        check = xp.any(
            (unique_values != 0)
            & (unique_values != 1)
            & (unique_values != ignore_index),
        )
    if check:
        raise RuntimeError(
            "Expected only the following values "
            f"{[0, 1] if ignore_index is None else [ignore_index]} in `target`. "
            f"But found the following values: {unique_values}",
        )

    if not is_floating_point(preds):
        unique_values = xp.unique_values(preds)
        if xp.any((unique_values != 0) & (unique_values != 1)):
            raise RuntimeError(
                "Expected only the following values "
                f"{[0, 1] if ignore_index is None else [ignore_index]} in `preds`. "
                f"But found the following values: {unique_values}",
            )

    return xp


def _binary_confusion_matrix_format_arrays(
    target: Array,
    preds: Array,
    threshold: float,
    ignore_index: Optional[int],
    *,
    xp: ModuleType,
) -> Tuple[Array, Array]:
    """Format the input arrays of the `binary_confusion_matrix` method."""
    preds = flatten(preds)
    target = flatten(target)

    if ignore_index is not None:
        target, preds = remove_ignore_index(target, preds, ignore_index=ignore_index)

    if is_floating_point(preds):
        # NOTE: in the 2021.12 version of the the array API standard the `__mul__`
        # operator is only defined for numeric arrays (including float and int scalars)
        # so we convert the boolean array to an integer array first.
        if not xp.all(to_int((preds >= 0)) * to_int((preds <= 1))):  # preds are logits
            preds = sigmoid(preds)  # convert to probabilities with sigmoid
        preds_device = apc.device(preds)
        preds = xp.where(
            preds > threshold,
            xp.asarray(1, dtype=xp.int32, device=preds_device),
            xp.asarray(0, dtype=xp.int32, device=preds_device),
        )

    return target, preds


def _binary_confusion_matrix_update_state(
    target: Array,
    preds: Array,
    *,
    xp: ModuleType,
) -> Tuple[Array, Array, Array, Array]:
    """Compute stat scores for the given `target` and `preds` arrays."""
    # NOTE: in the 2021.12 version of the array API standard, the `sum` method
    # only supports numeric types, so we have to cast the boolean arrays to integers.
    # Also, the `squeeze` method in the array API standard does not support `axis=None`
    # so we define a custom method `squeeze_all` to squeeze all singleton dimensions.
    tp = squeeze_all(xp.sum(to_int((target == preds) & (target == 1))))
    fn = squeeze_all(xp.sum(to_int((target != preds) & (target == 1))))
    fp = squeeze_all(xp.sum(to_int((target != preds) & (target == 0))))
    tn = squeeze_all(xp.sum(to_int((target == preds) & (target == 0))))

    return tn, fp, fn, tp


def _binary_confusion_matrix_compute(
    tn: Array,
    fp: Array,
    fn: Array,
    tp: Array,
    normalize: Optional[str] = None,
) -> Array:
    """Compute the confusion matrix from the given stat scores."""
    xp = apc.array_namespace(tn, fp, fn, tp)

    confmat = squeeze_all(
        xp.reshape(xp.stack([tn, fp, fn, tp], axis=0), shape=(-1, 2, 2)),
    )
    return _normalize_confusion_matrix(confmat, normalize=normalize, xp=xp)


def binary_confusion_matrix(
    target: Array,
    preds: Array,
    threshold: float = 0.5,
    normalize: Optional[Literal["pred", "true", "all", "none"]] = None,
    ignore_index: Optional[int] = None,
) -> Array:
    """Compute the confusion matrix for binary classification tasks.

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
    normalize : str, optional, default=None
        Normalization mode.
        If `None` or `'none'`, return the number of correctly classified samples
        for each class.
        If `'true'`, return the fraction of correctly classified samples for each
        class over the number of samples with the same true class.
        If `'pred'`, return the fraction of samples of each class that were correctly
        classified over the number of samples with the same predicted class.
        If `'all'`, return the fraction of correctly classified samples over all
        samples.
    ignore_index : int, optional, default=None
        Specifies a target value that is ignored and does not contribute to the
        confusion matrix. If `None`, ignore nothing.

    Returns
    -------
    Array
        The confusion matrix with shape `(2, 2)`.

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
    >>> from cyclops.evaluate.metrics.experimental.functional import (
    ...     binary_confusion_matrix,
    ... )
    >>> target = anp.asarray([0, 1, 0, 1, 0, 1])
    >>> preds = anp.asarray([0, 0, 1, 1, 0, 1])
    >>> binary_confusion_matrix(target, preds)
    Array([[2, 1],
           [1, 2]], dtype=int64)
    >>> target = anp.asarray([0, 1, 0, 1, 0, 1])
    >>> preds = anp.asarray([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])
    >>> binary_confusion_matrix(target, preds)
    Array([[2, 1],
           [1, 2]], dtype=int64)

    """  # noqa: W505
    _binary_confusion_matrix_validate_args(
        threshold=threshold,
        normalize=normalize,
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

    return _binary_confusion_matrix_compute(tn, fp, fn, tp, normalize=normalize)


def _multiclass_confusion_matrix_validate_args(
    num_classes: int,
    normalize: Optional[str] = None,
    ignore_index: Optional[Union[int, Tuple[int]]] = None,
) -> None:
    """Validate the arguments of the `multiclass_confusion_matrix` method."""
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError(
            "Expected argument `num_classes` to be an integer larger than 1, "
            f"but got {num_classes}.",
        )

    allowed_normalize = ("true", "pred", "all", "none", None)
    if normalize not in allowed_normalize:
        raise ValueError(
            f"Expected argument `normalize` to be one of {allowed_normalize}, "
            f"but got {normalize}",
        )

    if ignore_index is not None and not (
        isinstance(ignore_index, int)
        or (
            isinstance(ignore_index, tuple)
            and all(isinstance(i, int) for i in ignore_index)
        )
    ):
        raise ValueError(
            "Expected argument `ignore_index` to either be `None`, an integer, "
            f"or a tuple of integers got {ignore_index}",
        )


def _multiclass_confusion_matrix_validate_arrays(
    target: Array,
    preds: Array,
    num_classes: int,
    ignore_index: Optional[Union[int, Tuple[int]]] = None,
) -> ModuleType:
    """Validate the inputs of the `multiclass_confusion_matrix` method."""
    _basic_input_array_checks(target, preds)
    xp = apc.array_namespace(target, preds)

    if preds.ndim == target.ndim + 1:
        if not is_floating_point(preds):
            raise ValueError(
                "If `preds` have one dimension more than `target`, `preds` should "
                "contain floating point values.",
            )

        if target.ndim == 0 and preds.shape[0] != num_classes:
            raise ValueError(
                "If `target` is a scalar and `preds` has one dimension more than "
                "`target`, the first dimension of `preds` should be equal to number "
                "of classes.",
            )
        if target.ndim >= 1 and preds.shape[1] != num_classes:
            raise ValueError(
                "If `preds` have one dimension more than `target`, the second "
                "dimension of `preds` should be equal to number of classes.",
            )
        if preds.shape[2:] != target.shape[1:]:
            raise ValueError(
                "If `preds` have one dimension more than `target`, the shape of "
                "`preds` should be (N, C, ...), and the shape of `target` should "
                "be (N, ...).",
            )
    elif preds.ndim == target.ndim:
        _check_same_shape(target, preds)
    else:
        raise ValueError(
            "Either `preds` and `target` both should have the (same) shape (N, ...), "
            "or the shape of `target` should be (N, ...) and the shape of `preds` "
            "should be (N, C, ...).",
        )

    num_unique_values = apc.size(xp.unique_values(target))
    num_allowed_extra_values = 0
    if ignore_index is not None:
        num_allowed_extra_values = (
            1 if isinstance(ignore_index, int) else len(ignore_index)
        )
    check = num_unique_values is None or (
        num_unique_values > num_classes
        if ignore_index is None
        else num_unique_values > num_classes + num_allowed_extra_values
    )
    if check:
        raise RuntimeError(
            f"Expected only {num_classes if ignore_index is None else num_classes + num_allowed_extra_values} "
            f"values in `target` but found {num_unique_values} values.",
        )

    if not is_floating_point(preds):
        unique_values = xp.unique_values(preds)
        num_unique_values = apc.size(unique_values)
        if num_unique_values is None or num_unique_values > num_classes:
            raise RuntimeError(
                f"Expected only {num_classes} values in `preds` but found "
                f"{num_unique_values} values.",
            )

    return xp


def _multiclass_confusion_matrix_format_arrays(
    target: Array,
    preds: Array,
    ignore_index: Optional[Union[int, Tuple[int]]] = None,
    *,
    xp: ModuleType,
) -> Tuple[Array, Array]:
    """Format the input arrays of the `multiclass_confusion_matrix` method."""
    if preds.ndim == target.ndim + 1:
        axis = 1 if preds.ndim > 1 else 0
        preds = xp.argmax(preds, axis=axis)

    target, preds = flatten(target), flatten(preds)

    if ignore_index is not None:
        target, preds = remove_ignore_index(target, preds, ignore_index=ignore_index)

    return target, preds


def _multiclass_confusion_matrix_update_state(
    target: Array,
    preds: Array,
    num_classes: int,
    *,
    xp: ModuleType,
) -> Array:
    """Compute the confusion matrix for the given `target` and `preds` arrays."""
    unique_mapping = to_int(target) * num_classes + to_int(preds)
    bins = bincount(unique_mapping, minlength=num_classes**2)

    return squeeze_all(xp.reshape(bins, shape=(-1, num_classes, num_classes)))


def _multiclass_confusion_matrix_compute(
    confmat: Array,
    normalize: Optional[str] = None,
) -> Array:
    """Normalize the confusion matrix."""
    xp = apc.array_namespace(confmat)
    return _normalize_confusion_matrix(confmat, normalize=normalize, xp=xp)


def multiclass_confusion_matrix(
    target: Array,
    preds: Array,
    num_classes: int,
    normalize: Optional[Literal["pred", "true", "all", "none"]] = None,
    ignore_index: Optional[Union[int, Tuple[int]]] = None,
) -> Array:
    """Compute the confusion matrix for multiclass classification tasks.

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
    normalize : str, optional, default=None
        Normalization mode.
        If `None` or `'none'`, return the number of correctly classified samples
        for each class.
        If `'true'`, return the fraction of correctly classified samples for each
        class over the number of samples with the same true class.
        If `'pred'`, return the fraction of samples of each class that were correctly
        classified over the number of samples with the same predicted class.
        If `'all'`, return the fraction of correctly classified samples over all
        samples.
    ignore_index : int, Tuple[int], optional, default=None
        Specifies a target value(s) that is ignored and does not contribute to the
        confusion matrix. If `None`, ignore nothing.

    Returns
    -------
    Array
        The confusion matrix with shape `(C, C)`, where `C` is the number of classes.

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
        If `ignore_index` is not `None`, an integer or a tuple of integers.\
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
    >>> from cyclops.evaluate.metrics.experimental.functional import multiclass_confusion_matrix
    >>> target = anp.asarray([2, 1, 0, 0])
    >>> preds = anp.asarray([2, 1, 0, 1])
    >>> multiclass_confusion_matrix(target, preds, num_classes=3)
    Array([[1, 1, 0],
           [0, 1, 0],
           [0, 0, 1]], dtype=int64)
    >>> target = anp.asarray([2, 1, 0, 0])
    >>> preds = anp.asarray([[0.16, 0.26, 0.58],
    ...                     [0.22, 0.61, 0.17],
    ...                     [0.71, 0.09, 0.20],
    ...                     [0.05, 0.82, 0.13]])
    >>> multiclass_confusion_matrix(target, preds, num_classes=3)
    Array([[1, 1, 0],
           [0, 1, 0],
           [0, 0, 1]], dtype=int64)

    """  # noqa: W505
    _multiclass_confusion_matrix_validate_args(
        num_classes,
        normalize=normalize,
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

    return _multiclass_confusion_matrix_compute(confmat, normalize)


def _multilabel_confusion_matrix_validate_args(
    num_labels: int,
    threshold: float = 0.5,
    normalize: Optional[str] = None,
    ignore_index: Optional[int] = None,
) -> None:
    """Validate the arguments of the `multilabel_confusion_matrix` method."""
    if not isinstance(num_labels, int) or num_labels < 2:
        raise ValueError(
            "Expected argument `num_labels` to be an integer larger than 1, "
            f"but got {num_labels}.",
        )

    _binary_confusion_matrix_validate_args(
        threshold=threshold,
        normalize=normalize,
        ignore_index=ignore_index,
    )


def _multilabel_confusion_matrix_validate_arrays(
    target: Array,
    preds: Array,
    num_labels: int,
    ignore_index: Optional[int] = None,
) -> ModuleType:
    """Validate the input arrays of the `multilabel_confusion_matrix` method."""
    _basic_input_array_checks(target, preds)
    _check_same_shape(target, preds)

    xp = apc.array_namespace(target, preds)

    if preds.shape[1] != num_labels:
        raise ValueError(
            "Expected the second dimension of `preds` and `target` to be equal "
            f"to `num_labels`={num_labels}, but got {preds.shape[1]}.",
        )

    # Check that target only contains [0,1] values or value in ignore_index
    unique_values = xp.unique_values(target)
    if ignore_index is None:
        check = xp.any((unique_values != 0) & (unique_values != 1))
    else:
        check = xp.any(
            (unique_values != 0)
            & (unique_values != 1)
            & (unique_values != ignore_index),
        )
    if check:
        raise RuntimeError(
            "Expected only the following values "
            f"{[0, 1] if ignore_index is None else [ignore_index]} in `target`. "
            f"But found the following values: {unique_values}",
        )

    if not is_floating_point(preds):
        unique_values = xp.unique_values(preds)
        if xp.any((unique_values != 0) & (unique_values != 1)):
            raise RuntimeError(
                "Expected only 0s and 1s in `preds`, but found the following values: "
                f"{unique_values}",
            )

    return xp


def _multilabel_confusion_matrix_format_arrays(
    target: Array,
    preds: Array,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    *,
    xp: ModuleType,
) -> Tuple[Array, Array]:
    """Format the input arrays of the `multilabel_confusion_matrix` method."""
    if is_floating_point(preds):
        # NOTE: in the array API standard the `__mul__` operator is only defined
        # for numeric arrays (including float and int scalars) so we convert the
        # boolean array to an integer array first.
        if not xp.all(to_int((preds >= 0)) * to_int((preds <= 1))):
            preds = sigmoid(preds)  # convert logits to probabilities
        preds = to_int(preds > threshold)

    preds = xp.reshape(preds, shape=(*preds.shape[:2], -1))
    target = xp.reshape(target, shape=(*target.shape[:2], -1))

    if ignore_index is not None:
        idx = target == ignore_index
        target = clone(target)
        target[idx] = -1

    return target, preds


def _multilabel_confusion_matrix_update_state(
    target: Array,
    preds: Array,
    *,
    xp: ModuleType,
) -> Tuple[Array, Array, Array, Array]:
    """Compute the statistics for the given `target` and `preds` arrays."""
    sum_axis = (0, -1)
    tp = squeeze_all(xp.sum(to_int((target == preds) & (target == 1)), axis=sum_axis))
    fn = squeeze_all(xp.sum(to_int((target != preds) & (target == 1)), axis=sum_axis))
    fp = squeeze_all(xp.sum(to_int((target != preds) & (target == 0)), axis=sum_axis))
    tn = squeeze_all(xp.sum(to_int((target == preds) & (target == 0)), axis=sum_axis))

    return tn, fp, fn, tp


def _multilabel_confusion_matrix_compute(
    tn: Array,
    fp: Array,
    fn: Array,
    tp: Array,
    num_labels: int,
    normalize: Optional[str] = None,
) -> Array:
    """Compute the confusion matrix from the given stat scores."""
    xp = apc.array_namespace(tn, fp, fn, tp)

    confmat = squeeze_all(
        xp.reshape(xp.stack([tn, fp, fn, tp], axis=-1), shape=(-1, num_labels, 2, 2)),
    )

    return _normalize_confusion_matrix(confmat, normalize=normalize, xp=xp)


def multilabel_confusion_matrix(
    target: Array,
    preds: Array,
    num_labels: int,
    threshold: float = 0.5,
    normalize: Optional[str] = None,
    ignore_index: Optional[int] = None,
) -> Array:
    """Compute the confusion matrix for multilabel classification tasks.

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
    normalize : str, optional, default=None
        Normalization mode.
        If `None` or `'none'`, return the number of correctly classified samples
        for each class.
        If `'true'`, return the fraction of correctly classified samples for each
        class over the number of true samples for each class.
        If `'pred'`, return the fraction of samples of each class that were correctly
        classified over the number of samples predicted for each class.
        If `'all'`, return the fraction of correctly classified samples over all
        samples.
    ignore_index : int, optional, default=None
        Specifies a target value that is ignored and does not contribute to the
        confusion matrix. If `None`, ignore nothing.

    Returns
    -------
    Array
        The confusion matrix with shape `(L, 2, 2)`, where `L` is the number of labels.

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
    >>> from cyclops.evaluate.metrics.experimental.functional import (
    ...     multilabel_confusion_matrix,
    ... )
    >>> target = anp.asarray([[0, 1, 0], [1, 0, 1]])
    >>> preds = anp.asarray([[0, 0, 1], [1, 0, 1]])
    >>> multilabel_confusion_matrix(target, preds, num_labels=3)
    Array([[[1, 0],
            [0, 1]],
    <BLANKLINE>
           [[1, 0],
            [1, 0]],
    <BLANKLINE>
           [[0, 1],
            [0, 1]]], dtype=int64)
    >>> target = anp.asarray([[0, 1, 0], [1, 0, 1]])
    >>> preds = anp.asarray([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
    >>> multilabel_confusion_matrix(target, preds, num_labels=3)
    Array([[[1, 0],
            [0, 1]],
    <BLANKLINE>
           [[1, 0],
            [1, 0]],
    <BLANKLINE>
           [[0, 1],
            [0, 1]]], dtype=int64)

    """  # noqa: W505
    _multilabel_confusion_matrix_validate_args(
        num_labels,
        threshold=threshold,
        normalize=normalize,
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

    return _multilabel_confusion_matrix_compute(
        tn,
        fp,
        fn,
        tp,
        num_labels,
        normalize=normalize,
    )
