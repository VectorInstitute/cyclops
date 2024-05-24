"""Functions for computing the area under the ROC curve (AUROC)."""

import warnings
from types import ModuleType
from typing import List, Literal, Optional, Tuple, Union

import array_api_compat as apc

from cyclops.evaluate.metrics.experimental.functional.precision_recall_curve import (
    _binary_precision_recall_curve_format_arrays,
    _binary_precision_recall_curve_update,
    _binary_precision_recall_curve_validate_args,
    _binary_precision_recall_curve_validate_arrays,
    _multiclass_precision_recall_curve_format_arrays,
    _multiclass_precision_recall_curve_update,
    _multiclass_precision_recall_curve_validate_args,
    _multiclass_precision_recall_curve_validate_arrays,
    _multilabel_precision_recall_curve_format_arrays,
    _multilabel_precision_recall_curve_update,
    _multilabel_precision_recall_curve_validate_args,
    _multilabel_precision_recall_curve_validate_arrays,
)
from cyclops.evaluate.metrics.experimental.functional.roc import (
    _binary_roc_compute,
    _multiclass_roc_compute,
    _multilabel_roc_compute,
)
from cyclops.evaluate.metrics.experimental.utils.ops import (
    _auc_compute,
    _interp,
    _searchsorted,
    bincount,
    flatten,
    remove_ignore_index,
    safe_divide,
)
from cyclops.evaluate.metrics.experimental.utils.types import Array


def _binary_auroc_validate_args(
    max_fpr: Optional[float] = None,
    thresholds: Optional[Union[int, List[float], Array]] = None,
    ignore_index: Optional[int] = None,
) -> None:
    """Validate arguments for binary AUROC computation."""
    _binary_precision_recall_curve_validate_args(thresholds, ignore_index)
    if max_fpr is not None and not isinstance(max_fpr, float) and 0 < max_fpr <= 1:
        raise ValueError(
            f"Argument `max_fpr` should be a float in range (0, 1], but got: {max_fpr}",
        )


def _binary_auroc_compute(
    state: Union[Array, Tuple[Array, Array]],
    thresholds: Optional[Array],
    max_fpr: Optional[float] = None,
    pos_label: int = 1,
) -> Array:
    """Compute the area under the ROC curve for binary classification tasks."""
    fpr, tpr, _ = _binary_roc_compute(state, thresholds, pos_label)
    xp = (
        apc.array_namespace(*state)
        if isinstance(state, tuple)
        else apc.array_namespace(state)
    )
    if max_fpr is None or max_fpr == 1 or xp.sum(fpr) == 0 or xp.sum(tpr) == 0:
        return _auc_compute(fpr, tpr, 1.0)

    _device = apc.device(fpr) if apc.is_array_api_obj(fpr) else apc.device(fpr[0])
    max_area = xp.asarray(max_fpr, dtype=xp.float32, device=_device)

    # Add a single point at max_fpr and interpolate its tpr value
    stop = _searchsorted(fpr, max_area, side="right")
    x_interp = xp.asarray([fpr[stop - 1], fpr[stop]], dtype=xp.float32, device=_device)
    y_interp = xp.asarray([tpr[stop - 1], tpr[stop]], dtype=xp.float32, device=_device)
    interp_tpr = _interp(max_area, x_interp, y_interp)
    tpr = xp.concat([tpr[:stop], xp.reshape(interp_tpr, (1,))])
    fpr = xp.concat([fpr[:stop], xp.reshape(max_area, (1,))])

    # Compute partial AUC
    partial_auc = _auc_compute(fpr, tpr, 1.0)

    # McClish correction: standardize result to be 0.5 if non-discriminant and 1
    # if maximal
    min_area = 0.5 * max_area**2
    return 0.5 * (1 + (partial_auc - min_area) / (max_area - min_area))  # type: ignore[no-any-return]


def binary_auroc(
    target: Array,
    preds: Array,
    max_fpr: Optional[float] = None,
    thresholds: Optional[Union[int, List[float], Array]] = None,
    ignore_index: Optional[int] = None,
) -> Array:
    """Compute the area under the ROC curve for binary classification tasks.

    Parameters
    ----------
    target : Array
        An array object that is compatible with the Python array API standard
        and contains the ground truth labels in the range [0, 1]. The expected
        shape of the array is `(N, ...)`, where `N` is the number of samples.
    preds : Array
        An array object that is compatible with the Python array API standard and
        contains the probability/logit scores for the positive class. The expected
        shape of the array is `(N, ...)` where `N` is the number of samples. If
        `preds` contains floating point values that are not in the range `[0, 1]`,
        a sigmoid function will be applied to each value before thresholding.
    max_fpr : float, optional, default=None
        If not `None`, computes the maximum area under the curve up to the given
        false positive rate value. Must be a float in the range (0, 1].
    thresholds : Union[int, List[float], Array], optional, default=None
        The thresholds to use for computing the ROC curve. Can be one of the following:
        - `None`: use all unique values in `preds` as thresholds.
        - `int`: use `int` (larger than 1) uniformly spaced thresholds in the range
          [0, 1].
        - `List[float]`: use the values in the list as bins for the thresholds.
        - `Array`: use the values in the Array as bins for the thresholds. The
          array must be 1D.
    ignore_index : int, optional, default=None
        The value in `target` that should be ignored when computing the AUROC.
        If `None`, all values in `target` are used.

    Returns
    -------
    Array
        The area under the ROC curve.

    Raises
    ------
    TypeError
        If `thresholds` is not `None` and not an integer, a list of floats or an
        Array of floats.
    ValueError
        If `thresholds` is an integer and smaller than 2.
    ValueError
        If `thresholds` is a list of floats with values outside the range [0, 1]
        and not monotonically increasing.
    ValueError
        If `thresholds` is an Array of floats and not all values are in the range
        [0, 1] or the array is not one-dimensional.
    ValueError
        If `ignore_index` is not `None` or an integer.
    ValueError
        If `max_fpr` is not `None` and not a float in the range (0, 1].
    TypeError
        If `target` and `preds` are not compatible with the Python array API standard.
    ValueError
        If `target` and `preds` are empty.
    ValueError
        If `target` and `preds` are not numeric arrays.
    ValueError
        If `target` and `preds` do not have the same shape.
    ValueError
        If `target` contains floating point values.
    ValueError
        If `preds` contains non-floating point values.
    RuntimeError
        If `target` contains values outside the range [0, 1] or does not contain
        `ignore_index` if `ignore_index` is not `None`.
    ValueError
        If the array API namespace of `target` and `preds` are not the same as the
        array API namespace of `thresholds`.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental.functional import binary_auroc
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([0, 1, 0, 1, 0, 1])
    >>> preds = anp.asarray([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])
    >>> binary_auroc(target, preds, thresholds=None)
    Array(0.6666667, dtype=float32)
    >>> binary_auroc(target, preds, thresholds=5)
    Array(0.5555556, dtype=float32)
    >>> binary_auroc(target, preds, max_fpr=0.2)
    Array(0.6296296, dtype=float32)

    """
    _binary_auroc_validate_args(max_fpr, thresholds, ignore_index)
    xp = _binary_precision_recall_curve_validate_arrays(
        target,
        preds,
        thresholds=thresholds,
        ignore_index=ignore_index,
    )
    target, preds, thresholds = _binary_precision_recall_curve_format_arrays(
        target,
        preds,
        thresholds=thresholds,
        ignore_index=ignore_index,
        xp=xp,
    )
    state = _binary_precision_recall_curve_update(target, preds, thresholds, xp=xp)
    return _binary_auroc_compute(state, thresholds=thresholds, max_fpr=max_fpr)


def _reduce_auroc(
    fpr: Union[Array, List[Array]],
    tpr: Union[Array, List[Array]],
    average: Optional[Literal["macro", "weighted", "none"]] = None,
    weights: Optional[Array] = None,
    *,
    xp: ModuleType,
) -> Array:
    """Compute the area under the ROC curve and apply `average` method.

    Parameters
    ----------
    fpr : Array or list of Array
        False positive rate.
    tpr : Array or list of Array
        True positive rate.
    average : {"macro", "weighted", "none"}, default=None
        If not None, apply the method to compute the average area under the ROC curve.
    weights : Array, optional, default=None
        Sample weights.

    Returns
    -------
    Array
        Area under the ROC curve.

    Raises
    ------
    ValueError
        If ``average`` is not one of ``macro`` or ``weighted`` or if
        ``average`` is ``weighted`` and ``weights`` is None.

    Warns
    -----
    UserWarning
        If the AUROC for one or more classes is `nan` and ``average`` is not ``none``.

    """
    if apc.is_array_api_obj(fpr) and apc.is_array_api_obj(tpr):
        res = _auc_compute(fpr, tpr, 1.0, axis=1)  # type: ignore
    else:
        res = xp.stack(
            [_auc_compute(x, y, 1.0) for x, y in zip(fpr, tpr)],  # type: ignore
        )
    if average is None or average == "none":
        return res

    if xp.any(xp.isnan(res)):
        warnings.warn(
            "The AUROC for one or more classes was `nan`. Ignoring these classes "
            f"in {average}-average",
            UserWarning,
            stacklevel=1,
        )
    idx = ~xp.isnan(res)
    if average == "macro":
        return xp.mean(res[idx])  # type: ignore[no-any-return]
    if average == "weighted" and weights is not None:
        weights = safe_divide(weights[idx], xp.sum(weights[idx]))
        return xp.sum((res[idx] * weights))  # type: ignore[no-any-return]
    raise ValueError(
        "Received an incompatible combinations of inputs to make reduction.",
    )


def _multiclass_auroc_validate_args(
    num_classes: int,
    thresholds: Optional[Union[int, List[float], Array]] = None,
    average: Optional[Literal["macro", "weighted", "none"]] = "macro",
    ignore_index: Optional[Union[int, Tuple[int]]] = None,
) -> None:
    """Validate arguments for multiclass AUROC computation."""
    _multiclass_precision_recall_curve_validate_args(
        num_classes,
        thresholds=thresholds,
        ignore_index=ignore_index,
    )
    allowed_average = ("macro", "weighted", "none", None)
    if average not in allowed_average:
        raise ValueError(
            f"Expected argument `average` to be one of {allowed_average} but got {average}",
        )


def _multiclass_auroc_compute(
    state: Union[Array, Tuple[Array, Array]],
    num_classes: int,
    thresholds: Optional[Array] = None,
    average: Optional[Literal["macro", "weighted", "none"]] = "macro",
) -> Array:
    """Compute the area under the ROC curve for multiclass classification tasks."""
    fpr, tpr, _ = _multiclass_roc_compute(state, num_classes, thresholds=thresholds)
    xp = (
        apc.array_namespace(*state)
        if isinstance(state, tuple)
        else apc.array_namespace(state)
    )
    return _reduce_auroc(
        fpr,
        tpr,
        average=average,
        weights=(
            xp.astype(bincount(state[0], minlength=num_classes), xp.float32, copy=False)
            if thresholds is None
            else xp.sum(state[0, ...][:, 1, :], axis=-1)  # type: ignore[call-overload]
        ),
        xp=xp,
    )


def multiclass_auroc(
    target: Array,
    preds: Array,
    num_classes: int,
    thresholds: Optional[Union[int, List[float], Array]] = None,
    average: Optional[Literal["macro", "weighted", "none"]] = "macro",
    ignore_index: Optional[Union[int, Tuple[int]]] = None,
) -> Array:
    """Compute the area under the ROC curve for multiclass classification tasks.

    Parameters
    ----------
    target : Array
        An array object that is compatible with the Python array API standard
        and contains the ground truth labels in the range [0, `num_classes`]
        (except if `ignore_index` is specified). The expected shape of the array
        is `(N, ...)`, where `N` is the number of samples.
    preds : Array
        An array object that is compatible with the Python array API standard and
        contains the probability/logit scores for each sample. The expected shape
        of the array is `(N, C, ...)` where `N` is the number of samples and `C`
        is the number of classes. If `preds` contains floating point values that
        are not in the range `[0, 1]`, a softmax function will be applied to each
        value before thresholding.
    num_classes : int
        The number of classes in the classification problem.
    thresholds : Union[int, List[float], Array], optional, default=None
        The thresholds to use for computing the ROC curve. Can be one of the following:
        - `None`: use all unique values in `preds` as thresholds.
        - `int`: use `int` (larger than 1) uniformly spaced thresholds in the range
          [0, 1].
        - `List[float]`: use the values in the list as bins for the thresholds.
        - `Array`: use the values in the Array as bins for the thresholds. The
          array must be 1D.
    average : {"macro", "weighted", "none"}, optional, default="macro"
        The type of averaging to use for computing the AUROC. Can be one of
        the following:
        - `"macro"`: interpolates the curves from each class at a combined set of
          thresholds and then average over the classwise interpolated curves.
        - `"weighted"`: average over the classwise curves weighted by the support
          (the number of true instances for each class).
        - `"none"`: do not average over the classwise curves.
    ignore_index : int or Tuple[int], optional, default=None
        The value(s) in `target` that should be ignored when computing the AUROC.
        If `None`, all values in `target` are used.

    Returns
    -------
    Array
        The area under the ROC curve.

    Raises
    ------
    TypeError
        If `thresholds` is not `None` and not an integer, a list of floats or an
        Array of floats.
    ValueError
        If `thresholds` is an integer and smaller than 2.
    ValueError
        If `thresholds` is a list of floats with values outside the range [0, 1]
        and not monotonically increasing.
    ValueError
        If `thresholds` is an Array of floats and not all values are in the range
        [0, 1] or the array is not one-dimensional.
    ValueError
        If `num_classes` is not an integer larger than 1.
    ValueError
        If `ignore_index` is not `None`, an integer or a tuple of integers.
    ValueError
        If `average` is not `"macro"`, `"weighted"`, `"none"` or `None`.
    TypeError
        If `target` and `preds` are not compatible with the Python array API standard.
    ValueError
        If `target` and `preds` are empty.
    ValueError
        If `target` and `preds` are not numeric arrays.
    ValueError
        If `preds` does not have one more dimension than `target`.
    ValueError
        If `target` contains floating point values.
    ValueError
        If `preds` contains non-floating point values.
    ValueError
        If the second dimension of `preds` is not equal to `num_classes`.
    ValueError
        If the first dimension of `preds` is not equal to the first dimension of
        `target` or the third dimension of `preds` is not equal to the second
        dimension of `target`.
    RuntimeError
        If `target` contains more unique values than `num_classes` or `num_classes`
        plus the number of values in `ignore_index` if `ignore_index` is not `None`.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental.functional import multiclass_auroc
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([0, 1, 2, 0, 1, 2])
    >>> preds = anp.asarray(
    ...     [
    ...         [0.11, 0.22, 0.67],
    ...         [0.84, 0.73, 0.12],
    ...         [0.33, 0.92, 0.44],
    ...         [0.11, 0.22, 0.67],
    ...         [0.84, 0.73, 0.12],
    ...         [0.33, 0.92, 0.44],
    ...     ]
    ... )
    >>> multiclass_auroc(target, preds, num_classes=3, thresholds=None)
    Array(0.33333334, dtype=float32)
    >>> multiclass_auroc(target, preds, num_classes=3, thresholds=5)
    Array(0.33333334, dtype=float32)
    >>> multiclass_auroc(target, preds, num_classes=3, average=None)
    Array([0. , 0.5, 0.5], dtype=float32)

    """
    _multiclass_auroc_validate_args(
        num_classes,
        thresholds=thresholds,
        average=average,
        ignore_index=ignore_index,
    )
    xp = _multiclass_precision_recall_curve_validate_arrays(
        target,
        preds,
        num_classes,
        ignore_index=ignore_index,
    )
    target, preds, thresholds = _multiclass_precision_recall_curve_format_arrays(
        target,
        preds,
        num_classes,
        thresholds=thresholds,
        ignore_index=ignore_index,
        xp=xp,
    )
    state = _multiclass_precision_recall_curve_update(
        target,
        preds,
        num_classes,
        thresholds=thresholds,
        xp=xp,
    )
    return _multiclass_auroc_compute(
        state,
        num_classes,
        thresholds=thresholds,
        average=average,
    )


def _multilabel_auroc_validate_args(
    num_labels: int,
    thresholds: Optional[Union[int, List[float], Array]] = None,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    ignore_index: Optional[int] = None,
) -> None:
    """Validate arguments for multilabel AUROC computation."""
    _multilabel_precision_recall_curve_validate_args(
        num_labels,
        thresholds=thresholds,
        ignore_index=ignore_index,
    )
    allowed_average = ("micro", "macro", "weighted", "none", None)
    if average not in allowed_average:
        raise ValueError(
            f"Expected argument `average` to be one of {allowed_average} but got {average}",
        )


def _multilabel_auroc_compute(
    state: Union[Array, Tuple[Array, Array]],
    num_labels: int,
    thresholds: Optional[Array],
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    ignore_index: Optional[int] = None,
) -> Array:
    """Compute the area under the ROC curve for multilabel classification tasks."""
    if average == "micro":
        if apc.is_array_api_obj(state) and thresholds is not None:
            xp = apc.array_namespace(state)
            return _binary_auroc_compute(
                xp.sum(state, axis=1),
                thresholds,
                max_fpr=None,
            )

        target = flatten(state[0])
        preds = flatten(state[1])
        if ignore_index is not None:
            target, preds = remove_ignore_index(target, preds, ignore_index)
        return _binary_auroc_compute((target, preds), thresholds, max_fpr=None)

    fpr, tpr, _ = _multilabel_roc_compute(state, num_labels, thresholds, ignore_index)
    xp = (
        apc.array_namespace(*state)
        if isinstance(state, tuple)
        else apc.array_namespace(state)
    )
    return _reduce_auroc(
        fpr,
        tpr,
        average,
        weights=(
            xp.astype(
                xp.sum(xp.astype(state[0] == 1, xp.int32, copy=False), axis=0),
                xp.float32,
                copy=False,
            )
            if thresholds is None
            else xp.sum(state[0, ...][:, 1, :], axis=-1)  # type: ignore[call-overload]
        ),
        xp=xp,
    )


def multilabel_auroc(
    target: Array,
    preds: Array,
    num_labels: int,
    thresholds: Optional[Union[int, List[float], Array]] = None,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    ignore_index: Optional[int] = None,
) -> Array:
    """Compute the area under the ROC curve for multilabel classification tasks.

    Parameters
    ----------
    target : Array
        The target array of shape `(N, L, ...)` containing the ground truth labels
        in the range [0, 1], where `N` is the number of samples and `L` is the
        number of labels.
    preds : Array
        The prediction array of shape `(N, L, ...)` containing the probability/logit
        scores for each sample, where `N` is the number of samples and `L` is the
        number of labels. If `preds` contains floating point values that are not
        in the range [0,1], they will be converted to probabilities using the
        sigmoid function.
    num_labels : int
        The number of labels in the multilabel classification problem.
    thresholds : Union[int, List[float], Array], optional, default=None
        The thresholds to use for computing the ROC curve. Can be one of the following:
        - `None`: use all unique values in `preds` as thresholds.
        - `int`: use `int` (larger than 1) uniformly spaced thresholds in the range
          [0, 1].
        - `List[float]`: use the values in the list as bins for the thresholds.
        - `Array`: use the values in the Array as bins for the thresholds. The
          array must be 1D.
    average : {"micro", "macro", "weighted", "none"}, optional, default="macro"
        The type of averaging to use for computing the AUROC. Can be one of
        the following:
        - `"micro"`: compute the AUROC globally by considering each element of the
            label indicator matrix as a label.
        - `"macro"`: compute the AUROC for each label and average them.
        - `"weighted"`: compute the AUROC for each label and average them weighted
            by the support (the number of true instances for each label).
        - `"none"`: do not average over the labelwise AUROC.
    ignore_index : int, optional, default=None
        The value in `target` that should be ignored when computing the AUROC.
        If `None`, all values in `target` are used.

    Returns
    -------
    Array
        The area under the ROC curve.

    Raises
    ------
    TypeError
        If `thresholds` is not `None` and not an integer, a list of floats or an
        Array of floats.
    ValueError
        If `thresholds` is an integer and smaller than 2.
    ValueError
        If `thresholds` is a list of floats with values outside the range [0, 1]
        and not monotonically increasing.
    ValueError
        If `thresholds` is an Array of floats and not all values are in the range
        [0, 1] or the array is not one-dimensional.
    ValueError
        If `ignore_index` is not `None` or an integer.
    ValueError
        If `num_labels` is not an integer larger than 1.
    ValueError
        If `average` is not `"micro"`, `"macro"`, `"none"` or `None`.
    TypeError
        If `target` and `preds` are not compatible with the Python array API standard.
    ValueError
        If `target` and `preds` are empty.
    ValueError
        If `target` and `preds` are not numeric arrays.
    ValueError
        If `target` and `preds` do not have the same shape.
    ValueError
        If `target` contains floating point values.
    ValueError
        If `preds` contains non-floating point values.
    RuntimeError
        If `target` contains values outside the range [0, 1] or does not contain
        `ignore_index` if `ignore_index` is not `None`.
    ValueError
        If the array API namespace of `target` and `preds` are not the same as the
        array API namespace of `thresholds`.
    ValueError
        If the second dimension of `preds` is not equal to `num_labels`.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental.functional import multilabel_auroc
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([[0, 1, 0], [1, 1, 0], [0, 0, 1]])
    >>> preds = anp.asarray(
    ...     [[0.11, 0.22, 0.67], [0.84, 0.73, 0.12], [0.33, 0.92, 0.44]],
    ... )
    >>> multilabel_auroc(target, preds, num_labels=3, thresholds=None)
    Array(0.5, dtype=float32)
    >>> multilabel_auroc(target, preds, num_labels=3, thresholds=5)
    Array(0.5, dtype=float32)
    >>> multilabel_auroc(target, preds, num_labels=3, average=None)
    Array([1. , 0. , 0.5], dtype=float32)

    """
    _multilabel_auroc_validate_args(
        num_labels,
        thresholds=thresholds,
        average=average,
        ignore_index=ignore_index,
    )
    xp = _multilabel_precision_recall_curve_validate_arrays(
        target,
        preds,
        num_labels,
        thresholds=thresholds,
        ignore_index=ignore_index,
    )
    target, preds, thresholds = _multilabel_precision_recall_curve_format_arrays(
        target,
        preds,
        num_labels,
        thresholds=thresholds,
        ignore_index=ignore_index,
        xp=xp,
    )
    state = _multilabel_precision_recall_curve_update(
        target,
        preds,
        num_labels,
        thresholds=thresholds,
        xp=xp,
    )
    return _multilabel_auroc_compute(
        state,
        num_labels,
        thresholds=thresholds,
        average=average,
        ignore_index=ignore_index,
    )
