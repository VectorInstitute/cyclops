"""Functions for computing average precision (AUPRC) for classification tasks."""

import warnings
from types import ModuleType
from typing import List, Literal, Optional, Tuple, Union

import array_api_compat as apc

from cyclops.evaluate.metrics.experimental.functional.precision_recall_curve import (
    _binary_precision_recall_curve_compute,
    _binary_precision_recall_curve_format_arrays,
    _binary_precision_recall_curve_update,
    _binary_precision_recall_curve_validate_args,
    _binary_precision_recall_curve_validate_arrays,
    _multiclass_precision_recall_curve_compute,
    _multiclass_precision_recall_curve_format_arrays,
    _multiclass_precision_recall_curve_update,
    _multiclass_precision_recall_curve_validate_args,
    _multiclass_precision_recall_curve_validate_arrays,
    _multilabel_precision_recall_curve_compute,
    _multilabel_precision_recall_curve_format_arrays,
    _multilabel_precision_recall_curve_update,
    _multilabel_precision_recall_curve_validate_args,
    _multilabel_precision_recall_curve_validate_arrays,
)
from cyclops.evaluate.metrics.experimental.utils.ops import (
    _diff,
    bincount,
    flatten,
    remove_ignore_index,
    safe_divide,
)
from cyclops.evaluate.metrics.experimental.utils.types import Array


def _binary_average_precision_compute(
    state: Union[Tuple[Array, Array], Array],
    thresholds: Optional[Array],
    pos_label: int = 1,
) -> Array:
    """Compute average precision for binary classification task.

    Parameters
    ----------
    state : Array or Tuple[Array, Array]
        State from which the precision-recall curve can be computed. Can be
        either a tuple of (target, preds) or a multi-threshold confusion matrix.
    thresholds : Array, optional
        Thresholds used for computing the precision and recall scores. If not None,
        must be a 1D numpy array of floats in the [0, 1] range and monotonically
        increasing.
    pos_label : int, optional, default=1
        The label of the positive class.

    Returns
    -------
    Array
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
    xp = apc.array_namespace(precision, recall)
    return -xp.sum(_diff(recall) * precision[:-1], dtype=xp.float32)  # type: ignore


def binary_average_precision(
    target: Array,
    preds: Array,
    thresholds: Optional[Union[int, List[float], Array]] = None,
    ignore_index: Optional[int] = None,
) -> Array:
    """Compute average precision score for binary classification task.

    The average precision score summarizes a precision-recall curve as the weighted
    mean of precisions achieved at each threshold, with the increase in recall from
    the previous threshold used as the weight.

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
    thresholds : Union[int, List[float], Array], optional, default=None
        The thresholds to use for computing the average precision. Can be one
        of the following:
        - `None`: use all unique values in `preds` as thresholds.
        - `int`: use `int` (larger than 1) uniformly spaced thresholds in the range
          [0, 1].
        - `List[float]`: use the values in the list as bins for the thresholds.
        - `Array`: use the values in the Array as bins for the thresholds. The
          array must be 1D.
    ignore_index : int, optional, default=None
        The value in `target` that should be ignored when computing the average
        precision. If `None`, all values in `target` are used.

    Returns
    -------
    Array
        The average precision score.

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
    >>> import numpy.array_api as anp
    >>> from cyclops.evaluate.metrics.experimental.functional import (
    ...     binary_average_precision,
    ... )
    >>> target = anp.asarray([0, 1, 1, 0])
    >>> preds = anp.asarray([0, 0.5, 0.7, 0.8])
    >>> binary_average_precision(target, preds, thresholds=None)
    Array(0.5833334, dtype=float32)

    """
    _binary_precision_recall_curve_validate_args(thresholds, ignore_index)
    xp = _binary_precision_recall_curve_validate_arrays(
        target,
        preds,
        thresholds,
        ignore_index,
    )
    target, preds, thresholds = _binary_precision_recall_curve_format_arrays(
        target,
        preds,
        thresholds=thresholds,
        ignore_index=ignore_index,
        xp=xp,
    )
    state = _binary_precision_recall_curve_update(
        target,
        preds,
        thresholds=thresholds,
        xp=xp,
    )
    return _binary_average_precision_compute(state, thresholds, pos_label=1)


def _reduce_average_precision(
    precision: Union[Array, List[Array]],
    recall: Union[Array, List[Array]],
    average: Optional[Literal["macro", "weighted", "none"]] = "macro",
    weights: Optional[Array] = None,
    *,
    xp: ModuleType,
) -> Array:
    """Reduce the precision-recall curve to a single average precision score.

    Applies the specified `average` after computing the average precision score
    for each class/label.

    Parameters
    ----------
    precision : Array or List[Array]
        The precision values for each class/label, computed at different thresholds.
    recall : Array or List[Array]
        The recall values for each class/label, computed at different thresholds.
    average : {"macro", "weighted", "none"}, optional, default="macro"
        The type of averaging to use for computing the average precision score. Can
        be one of the following:
        - `"macro"`: computes the average precision score for each class/label and
          average over the scores.
        - `"weighted"`: computes the average of the precision score for each
          class/label and average over the classwise/labelwise scores using
          `weights` as weights.
        - `"none"`: do not average over the classwise/labelwise scores.
    weights : Array, optional, default=None
        The weights to use for computing the weighted average precision score.
    xp : ModuleType
        The array API module to use for computations.

    Returns
    -------
    Array
        The average precision score.

    Raises
    ------
    ValueError
        If `average` is not `"macro"`, `"weighted"` or `"none"` or `None` or
        average is `"weighted"` and `weights` is `None`.
    """
    if apc.is_array_api_obj(precision) and apc.is_array_api_obj(recall):
        avg_prec = -xp.sum(
            (recall[:, 1:] - recall[:, :-1]) * precision[:, :-1],  # type: ignore
            axis=1,
            dtype=xp.float32,
        )
    else:
        avg_prec = xp.stack(
            [
                -xp.sum((rec[1:] - rec[:-1]) * prec[:-1], dtype=xp.float32)
                for prec, rec in zip(precision, recall)  # type: ignore[arg-type]
            ],
        )
    if average is None or average == "none":
        return avg_prec  # type: ignore[no-any-return]
    if xp.any(xp.isnan(avg_prec)):
        warnings.warn(
            f"Average precision score for one or more classes was `nan`. Ignoring these classes in {average}-average",
            UserWarning,
            stacklevel=1,
        )
    idx = ~xp.isnan(avg_prec)
    if average == "macro":
        return xp.mean(avg_prec[idx])  # type: ignore[no-any-return]
    if average == "weighted" and weights is not None:
        weights = safe_divide(weights[idx], xp.sum(weights[idx]))
        return xp.sum(avg_prec[idx] * weights, dtype=xp.float32)  # type: ignore[no-any-return]
    raise ValueError(
        "Received an incompatible combinations of inputs to make reduction.",
    )


def _multiclass_average_precision_validate_args(
    num_classes: int,
    thresholds: Optional[Union[int, List[float], Array]] = None,
    average: Optional[Literal["macro", "weighted", "none"]] = "macro",
    ignore_index: Optional[Union[int, Tuple[int]]] = None,
) -> None:
    """Validate the arguments for the `multiclass_average_precision` function."""
    _multiclass_precision_recall_curve_validate_args(
        num_classes,
        thresholds=thresholds,
        ignore_index=ignore_index,
    )
    allowed_averages = ["macro", "weighted", "none"]
    if average is not None and average not in allowed_averages:
        raise ValueError(
            f"Expected `average` to be one of {allowed_averages}, got {average}.",
        )


def _multiclass_average_precision_compute(
    state: Union[Tuple[Array, Array], Array],
    num_classes: int,
    thresholds: Optional[Array],
    average: Optional[Literal["macro", "weighted", "none"]] = "macro",
) -> Array:
    """Compute the average precision score for multiclass classification task."""
    precision, recall, _ = _multiclass_precision_recall_curve_compute(
        state,
        num_classes,
        thresholds=thresholds,
        average=None,
    )
    xp = (
        apc.array_namespace(*state)
        if isinstance(state, tuple)
        else apc.array_namespace(state)
    )
    return _reduce_average_precision(
        precision,
        recall,
        average=average,
        weights=(
            xp.astype(bincount(state[0], minlength=num_classes), xp.float32, copy=False)
            if thresholds is None
            else xp.sum(state[0, ...][:, 1, :], axis=-1, dtype=xp.float32)  # type: ignore[call-overload]
        ),
        xp=xp,
    )


def multiclass_average_precision(
    target: Array,
    preds: Array,
    num_classes: int,
    thresholds: Optional[Union[int, List[float], Array]] = None,
    average: Optional[Literal["macro", "weighted", "none"]] = "macro",
    ignore_index: Optional[Union[int, Tuple[int]]] = None,
) -> Array:
    """Compute the average precision score for multiclass classification task.

    The average precision score summarizes a precision-recall curve as the weighted
    mean of precisions achieved at each threshold, with the increase in recall from
    the previous threshold used as the weight.

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
        The thresholds to use for computing the average precision score. Can be one
        of the following:
        - `None`: use all unique values in `preds` as thresholds.
        - `int`: use `int` (larger than 1) uniformly spaced thresholds in the range
          [0, 1].
        - `List[float]`: use the values in the list as bins for the thresholds.
        - `Array`: use the values in the Array as bins for the thresholds. The
          array must be 1D.
    average : {"macro", "weighted", "none"}, optional, default="macro"
        The type of averaging to use for computing the average precision score. Can
        be one of the following:
        - `"macro"`: compute the average precision score for each class and average
            over the classes.
        - `"weighted"`: computes the average of the precision for each class and
            average over the classwise scores using the support of each class as
            weights.
        - `"none"`: do not average over the classwise scores.
    ignore_index : int or Tuple[int], optional, default=None
        The value(s) in `target` that should be ignored when computing the average
        precision score. If `None`, all values in `target` are used.

    Returns
    -------
    Array
        The average precision score.

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
    >>> from cyclops.evaluate.metrics.experimental.functional import (
    ...     multiclass_average_precision,
    ... )
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
    >>> multiclass_average_precision(
    ...     target,
    ...     preds,
    ...     num_classes=3,
    ...     thresholds=None,
    ...     average=None,
    ... )
    Array([0.33333334, 0.5       , 0.5       ], dtype=float32)
    >>> multiclass_average_precision(
    ...     target,
    ...     preds,
    ...     num_classes=3,
    ...     thresholds=None,
    ...     average="macro",
    ... )
    Array(0.44444445, dtype=float32)
    >>> multiclass_average_precision(
    ...     target,
    ...     preds,
    ...     num_classes=3,
    ...     thresholds=None,
    ...     average="weighted",
    ... )
    Array(0.44444448, dtype=float32)
    """
    _multiclass_average_precision_validate_args(
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
    return _multiclass_average_precision_compute(
        state,
        num_classes,
        thresholds=thresholds,
        average=average,
    )


def _multilabel_average_precision_validate_args(
    num_labels: int,
    thresholds: Optional[Union[int, List[float], Array]] = None,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    ignore_index: Optional[int] = None,
) -> None:
    """Validate the arguments for the `multilabel_average_precision` function."""
    _multilabel_precision_recall_curve_validate_args(
        num_labels,
        thresholds=thresholds,
        ignore_index=ignore_index,
    )
    allowed_averages = ["micro", "macro", "weighted", "none"]
    if average is not None and average not in allowed_averages:
        raise ValueError(
            f"Expected `average` to be one of {allowed_averages}, got {average}.",
        )


def _multilabel_average_precision_compute(
    state: Union[Tuple[Array, Array], Array],
    num_labels: int,
    thresholds: Optional[Array],
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    ignore_index: Optional[int] = None,
) -> Array:
    """Compute the average precision score for multilabel classification task."""
    xp = (
        apc.array_namespace(*state)
        if isinstance(state, tuple)
        else apc.array_namespace(state)
    )
    if average == "micro":
        if apc.is_array_api_obj(state) and thresholds is not None:
            state = xp.sum(state, axis=1)
        else:
            target, preds = flatten(state[0]), flatten(state[1])
            target, preds = remove_ignore_index(target, preds, ignore_index)
            state = (target, preds)
        return _binary_average_precision_compute(state, thresholds)

    precision, recall, _ = _multilabel_precision_recall_curve_compute(
        state,
        num_labels,
        thresholds=thresholds,
        ignore_index=ignore_index,
    )
    return _reduce_average_precision(
        precision,
        recall,
        average=average,
        weights=(
            xp.sum(
                xp.astype(state[0] == 1, xp.int32, copy=False), axis=0, dtype=xp.float32
            )
            if thresholds is None
            else xp.sum(state[0, ...][:, 1, :], axis=-1)  # type: ignore[call-overload]
        ),
        xp=xp,
    )


def multilabel_average_precision(
    target: Array,
    preds: Array,
    num_labels: int,
    thresholds: Optional[Union[int, List[float], Array]] = None,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    ignore_index: Optional[int] = None,
) -> Array:
    """Compute the average precision score for multilabel classification task.

    The average precision score summarizes a precision-recall curve as the weighted
    mean of precisions achieved at each threshold, with the increase in recall from
    the previous threshold used as the weight.

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
        The thresholds to use for computing the average precision score. Can be one
        of the following:
        - `None`: use all unique values in `preds` as thresholds.
        - `int`: use `int` (larger than 1) uniformly spaced thresholds in the range
          [0, 1].
        - `List[float]`: use the values in the list as bins for the thresholds.
        - `Array`: use the values in the Array as bins for the thresholds. The
          array must be 1D.
    average : {"micro", "macro", "weighted", "none"}, optional, default="macro"
        The type of averaging to use for computing the average precision score. Can
        be one of the following:
        - `"micro"`: computes the average precision score globally by summing over
            the average precision scores for each label.
        - `"macro"`: compute the average precision score for each label and average
            over the labels.
        - `"weighted"`: computes the average of the precision for each label and
            average over the labelwise scores using the support of each label as
            weights.
        - `"none"`: do not average over the labelwise scores.
    ignore_index : int, optional, default=None
        The value in `target` that should be ignored when computing the average
        precision score. If `None`, all values in `target` are used.

    Returns
    -------
    Array
        The average precision score.

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
        If `average` is not `"micro"`, `"macro"`, `"weighted"`, `"none"` or `None`.
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
    >>> from cyclops.evaluate.metrics.experimental.functional import (
    ...     multilabel_average_precision,
    ... )
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([[0, 1, 0], [1, 1, 0], [0, 0, 1]])
    >>> preds = anp.asarray(
    ...     [[0.11, 0.22, 0.67], [0.84, 0.73, 0.12], [0.33, 0.92, 0.44]],
    ... )
    >>> multilabel_average_precision(
    ...     target,
    ...     preds,
    ...     num_labels=3,
    ...     thresholds=None,
    ...     average=None,
    ... )
    Array([1.       , 0.5833334, 0.5      ], dtype=float32)
    >>> multilabel_average_precision(
    ...     target,
    ...     preds,
    ...     num_labels=3,
    ...     thresholds=None,
    ...     average="micro",
    ... )
    Array(0.58452386, dtype=float32)
    >>> multilabel_average_precision(
    ...     target,
    ...     preds,
    ...     num_labels=3,
    ...     thresholds=None,
    ...     average="macro",
    ... )
    Array(0.6944445, dtype=float32)
    >>> multilabel_average_precision(
    ...     target,
    ...     preds,
    ...     num_labels=3,
    ...     thresholds=None,
    ...     average="weighted",
    ... )
    Array(0.6666667, dtype=float32)
    """
    _multilabel_average_precision_validate_args(
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
    return _multilabel_average_precision_compute(
        state,
        num_labels,
        thresholds=thresholds,
        average=average,
        ignore_index=ignore_index,
    )
