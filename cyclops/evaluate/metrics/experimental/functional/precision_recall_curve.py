"""Functions for computing the precision and recall for different unique thresholds."""

from types import ModuleType
from typing import Any, List, Literal, NamedTuple, Optional, Sequence, Tuple, Union

import array_api_compat as apc
import numpy as np

from cyclops.evaluate.metrics.experimental.utils.ops import (
    _array_indexing,
    _cumsum,
    _interp,
    _to_one_hot,
    bincount,
    clone,
    flatten,
    moveaxis,
    remove_ignore_index,
    safe_divide,
    sigmoid,
    softmax,
    to_int,
)
from cyclops.evaluate.metrics.experimental.utils.types import Array
from cyclops.evaluate.metrics.experimental.utils.validation import (
    _basic_input_array_checks,
    _check_same_shape,
    is_floating_point,
)


class PRCurve(NamedTuple):
    """Named tuple with Precision-Recall curve (Precision, Recall and thresholds)."""

    precision: Union[Array, List[Array]]
    recall: Union[Array, List[Array]]
    thresholds: Union[Array, List[Array]]


def _validate_thresholds(thresholds: Optional[Union[int, List[float], Array]]) -> None:
    """Validate the `thresholds` argument."""
    if thresholds is not None and not (
        isinstance(thresholds, (int, list)) or apc.is_array_api_obj(thresholds)
    ):
        raise TypeError(
            "Expected argument `thresholds` to either be an integer, a list of floats or "
            f"an Array of floats, but got {thresholds}",
        )
    if isinstance(thresholds, int) and thresholds < 2:
        raise ValueError(
            "Expected argument `thresholds` to be an integer greater than 1, "
            f"but got {thresholds}",
        )
    if isinstance(thresholds, list):
        if not all(isinstance(t, float) and 0 <= t <= 1 for t in thresholds):
            raise ValueError(
                "Expected argument `thresholds` to be a list of floats in the [0,1] range, "
                f"but got {thresholds}",
            )
        if not all(np.diff(thresholds) > 0):
            raise ValueError(
                "Expected argument `thresholds` to be monotonically increasing,"
                f" but got {thresholds}",
            )

    if apc.is_array_api_obj(thresholds):
        xp = apc.array_namespace(thresholds)
        if not xp.all((thresholds >= 0) & (thresholds <= 1)):  # type: ignore
            raise ValueError(
                "Expected argument `thresholds` to be an Array of floats in the [0,1] "
                f"range, but got {thresholds}",
            )
        if not thresholds.ndim == 1:  # type: ignore
            raise ValueError(
                "Expected argument `thresholds` to be a 1D Array, but got an Array with "
                f"{thresholds.ndim} dimensions",  # type: ignore
            )


def _binary_precision_recall_curve_validate_args(
    thresholds: Optional[Union[int, List[float], Array]],
    ignore_index: Optional[int],
) -> None:
    """Validate the arguments for the `binary_precision_recall_curve` function."""
    _validate_thresholds(thresholds)
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(
            "Expected argument `ignore_index` to either be `None` or an integer, "
            f"but got {ignore_index}",
        )


def _binary_precision_recall_curve_validate_arrays(
    target: Array,
    preds: Array,
    thresholds: Optional[Union[int, List[float], Array]],
    ignore_index: Optional[int],
) -> ModuleType:
    """Validate the arrays for the `binary_precision_recall_curve` function."""
    _basic_input_array_checks(target, preds)
    _check_same_shape(target, preds)

    if is_floating_point(target):
        raise ValueError(
            "Expected argument `target` to be an Array of integers representing "
            f"binary groundtruth labels, but got tensor with dtype {target.dtype}",
        )

    if not is_floating_point(preds):
        raise ValueError(
            "Expected argument `preds` to be an floating tensor with probability/logit scores,"
            f" but got tensor with dtype {preds.dtype}",
        )

    xp: ModuleType = apc.array_namespace(target, preds)
    # check that target only contains {0,1} values or value in ignore_index
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

    if apc.is_array_api_obj(thresholds) and xp != apc.array_namespace(thresholds):
        raise ValueError(
            "Expected the array API namespace of `target` and `preds` to be the same as "
            f"the array API namespace of `thresholds`, but got {xp} and "
            f"{apc.array_namespace(thresholds)}",
        )

    return xp


def _format_thresholds(
    thresholds: Optional[Union[int, List[float], Array]] = None,
    device: Optional[Any] = None,
    *,
    xp: ModuleType,
) -> Optional[Array]:
    """Convert the `thresholds` argument to an Array."""
    if isinstance(thresholds, int):
        return xp.linspace(  # type: ignore[no-any-return]
            0,
            1,
            thresholds,
            dtype=xp.float32,
            device=device,
        )
    if isinstance(thresholds, list):
        return xp.asarray(  # type: ignore[no-any-return]
            thresholds,
            dtype=xp.float32,
            device=device,
        )
    return thresholds


def _binary_precision_recall_curve_format_arrays(
    target: Array,
    preds: Array,
    thresholds: Optional[Union[int, List[float], Array]] = None,
    ignore_index: Optional[int] = None,
    *,
    xp: ModuleType,
) -> Tuple[Array, Array, Optional[Array]]:
    """Format the arrays for the `binary_precision_recall_curve` function."""
    preds = flatten(preds)
    target = flatten(target)

    if ignore_index is not None:
        target, preds = remove_ignore_index(target, preds, ignore_index=ignore_index)

    if not xp.all(to_int((preds >= 0)) * to_int((preds <= 1))):  # preds are logits
        preds = sigmoid(preds)

    thresholds = _format_thresholds(thresholds, device=apc.device(preds), xp=xp)
    return target, preds, thresholds


def _binary_precision_recall_curve_update(
    target: Array,
    preds: Array,
    thresholds: Optional[Array],
    *,
    xp: ModuleType,
) -> Union[Array, Tuple[Array, Array]]:
    """Update the state for the `binary_precision_recall_curve` function."""
    if thresholds is None:
        return target, preds

    len_t = int(apc.size(thresholds) or 0)
    target = target == 1
    confmat = xp.empty((len_t, 2, 2), dtype=xp.int32, device=apc.device(preds))

    for i in range(len_t):
        preds_t = preds >= thresholds[i]
        confmat[i, 1, 1] = xp.sum(to_int(target & preds_t))
        confmat[i, 0, 1] = xp.sum(to_int(((~target) & preds_t)))
        confmat[i, 1, 0] = xp.sum(to_int((target & (~preds_t))))
    confmat[:, 0, 0] = (
        preds_t.shape[0] - confmat[:, 0, 1] - confmat[:, 1, 0] - confmat[:, 1, 1]
    )
    return confmat  # type: ignore[no-any-return]


def _binary_clf_curve(
    target: Array,
    preds: Array,
    sample_weights: Optional[Union[Sequence[float], Array]] = None,
    pos_label: int = 1,
) -> Tuple[Array, Array, Array]:
    """Compute the TPs and FPs for all unique thresholds in the `preds` Array.

    Adapted from
    https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/classification/precision_recall_curve.py#L28.
    """
    xp = apc.array_namespace(target, preds)
    if sample_weights is not None and not apc.is_array_api_obj(sample_weights):
        sample_weights = xp.asarray(
            sample_weights,
            device=apc.device(preds),
            dtype=xp.float32,
        )

    # remove class dimension if necessary
    if preds.ndim > target.ndim:
        preds = preds[:, 0]

    # sort preds in descending order
    sort_index = xp.argsort(preds, descending=True)
    preds = _array_indexing(preds, sort_index)
    target = _array_indexing(target, sort_index)
    weight = (
        _array_indexing(sample_weights, sort_index)  # type: ignore[arg-type]
        if sample_weights is not None
        else xp.asarray(1, device=apc.device(preds), dtype=xp.float32)
    )

    # extract indices of distinct values in preds to avoid ties
    distinct_value_indices = (
        xp.nonzero(preds[1:] - preds[:-1])[0]
        if int(apc.size(preds) or 0) > 1
        else xp.empty(0, dtype=xp.int32, device=apc.device(preds))
    )

    # concatenate a value for the end of the curve
    threshold_idxs = xp.concat(
        [
            distinct_value_indices,
            xp.asarray(
                [int(apc.size(target) or 0) - 1],
                dtype=xp.int32,
                device=apc.device(preds),
            ),
        ],
    )

    target = xp.astype(target == pos_label, xp.float32, copy=False)
    tps = _array_indexing(_cumsum(target * weight, axis=0), threshold_idxs)
    if sample_weights is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        fps = _array_indexing(_cumsum((1 - target) * weight, axis=0), threshold_idxs)
    else:
        fps = 1 + xp.astype(threshold_idxs, xp.float32, copy=False) - tps

    return fps, tps, _array_indexing(preds, threshold_idxs)


def _binary_precision_recall_curve_compute(
    state: Union[Array, Tuple[Array, Array]],
    thresholds: Optional[Array],
    pos_label: int = 1,
) -> Tuple[Array, Array, Array]:
    """Compute the precision and recall for all unique thresholds."""
    if apc.is_array_api_obj(state) and thresholds is not None:
        xp = apc.array_namespace(state, thresholds)
        tps = state[:, 1, 1]  # type: ignore[call-overload]
        fps = state[:, 0, 1]  # type: ignore[call-overload]
        fns = state[:, 1, 0]  # type: ignore[call-overload]
        precision = safe_divide(tps, tps + fps)
        recall = safe_divide(tps, tps + fns)
        precision = xp.concat(
            [
                precision,
                xp.ones(1, dtype=precision.dtype, device=apc.device(precision)),
            ],
        )
        recall = xp.concat(
            [recall, xp.zeros(1, dtype=recall.dtype, device=apc.device(recall))],
        )
        return precision, recall, thresholds

    fps, tps, thresholds = _binary_clf_curve(state[0], state[1], pos_label=pos_label)
    precision = tps / (tps + fps)
    recall = tps / tps[-1]

    xp = apc.array_namespace(precision, recall)

    # need to call reversed explicitly, since including that to slice would
    # introduce negative strides that are not yet supported in pytorch
    precision = xp.concat(
        [
            xp.flip(precision, axis=0),
            xp.ones(1, dtype=precision.dtype, device=apc.device(precision)),
        ],
    )
    recall = xp.concat(
        [
            xp.flip(recall, axis=0),
            xp.zeros(1, dtype=recall.dtype, device=apc.device(recall)),
        ],
    )
    thresholds = xp.flip(thresholds, axis=0)
    if hasattr(thresholds, "detach"):
        thresholds = clone(thresholds.detach())  # type: ignore
    return precision, recall, thresholds  # type: ignore[return-value]


def binary_precision_recall_curve(
    target: Array,
    preds: Array,
    thresholds: Optional[Union[int, List[float], Array]] = None,
    ignore_index: Optional[int] = None,
) -> Tuple[Array, Array, Array]:
    """Compute the precision and recall for all unique thresholds.

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
        The thresholds to use for computing the precision and recall. Can be one
        of the following:
        - `None`: use all unique values in `preds` as thresholds.
        - `int`: use `int` (larger than 1) uniformly spaced thresholds in the range
          [0, 1].
        - `List[float]`: use the values in the list as bins for the thresholds.
        - `Array`: use the values in the Array as bins for the thresholds. The
          array must be 1D.
    ignore_index : int, optional, default=None
        The value in `target` that should be ignored when computing the precision
        and recall. If `None`, all values in `target` are used.

    Returns
    -------
    PRCurve
        A named tuple that contains the following elements:
        - `precision` values for all unique thresholds. The shape of the array is
        `(num_thresholds + 1,)`.
        - `recall` values for all unique thresholds. The shape of the array is
        `(num_thresholds + 1,)`.
        - `thresholds` used for computing the precision and recall values, in
        ascending order. The shape of the array is `(num_thresholds,)`.

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
    >>> from cyclops.evaluate.metrics.experimental.functional import (
    ...     binary_precision_recall_curve,
    ... )
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([0, 1, 0, 1, 0, 1])
    >>> preds = anp.asarray([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])
    >>> precision, recall, thresholds = binary_precision_recall_curve(
    ...     target,
    ...     preds,
    ...     thresholds=None,
    ... )
    >>> precision
    Array([0.5      , 0.6      , 0.5      , 0.6666667,
           0.5      , 1.       , 1.       ], dtype=float32)
    >>> recall
    Array([1.        , 1.        , 0.6666667 , 0.6666667 ,
           0.33333334, 0.33333334, 0.        ], dtype=float32)
    >>> thresholds
    Array([0.11, 0.22, 0.33, 0.73, 0.84, 0.92], dtype=float64)
    >>> precision, recall, thresholds = binary_precision_recall_curve(
    ...     target,
    ...     preds,
    ...     thresholds=5,
    ... )
    >>> precision
    Array([0.5      , 0.5      , 0.6666667, 0.5      ,
           0.       , 1.       ], dtype=float32)
    >>> recall
    Array([1.        , 0.6666667 , 0.6666667 , 0.33333334,
           0.        , 0.        ], dtype=float32)
    >>> thresholds
    Array([0.  , 0.25, 0.5 , 0.75, 1.  ], dtype=float32)

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
        thresholds,
        ignore_index,
        xp=xp,
    )
    state = _binary_precision_recall_curve_update(target, preds, thresholds, xp=xp)
    return _binary_precision_recall_curve_compute(state, thresholds)


def _multiclass_precision_recall_curve_validate_args(
    num_classes: int,
    thresholds: Optional[Union[int, List[float], Array]] = None,
    average: Optional[Literal["macro", "micro", "none"]] = None,
    ignore_index: Optional[Union[int, Tuple[int]]] = None,
) -> None:
    """Validate the arguments for the `multiclass_precision_recall_curve` function."""
    _validate_thresholds(thresholds)
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError(
            "Expected argument `num_classes` to be an integer larger than 1, "
            f"but got {num_classes}.",
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
            f"or a tuple of integers but got {ignore_index}",
        )
    allowed_average = ("micro", "macro", "none", None)
    if average not in allowed_average:
        raise ValueError(
            f"Expected argument `average` to be one of {allowed_average}, "
            f"but got {average}",
        )


def _multiclass_precision_recall_curve_validate_arrays(
    target: Array,
    preds: Array,
    num_classes: int,
    ignore_index: Optional[Union[int, Tuple[int]]] = None,
) -> ModuleType:
    """Validate the arrays for the `multiclass_precision_recall_curve` function."""
    _basic_input_array_checks(target, preds)
    if not preds.ndim == target.ndim + 1:
        raise ValueError(
            f"Expected `preds` to have one more dimension than `target` but got {preds.ndim} and {target.ndim}",
        )
    if is_floating_point(target):
        raise ValueError(
            "Expected argument `target` to be an integer array, but got array "
            f"with dtype {target.dtype}",
        )
    if not is_floating_point(preds):
        raise ValueError(
            f"Expected `preds` to be an array with floating point values, but got "
            f"array with dtype {preds.dtype}",
        )
    if preds.shape[1] != num_classes:
        raise ValueError(
            f"Expected `preds.shape[1]={preds.shape[1]}` to be equal to "
            f"`num_classes={num_classes}`",
        )
    if preds.shape[0] != target.shape[0] or preds.shape[2:] != target.shape[1:]:
        raise ValueError(
            "Expected the shape of `preds` should be (N, C, ...) and the shape of "
            f"`target` should be (N, ...) but got {preds.shape} and {target.shape}",
        )

    xp = apc.array_namespace(target, preds)
    num_unique_values = xp.unique_values(target).shape[0]
    num_allowed_extra_values = 0
    if ignore_index is not None:
        num_allowed_extra_values = (
            1 if isinstance(ignore_index, int) else len(ignore_index)
        )
    check = (
        num_unique_values > num_classes
        if ignore_index is None
        else num_unique_values > num_classes + num_allowed_extra_values
    )
    if check:
        raise RuntimeError(
            f"Expected only {num_classes if ignore_index is None else num_classes + num_allowed_extra_values} "
            f"values in `target` but found {num_unique_values} values.",
        )

    return xp  # type: ignore[no-any-return]


def _multiclass_precision_recall_curve_format_arrays(
    target: Array,
    preds: Array,
    num_classes: int,
    thresholds: Optional[Union[int, List[float], Array]],
    ignore_index: Optional[Union[int, Tuple[int]]] = None,
    average: Optional[Literal["macro", "micro", "none"]] = None,
    *,
    xp: ModuleType,
) -> Tuple[Array, Array, Optional[Array]]:
    """Format the arrays for the `multiclass_precision_recall_curve` function."""
    preds = xp.reshape(moveaxis(preds, 0, 1), (num_classes, -1)).T
    target = flatten(target)

    if ignore_index is not None:
        target, preds = remove_ignore_index(target, preds, ignore_index=ignore_index)

    if not xp.all(to_int(preds >= 0) * to_int(preds <= 1)):
        preds = softmax(preds, axis=1)

    if average == "micro":
        preds = flatten(preds)
        target = flatten(_to_one_hot(target, num_classes=num_classes))

    thresholds = _format_thresholds(thresholds, device=apc.device(preds), xp=xp)
    return target, preds, thresholds


def _multiclass_precision_recall_curve_update(
    target: Array,
    preds: Array,
    num_classes: int,
    thresholds: Optional[Array],
    average: Optional[Literal["macro", "micro", "none"]] = None,
    *,
    xp: ModuleType,
) -> Union[Array, Tuple[Array, Array]]:
    """Update the state for the `multiclass_precision_recall_curve` function."""
    if thresholds is None:
        return target, preds

    if average == "micro":
        return _binary_precision_recall_curve_update(target, preds, thresholds, xp=xp)

    len_t = thresholds.shape[0] if thresholds.ndim > 0 else 1
    preds_t = to_int(
        (
            xp.expand_dims(preds, axis=-1)
            >= xp.expand_dims(xp.expand_dims(thresholds, axis=0), axis=0)
        ),
    )
    target_t = _to_one_hot(target, num_classes=num_classes)
    unique_mapping = preds_t + 2 * xp.expand_dims(to_int(target_t), axis=-1)
    unique_mapping += 4 * xp.expand_dims(
        xp.expand_dims(xp.arange(num_classes, device=apc.device(preds)), axis=0),
        axis=-1,
    )
    unique_mapping += 4 * num_classes * xp.arange(len_t, device=apc.device(preds))
    bins = bincount(flatten(unique_mapping), minlength=4 * num_classes * len_t)
    return xp.reshape(xp.astype(bins, xp.int32, copy=False), (len_t, num_classes, 2, 2))  # type: ignore[no-any-return]


def _multiclass_precision_recall_curve_compute(
    state: Union[Array, Tuple[Array, Array]],
    num_classes: int,
    thresholds: Optional[Array],
    average: Optional[Literal["macro", "micro", "none"]],
) -> Union[Tuple[Array, Array, Array], Tuple[List[Array], List[Array], List[Array]]]:
    """Compute the precision and recall for all unique thresholds."""
    if average == "micro":
        return _binary_precision_recall_curve_compute(state, thresholds)

    if apc.is_array_api_obj(state) and thresholds is not None:
        xp = apc.array_namespace(state, thresholds)
        tps = state[:, :, 1, 1]  # type: ignore[call-overload]
        fps = state[:, :, 0, 1]  # type: ignore[call-overload]
        fns = state[:, :, 1, 0]  # type: ignore[call-overload]
        precision = safe_divide(tps, tps + fps)
        recall = safe_divide(tps, tps + fns)
        precision = xp.concat(
            [
                precision,
                xp.ones(
                    (1, num_classes),
                    dtype=precision.dtype,
                    device=apc.device(precision),
                ),
            ],
        )
        recall = xp.concat(
            [
                recall,
                xp.zeros(
                    (1, num_classes),
                    dtype=recall.dtype,
                    device=apc.device(recall),
                ),
            ],
        )
        precision = precision.T
        recall = recall.T
        thres = thresholds
        array_state = True
    else:
        xp = apc.array_namespace(state[0], state[1])
        precision_list, recall_list, thres_list = [], [], []
        for i in range(num_classes):
            res = _binary_precision_recall_curve_compute(
                (state[0], state[1][:, i]),
                thresholds=None,
                pos_label=i,
            )
            precision_list.append(res[0])
            recall_list.append(res[1])
            thres_list.append(res[2])
        array_state = False

    if average == "macro":
        thres = (
            xp.concat([xp.expand_dims(thres, axis=0)] * num_classes, axis=0)  # repeat
            if array_state
            else xp.concat(xp.asarray(thres_list), 0)
        )
        thres = xp.sort(thres)
        mean_precision = (
            flatten(precision)
            if array_state
            else xp.concat(xp.asarray(precision_list), 0)
        )
        mean_precision = xp.sort(mean_precision)
        mean_recall = xp.zeros_like(mean_precision)
        for i in range(num_classes):
            mean_recall += _interp(
                mean_precision,
                precision[i] if array_state else precision_list[i],
                recall[i] if array_state else recall_list[i],
            )
        mean_recall /= num_classes
        return mean_precision, mean_recall, thres

    if array_state:
        return precision, recall, thres
    return precision_list, recall_list, thres_list


def multiclass_precision_recall_curve(
    target: Array,
    preds: Array,
    num_classes: int,
    thresholds: Optional[Union[int, List[float], Array]] = None,
    average: Optional[Literal["macro", "micro", "none"]] = None,
    ignore_index: Optional[Union[int, Tuple[int]]] = None,
) -> PRCurve:
    """Compute the precision and recall for all unique thresholds.

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
        The thresholds to use for computing the precision and recall. Can be one
        of the following:
        - `None`: use all unique values in `preds` as thresholds.
        - `int`: use `int` (larger than 1) uniformly spaced thresholds in the range
          [0, 1].
        - `List[float]`: use the values in the list as bins for the thresholds.
        - `Array`: use the values in the Array as bins for the thresholds. The
          array must be 1D.
    average : {"macro", "micro", "none"}, optional, default=None
        The type of averaging to use for computing the precision and recall. Can
        be one of the following:
        - `"macro"`: interpolates the curves from each class at a combined set of
          thresholds and then average over the classwise interpolated curves.
        - `"micro"`: one-hot encodes the targets and flattens the predictions,
          considering all classes jointly as a binary problem.
        - `"none"`: do not average over the classwise curves.
    ignore_index : int or Tuple[int], optional, default=None
        The value(s) in `target` that should be ignored when computing the
        precision and recall. If `None`, all values in `target` are used.

    Returns
    -------
    PRCurve
        A named tuple that contains the following elements:
        - `precision` values for all unique thresholds. If `thresholds` is `"none"`
        or `None`, a list for each class is returned with 1-D Arrays of shape
        `(num_thresholds + 1,)`. Otherwise, a 2-D Array of shape
        `(num_thresholds + 1, num_classes)` is returned.
        - `recall` values for all unique thresholds. If `thresholds` is `"none"`
        or `None`, a list for each class is returned with 1-D Arrays of shape
        `(num_thresholds + 1,)`. Otherwise, a 2-D Array of shape
        `(num_thresholds + 1, num_classes)` is returned.
        - `thresholds` used for computing the precision and recall values, in
        ascending order. If `thresholds` is `"none"` or `None`, a list for each
        class is returned with 1-D Arrays of shape `(num_thresholds,)`. Otherwise,
        a 1-D Array of shape `(num_thresholds,)` is returned.

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
        If `average` is not `"macro"`, `"micro"`, `"none"` or `None`.
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
    ...     multiclass_precision_recall_curve,
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
    >>> precision, recall, thresholds = multiclass_precision_recall_curve(
    ...     target,
    ...     preds,
    ...     num_classes=3,
    ...     thresholds=None,
    ... )
    >>> precision
    [Array([0.33333334, 0.        , 0.        , 1.        ], dtype=float32),
    Array([0.33333334, 0.5       , 0.        , 1.        ], dtype=float32),
    Array([0.33333334, 0.5       , 0.        , 1.        ], dtype=float32)]
    >>> recall
    [Array([1., 0., 0., 0.], dtype=float32), Array([1., 1., 0., 0.], dtype=float32), Array([1., 1., 0., 0.], dtype=float32)]
    >>> thresholds
    [Array([0.11, 0.33, 0.84], dtype=float64), Array([0.22, 0.73, 0.92], dtype=float64), Array([0.12, 0.44, 0.67], dtype=float64)]
    >>> precision, recall, thresholds = multiclass_precision_recall_curve(
    ...     target,
    ...     preds,
    ...     num_classes=3,
    ...     thresholds=5,
    ... )
    >>> precision
    Array([[0.33333334, 0.        , 0.        , 0.        ,
            0.        , 1.        ],
           [0.33333334, 0.5       , 0.5       , 0.        ,
            0.        , 1.        ],
           [0.33333334, 0.5       , 0.        , 0.        ,
            0.        , 1.        ]], dtype=float32)
    >>> recall
    Array([[1., 0., 0., 0., 0., 0.],
           [1., 1., 1., 0., 0., 0.],
           [1., 1., 0., 0., 0., 0.]], dtype=float32)
    >>> thresholds
    Array([0.  , 0.25, 0.5 , 0.75, 1.  ], dtype=float32)

    """  # noqa: W505
    _multiclass_precision_recall_curve_validate_args(
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
        thresholds,
        ignore_index,
        average,
        xp=xp,
    )
    state = _multiclass_precision_recall_curve_update(
        target,
        preds,
        num_classes,
        thresholds,
        average,
        xp=xp,
    )
    precision, recall, thresholds_ = _multiclass_precision_recall_curve_compute(
        state,
        num_classes,
        thresholds=thresholds,
        average=average,
    )
    return PRCurve(precision, recall, thresholds_)


def _multilabel_precision_recall_curve_validate_args(
    num_labels: int,
    thresholds: Optional[Union[int, List[float], Array]],
    ignore_index: Optional[int],
) -> None:
    """Validate the arguments for the `multilabel_precision_recall_curve` function."""
    if not isinstance(num_labels, int) or num_labels < 2:
        raise ValueError(
            "Expected argument `num_labels` to be an integer larger than 1, "
            f"but got {num_labels}.",
        )
    _binary_precision_recall_curve_validate_args(thresholds, ignore_index)


def _multilabel_precision_recall_curve_validate_arrays(
    target: Array,
    preds: Array,
    num_labels: int,
    thresholds: Optional[Union[int, List[float], Array]],
    ignore_index: Optional[int],
) -> ModuleType:
    xp = _binary_precision_recall_curve_validate_arrays(
        target,
        preds,
        thresholds,
        ignore_index=ignore_index,
    )

    if preds.shape[1] != num_labels:
        raise ValueError(
            "Expected both `target.shape[1]` and `preds.shape[1]` to be equal to the number of labels "
            f"but got {preds.shape[1]} and expected {num_labels}, respectively.",
        )

    return xp


def _multilabel_precision_recall_curve_format_arrays(
    target: Array,
    preds: Array,
    num_labels: int,
    thresholds: Optional[Union[int, List[float], Array]],
    ignore_index: Optional[int],
    *,
    xp: ModuleType,
) -> Tuple[Array, Array, Optional[Array]]:
    """Format the arrays for the `multilabel_precision_recall_curve` function."""
    preds = xp.reshape(moveaxis(preds, 0, 1), (num_labels, -1)).T
    target = xp.reshape(moveaxis(target, 0, 1), (num_labels, -1)).T
    if not xp.all(to_int(preds >= 0) * to_int(preds <= 1)):
        preds = sigmoid(preds)

    thresholds = _format_thresholds(thresholds, device=apc.device(preds), xp=xp)
    if ignore_index is not None and thresholds is not None:
        preds = clone(preds)
        target = clone(target)
        # make sure that when we map, it will always result in a negative number
        # that we can filter away
        idx = target == ignore_index
        preds[idx] = -4 * num_labels * thresholds.shape[0] if thresholds.ndim > 0 else 1
        target[idx] = (
            -4 * num_labels * thresholds.shape[0] if thresholds.ndim > 0 else 1
        )

    return target, preds, thresholds


def _multilabel_precision_recall_curve_update(
    target: Array,
    preds: Array,
    num_labels: int,
    thresholds: Optional[Array],
    *,
    xp: ModuleType,
) -> Union[Array, Tuple[Array, Array]]:
    """Update the state for the `multilabel_precision_recall_curve` function."""
    if thresholds is None:
        return target, preds

    len_t = thresholds.shape[0] if thresholds.ndim > 0 else 1
    # num_samples x num_labels x num_thresholds
    preds_t = to_int(
        xp.expand_dims(xp.astype(preds, xp.float32, copy=False), axis=-1)
        >= xp.expand_dims(xp.expand_dims(thresholds, axis=0), axis=0),
    )
    unique_mapping = preds_t + 2 * xp.expand_dims(to_int(target), axis=-1)
    unique_mapping += 4 * xp.expand_dims(
        xp.expand_dims(xp.arange(num_labels, device=apc.device(preds)), axis=0),
        axis=-1,
    )
    unique_mapping += 4 * num_labels * xp.arange(len_t, device=apc.device(preds))
    unique_mapping = unique_mapping[unique_mapping >= 0]
    bins = bincount(unique_mapping, minlength=4 * num_labels * len_t)
    return xp.reshape(xp.astype(bins, xp.int32, copy=False), (len_t, num_labels, 2, 2))  # type: ignore[no-any-return]


def _multilabel_precision_recall_curve_compute(
    state: Union[Array, Tuple[Array, Array]],
    num_labels: int,
    thresholds: Optional[Array],
    ignore_index: Optional[int],
) -> Union[Tuple[Array, Array, Array], Tuple[List[Array], List[Array], List[Array]]]:
    """Compute the precision and recall for all unique thresholds."""
    if apc.is_array_api_obj(state) and thresholds is not None:
        xp = apc.array_namespace(state)
        tps = state[:, :, 1, 1]  # type: ignore[call-overload]
        fps = state[:, :, 0, 1]  # type: ignore[call-overload]
        fns = state[:, :, 1, 0]  # type: ignore[call-overload]
        precision = safe_divide(tps, tps + fps)
        recall = safe_divide(tps, tps + fns)
        precision = xp.concat(
            [
                precision,
                xp.ones(
                    (1, num_labels),
                    dtype=precision.dtype,
                    device=apc.device(precision),
                ),
            ],
        )
        recall = xp.concat(
            [
                recall,
                xp.zeros(
                    (1, num_labels),
                    dtype=recall.dtype,
                    device=apc.device(recall),
                ),
            ],
        )
        return precision.T, recall.T, thresholds

    precision_list, recall_list, thres_list = [], [], []
    for i in range(num_labels):
        target = state[0][:, i]
        preds = state[1][:, i]
        if ignore_index is not None:
            target, preds = remove_ignore_index(
                target,
                preds,
                ignore_index=ignore_index,
            )
        res = _binary_precision_recall_curve_compute(
            (target, preds),
            thresholds=None,
            pos_label=1,
        )
        precision_list.append(res[0])
        recall_list.append(res[1])
        thres_list.append(res[2])
    return precision_list, recall_list, thres_list


def multilabel_precision_recall_curve(
    target: Array,
    preds: Array,
    num_labels: int,
    thresholds: Optional[Union[int, List[float], Array]] = None,
    ignore_index: Optional[int] = None,
) -> PRCurve:
    """Compute the precision and recall for all unique thresholds.

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
        The thresholds to use for computing the precision and recall. Can be one
        of the following:
        - `None`: use all unique values in `preds` as thresholds.
        - `int`: use `int` (larger than 1) uniformly spaced thresholds in the range
          [0, 1].
        - `List[float]`: use the values in the list as bins for the thresholds.
        - `Array`: use the values in the Array as bins for the thresholds. The
          array must be 1D.
    ignore_index : int, optional, default=None
        The value in `target` that should be ignored when computing the precision
        and recall. If `None`, all values in `target` are used.

    Returns
    -------
    PRCurve
        A named tuple that contains the following elements:
        - `precision` values for all unique thresholds. If `thresholds` is `"none"`
        or `None`, a list for each class is returned with 1-D Arrays of shape
        `(num_thresholds + 1,)`. Otherwise, a 2-D Array of shape
        `(num_thresholds + 1, num_classes)` is returned.
        - `recall` values for all unique thresholds. If `thresholds` is `"none"`
        or `None`, a list for each class is returned with 1-D Arrays of shape
        `(num_thresholds + 1,)`. Otherwise, a 2-D Array of shape
        `(num_thresholds + 1, num_classes)` is returned.
        - `thresholds` used for computing the precision and recall values, in
        ascending order. If `thresholds` is `"none"` or `None`, a list for each
        class is returned with 1-D Arrays of shape `(num_thresholds,)`. Otherwise,
        a 1-D Array of shape `(num_thresholds,)` is returned.

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
    ...     multilabel_precision_recall_curve,
    ... )
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([[0, 1, 0], [1, 1, 0], [0, 0, 1]])
    >>> preds = anp.asarray(
    ...     [[0.11, 0.22, 0.67], [0.84, 0.73, 0.12], [0.33, 0.92, 0.44]],
    ... )
    >>> precision, recall, thresholds = multilabel_precision_recall_curve(
    ...     target,
    ...     preds,
    ...     num_labels=3,
    ...     thresholds=None,
    ... )
    >>> precision
    [Array([0.33333334, 0.5       , 1.        , 1.        ], dtype=float32),
    Array([0.6666667, 0.5      , 0.       , 1.       ], dtype=float32),
    Array([0.33333334, 0.5       , 0.        , 1.        ], dtype=float32)]
    >>> recall
    [Array([1., 1., 1., 0.], dtype=float32), Array([1. , 0.5, 0. , 0. ], dtype=float32), Array([1., 1., 0., 0.], dtype=float32)]
    >>> thresholds
    [Array([0.11, 0.33, 0.84], dtype=float64), Array([0.22, 0.73, 0.92], dtype=float64), Array([0.12, 0.44, 0.67], dtype=float64)]
    >>> precision, recall, thresholds = multilabel_precision_recall_curve(
    ...     target,
    ...     preds,
    ...     num_labels=3,
    ...     thresholds=5,
    ... )
    >>> precision
    Array([[0.33333334, 0.5       , 1.        , 1.        ,
            0.        , 1.        ],
           [0.6666667 , 0.5       , 0.5       , 0.        ,
            0.        , 1.        ],
           [0.33333334, 0.5       , 0.        , 0.        ,
            0.        , 1.        ]], dtype=float32)
    >>> recall
    Array([[1. , 1. , 1. , 1. , 0. , 0. ],
           [1. , 0.5, 0.5, 0. , 0. , 0. ],
           [1. , 1. , 0. , 0. , 0. , 0. ]], dtype=float32)
    >>> thresholds
    Array([0.  , 0.25, 0.5 , 0.75, 1.  ], dtype=float32)

    """  # noqa: W505
    _multilabel_precision_recall_curve_validate_args(
        num_labels,
        thresholds,
        ignore_index,
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
        thresholds,
        ignore_index,
        xp=xp,
    )
    state = _multilabel_precision_recall_curve_update(
        target,
        preds,
        num_labels,
        thresholds,
        xp=xp,
    )
    precision, recall, thresholds_ = _multilabel_precision_recall_curve_compute(
        state,
        num_labels,
        thresholds,
        ignore_index,
    )
    return PRCurve(precision, recall, thresholds_)
