"""Functions for computing Receiver Operating Characteristic (ROC) curves."""

import warnings
from typing import List, Literal, NamedTuple, Optional, Tuple, Union

import array_api_compat as apc

from cyclops.evaluate.metrics.experimental.functional.precision_recall_curve import (
    _binary_clf_curve,
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
from cyclops.evaluate.metrics.experimental.utils.ops import (
    _interp,
    flatten,
    remove_ignore_index,
    safe_divide,
)
from cyclops.evaluate.metrics.experimental.utils.types import Array


class ROCCurve(NamedTuple):
    """Named tuple to store ROC curve (FPR, TPR and thresholds)."""

    fpr: Union[Array, List[Array]]
    tpr: Union[Array, List[Array]]
    thresholds: Union[Array, List[Array]]


def _binary_roc_compute(
    state: Union[Array, Tuple[Array, Array]],
    thresholds: Optional[Array],
    pos_label: int = 1,
) -> Tuple[Array, Array, Array]:
    """Compute the binary ROC curve."""
    if apc.is_array_api_obj(state) and thresholds is not None:
        xp = apc.array_namespace(state, thresholds)
        tps = state[:, 1, 1]  # type: ignore[call-overload]
        fps = state[:, 0, 1]  # type: ignore[call-overload]
        fns = state[:, 1, 0]  # type: ignore[call-overload]
        tns = state[:, 0, 0]  # type: ignore[call-overload]
        tpr = xp.flip(safe_divide(tps, tps + fns), axis=0)
        fpr = xp.flip(safe_divide(fps, fps + tns), axis=0)
        thresh = xp.flip(thresholds, axis=0)
    else:
        xp = apc.array_namespace(state[0], state[1])
        fps, tps, thresh = _binary_clf_curve(
            state[0],
            state[1],
            pos_label=pos_label,
        )

        # add extra threshold position so that the curve starts at (0, 0)
        tps = xp.concat([xp.zeros(1, dtype=tps.dtype, device=apc.device(tps)), tps])
        fps = xp.concat([xp.zeros(1, dtype=fps.dtype, device=apc.device(fps)), fps])
        thresh = xp.concat(
            [
                xp.ones(1, dtype=thresh.dtype, device=apc.device(thresh)),
                thresh,
            ],
        )

        if fps[-1] <= 0:
            warnings.warn(
                "No negative samples in targets false positive value should be "
                "meaningless. Returning an array of 0s instead.",
                UserWarning,
                stacklevel=1,
            )
            fpr = xp.zeros_like(thresh)
        else:
            fpr = fps / fps[-1]

        if tps[-1] <= 0:
            warnings.warn(
                "No positive samples in targets true positive value should be "
                "meaningless. Returning an array of 0s instead.",
                UserWarning,
                stacklevel=1,
            )
            tpr = xp.zeros_like(fpr)
        else:
            tpr = tps / tps[-1]

    return fpr, tpr, thresh


def binary_roc(
    target: Array,
    preds: Array,
    thresholds: Optional[Union[int, List[float], Array]] = None,
    ignore_index: Optional[int] = None,
) -> ROCCurve:
    """Compute the receiver operating characteristic (ROC) curve for binary tasks.

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
        The thresholds to use for computing the ROC curve. Can be one of the following:
        - `None`: use all unique values in `preds` as thresholds.
        - `int`: use `int` (larger than 1) uniformly spaced thresholds in the range
          [0, 1].
        - `List[float]`: use the values in the list as bins for the thresholds.
        - `Array`: use the values in the Array as bins for the thresholds. The
          array must be 1D.
    ignore_index : int, optional, default=None
        The value in `target` that should be ignored when computing the ROC curve.
        If `None`, all values in `target` are used.

    Returns
    -------
    ROCCurve
        A named tuple containing the false positive rate (FPR), true positive rate
        (TPR) and thresholds. The FPR and TPR are arrays of shape
        `(num_thresholds + 1,)` and the thresholds are an array of shape
        `(num_thresholds,)`.

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
    >>> from cyclops.evaluate.metrics.experimental.functional import binary_roc
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([0, 1, 0, 1, 0, 1])
    >>> preds = anp.asarray([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])
    >>> fpr, tpr, thresholds = binary_roc(target, preds, thresholds=None)
    >>> fpr
    Array([0.        , 0.        , 0.33333334, 0.33333334,
           0.6666667 , 0.6666667 , 1.        ], dtype=float32)
    >>> tpr
    Array([0.        , 0.33333334, 0.33333334, 0.6666667 ,
           0.6666667 , 1.        , 1.        ], dtype=float32)
    >>> thresholds
    Array([1.  , 0.92, 0.84, 0.73, 0.33, 0.22, 0.11], dtype=float64)
    >>> fpr, tpr, thresholds = binary_roc(
    ...     target,
    ...     preds,
    ...     thresholds=5,
    ... )
    >>> fpr
    Array([0.        , 0.33333334, 0.33333334, 0.6666667 ,
           1.        ], dtype=float32)
    >>> tpr
    Array([0.        , 0.33333334, 0.6666667 , 0.6666667 ,
           1.        ], dtype=float32)
    >>> thresholds
    Array([1.  , 0.75, 0.5 , 0.25, 0.  ], dtype=float32)

    """
    _binary_precision_recall_curve_validate_args(thresholds, ignore_index)
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
    fpr, tpr, thresh = _binary_roc_compute(state, thresholds)
    return ROCCurve(fpr, tpr, thresh)


def _multiclass_roc_compute(
    state: Union[Array, Tuple[Array, Array]],
    num_classes: int,
    thresholds: Optional[Array],
    average: Optional[Literal["macro", "micro", "none"]] = None,
) -> Union[Tuple[Array, Array, Array], Tuple[List[Array], List[Array], List[Array]]]:
    """Compute the multiclass ROC curve."""
    if average == "micro":
        return _binary_roc_compute(state, thresholds)

    if apc.is_array_api_obj(state) and thresholds is not None:
        xp = apc.array_namespace(state, thresholds)
        tps = state[:, :, 1, 1]  # type: ignore[call-overload]
        fps = state[:, :, 0, 1]  # type: ignore[call-overload]
        fns = state[:, :, 1, 0]  # type: ignore[call-overload]
        tns = state[:, :, 0, 0]  # type: ignore[call-overload]
        tpr = xp.flip(safe_divide(tps, tps + fns), axis=0).T
        fpr = xp.flip(safe_divide(fps, fps + tns), axis=0).T
        thresh = xp.flip(thresholds, axis=0)
        array_state = True
    else:
        xp = apc.array_namespace(state[0], state[1])
        fpr_list, tpr_list, thresh_list = [], [], []
        for i in range(num_classes):
            res = _binary_roc_compute(
                (state[0], state[1][:, i]),
                thresholds=None,
                pos_label=i,
            )
            fpr_list.append(res[0])
            tpr_list.append(res[1])
            thresh_list.append(res[2])
        array_state = False

    if average == "macro":
        thresh = (
            xp.concat([xp.expand_dims(thresh, axis=0)] * num_classes, axis=0)  # repeat
            if array_state
            else xp.concat(xp.asarray(thresh_list), 0)
        )
        thresh = xp.sort(thresh, descending=True)
        mean_fpr = flatten(fpr) if array_state else xp.concat(xp.asarray(fpr_list), 0)
        mean_fpr = xp.sort(mean_fpr)
        mean_tpr = xp.zeros_like(mean_fpr)
        for i in range(num_classes):
            mean_tpr += _interp(
                mean_fpr,
                fpr[i] if array_state else fpr_list[i],
                tpr[i] if array_state else tpr_list[i],
            )
        mean_tpr /= num_classes
        return mean_fpr, mean_tpr, thresh

    if array_state:
        return fpr, tpr, thresh
    return fpr_list, tpr_list, thresh_list


def multiclass_roc(
    target: Array,
    preds: Array,
    num_classes: int,
    thresholds: Optional[Union[int, List[float], Array]] = None,
    average: Optional[Literal["macro", "micro", "none"]] = None,
    ignore_index: Optional[Union[int, Tuple[int]]] = None,
) -> ROCCurve:
    """Compute the receiver operating characteristic (ROC) curve for multiclass tasks.

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
    average : {"macro", "micro", "none"}, optional, default=None
        The type of averaging to use for computing the ROC curve. Can be one of
        the following:
        - `"macro"`: interpolates the curves from each class at a combined set of
          thresholds and then average over the classwise interpolated curves.
        - `"micro"`: one-hot encodes the targets and flattens the predictions,
          considering all classes jointly as a binary problem.
        - `"none"`: do not average over the classwise curves.
    ignore_index : int or Tuple[int], optional, default=None
        The value(s) in `target` that should be ignored when computing the ROC curve.
        If `None`, all values in `target` are used.

    Returns
    -------
    ROCCurve
        A named tuple that contains the false positive rate, true positive rate,
        and the thresholds used for computing the ROC curve. If `thresholds` is `"none"`
        or `None`, a list of TPRs and FPRs for each class is returned with 1-D Arrays
        of shape `(num_thresholds + 1,)`. Otherwise, a 2-D Array of shape
        `(num_thresholds + 1, num_classes)` is returned.
        Similarly, a list of thresholds for each class is returned
        with 1-D Arrays of shape `(num_thresholds,)`. Otherwise, a 1-D Array of
        shape `(num_thresholds,)` is returned.

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
    >>> from cyclops.evaluate.metrics.experimental.functional import multiclass_roc
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
    >>> fpr, tpr, thresholds = multiclass_roc(
    ...     target,
    ...     preds,
    ...     num_classes=3,
    ...     thresholds=None,
    ... )
    >>> fpr
    [Array([0. , 0.5, 1. , 1. ], dtype=float32),
    Array([0. , 0.5, 0.5, 1. ], dtype=float32),
    Array([0. , 0.5, 0.5, 1. ], dtype=float32)]
    >>> tpr
    [Array([0., 0., 0., 1.], dtype=float32),
    Array([0., 0., 1., 1.], dtype=float32),
    Array([0., 0., 1., 1.], dtype=float32)]
    >>> thresholds
    [Array([1.  , 0.84, 0.33, 0.11], dtype=float64),
    Array([1.  , 0.92, 0.73, 0.22], dtype=float64),
    Array([1.  , 0.67, 0.44, 0.12], dtype=float64)]
    >>> fpr, tpr, thresholds = multiclass_roc(
    ...     target,
    ...     preds,
    ...     num_classes=3,
    ...     thresholds=5,
    ... )
    >>> fpr
    Array([[0. , 0.5, 0.5, 1. , 1. ],
           [0. , 0.5, 0.5, 0.5, 1. ],
           [0. , 0. , 0.5, 0.5, 1. ]], dtype=float32)
    >>> tpr
    Array([[0., 0., 0., 0., 1.],
           [0., 0., 1., 1., 1.],
           [0., 0., 0., 1., 1.]], dtype=float32)
    >>> thresholds
    Array([1.  , 0.75, 0.5 , 0.25, 0.  ], dtype=float32)

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
        ignore_index,
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
    fpr_, tpr_, thresholds_ = _multiclass_roc_compute(
        state,
        num_classes,
        thresholds=thresholds,
        average=average,
    )
    return ROCCurve(fpr_, tpr_, thresholds_)


def _multilabel_roc_compute(
    state: Union[Array, Tuple[Array, Array]],
    num_labels: int,
    thresholds: Optional[Array],
    ignore_index: Optional[int],
) -> Union[Tuple[Array, Array, Array], Tuple[List[Array], List[Array], List[Array]]]:
    """Compute the multilabel ROC curve."""
    if apc.is_array_api_obj(state) and thresholds is not None:
        xp = apc.array_namespace(state)
        tps = state[:, :, 1, 1]  # type: ignore[call-overload]
        fps = state[:, :, 0, 1]  # type: ignore[call-overload]
        fns = state[:, :, 1, 0]  # type: ignore[call-overload]
        tns = state[:, :, 0, 0]  # type: ignore[call-overload]
        tpr = xp.flip(safe_divide(tps, tps + fns), axis=0).T
        fpr = xp.flip(safe_divide(fps, fps + tns), axis=0).T
        thresh = xp.flip(thresholds, axis=0)
        return fpr, tpr, thresh

    fpr_list, tpr_list, thresh_list = [], [], []
    for i in range(num_labels):
        target = state[0][:, i]
        preds = state[1][:, i]
        if ignore_index is not None:
            target, preds = remove_ignore_index(
                target,
                preds,
                ignore_index=ignore_index,
            )
        res = _binary_roc_compute((target, preds), thresholds=None, pos_label=1)
        fpr_list.append(res[0])
        tpr_list.append(res[1])
        thresh_list.append(res[2])
    return fpr_list, tpr_list, thresh_list


def multilabel_roc(
    target: Array,
    preds: Array,
    num_labels: int,
    thresholds: Optional[Union[int, List[float], Array]] = None,
    ignore_index: Optional[int] = None,
) -> ROCCurve:
    """Compute the receiver operating characteristic (ROC) curve for multilabel tasks.

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
    ignore_index : int, optional, default=None
        The value in `target` that should be ignored when computing the ROC curve.
        If `None`, all values in `target` are used.

    Returns
    -------
    ROCCurve
        A named tuple that contains the false positive rate, true positive rate,
        and the thresholds used for computing the ROC curve. If `thresholds` is `"none"`
        or `None`, a list of TPRs and FPRs for each class is returned with 1-D Arrays
        of shape `(num_thresholds + 1,)`. Otherwise, a 2-D Array of shape
        `(num_thresholds + 1, num_classes)` is returned.
        Similarly, a list of thresholds for each class is returned
        with 1-D Arrays of shape `(num_thresholds,)`. Otherwise, a 1-D Array of
        shape `(num_thresholds,)` is returned.

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
    >>> from cyclops.evaluate.metrics.experimental.functional import multilabel_roc
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([[0, 1, 0], [1, 1, 0], [0, 0, 1]])
    >>> preds = anp.asarray(
    ...     [[0.11, 0.22, 0.67], [0.84, 0.73, 0.12], [0.33, 0.92, 0.44]],
    ... )
    >>> fpr, tpr, thresholds = multilabel_roc(
    ...     target,
    ...     preds,
    ...     num_labels=3,
    ...     thresholds=None,
    ... )
    >>> fpr
    [Array([0. , 0. , 0.5, 1. ], dtype=float32),
    Array([0., 1., 1., 1.], dtype=float32),
    Array([0. , 0.5, 0.5, 1. ], dtype=float32)]
    >>> tpr
    [Array([0., 1., 1., 1.], dtype=float32),
    Array([0. , 0. , 0.5, 1. ], dtype=float32),
    Array([0., 0., 1., 1.], dtype=float32)]
    >>> thresholds
    [Array([1.  , 0.84, 0.33, 0.11], dtype=float64),
    Array([1.  , 0.92, 0.73, 0.22], dtype=float64),
    Array([1.  , 0.67, 0.44, 0.12], dtype=float64)]
    >>> fpr, tpr, thresholds = multilabel_roc(
    ...     target,
    ...     preds,
    ...     num_labels=3,
    ...     thresholds=5,
    ... )
    >>> fpr
    Array([[0. , 0. , 0. , 0.5, 1. ],
           [0. , 1. , 1. , 1. , 1. ],
           [0. , 0. , 0.5, 0.5, 1. ]], dtype=float32)
    >>> tpr
    Array([[0. , 1. , 1. , 1. , 1. ],
           [0. , 0. , 0.5, 0.5, 1. ],
           [0. , 0. , 0. , 1. , 1. ]], dtype=float32)
    >>> thresholds
    Array([1.  , 0.75, 0.5 , 0.25, 0.  ], dtype=float32)

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
    fpr_, tpr_, thresholds_ = _multilabel_roc_compute(
        state,
        num_labels,
        thresholds,
        ignore_index,
    )
    return ROCCurve(fpr_, tpr_, thresholds_)
