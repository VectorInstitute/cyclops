"""Functions for computing average precision (AUPRC) for classification tasks."""
from typing import List, Optional, Tuple, Union

import array_api_compat as apc

from cyclops.evaluate.metrics.experimental.functional.precision_recall_curve import (
    _binary_precision_recall_curve_compute,
    _binary_precision_recall_curve_format_arrays,
    _binary_precision_recall_curve_update,
    _binary_precision_recall_curve_validate_args,
    _binary_precision_recall_curve_validate_arrays,
)
from cyclops.evaluate.metrics.experimental.utils.ops import _diff
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
    """Compute average precision for binary classification task.

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
    ...     binary_average_precision
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
