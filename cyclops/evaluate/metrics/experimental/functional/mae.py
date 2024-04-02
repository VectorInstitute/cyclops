"""Functional interface for the mean absolute error metric."""

from typing import Tuple, Union

import array_api_compat as apc

from cyclops.evaluate.metrics.experimental.utils.types import Array
from cyclops.evaluate.metrics.experimental.utils.validation import (
    _basic_input_array_checks,
    _check_same_shape,
    is_floating_point,
)


def _mean_absolute_error_update(target: Array, preds: Array) -> Tuple[Array, int]:
    """Update and return variables required to compute Mean Absolute Error."""
    _basic_input_array_checks(target, preds)
    _check_same_shape(target, preds)
    xp = apc.array_namespace(target, preds)

    target = (
        target
        if is_floating_point(target)
        else xp.astype(target, xp.float32, copy=False)
    )
    preds = (
        preds if is_floating_point(preds) else xp.astype(preds, xp.float32, copy=False)
    )

    sum_abs_error = xp.sum(xp.abs(preds - target), dtype=xp.float32)
    num_obs = int(apc.size(target) or 0)
    return sum_abs_error, num_obs


def _mean_absolute_error_compute(
    sum_abs_error: Array,
    num_obs: Union[int, Array],
) -> Array:
    """Compute Mean Absolute Error.

    Parameters
    ----------
    sum_abs_error : Array
        Sum of absolute value of errors over all observations.
    num_obs : int, Array
        Total number of observations.

    Returns
    -------
    Array
        The mean absolute error.

    """
    return sum_abs_error / num_obs  # type: ignore[no-any-return]


def mean_absolute_error(target: Array, preds: Array) -> Array:
    """Compute the mean absolute error.

    Parameters
    ----------
    target : Array
        Ground truth target values.
    preds : Array
        Estimated target values.

    Return
    ------
    Array
        The mean absolute error.

    Raises
    ------
    TypeError
        If `target` or `preds` is not an array object that is compatible with
        the Python array API standard.
    ValueError
        If `target` or `preds` is empty.
    ValueError
        If `target` or `preds` is not a numeric array.
    ValueError
        If the shape of `target` and `preds` are not the same.

    Examples
    --------
    >>> import numpy.array_api as anp
    >>> from cyclops.evaluate.metrics.experimental.functional import mean_absolute_error
    >>> target = anp.asarray([0.009, 1.05, 2.0, 3.0])
    >>> preds = anp.asarray([0.0, 1.0, 2.0, 2.0])
    >>> mean_absolute_error(target, preds)
    Array(0.26475, dtype=float32)

    """
    sum_abs_error, num_obs = _mean_absolute_error_update(target, preds)
    return _mean_absolute_error_compute(sum_abs_error, num_obs)
