"""Functional interface for the Symmetric Mean Absolute Percentage Error metric."""

from typing import Tuple, Union

import array_api_compat as apc

from cyclops.evaluate.metrics.experimental.utils.types import Array
from cyclops.evaluate.metrics.experimental.utils.validation import (
    _basic_input_array_checks,
    _check_same_shape,
)


def _symmetric_mean_absolute_percentage_error_update(
    target: Array,
    preds: Array,
    epsilon: float = 1.17e-06,
) -> Tuple[Array, int]:
    """Update and return variables required to compute Symmetric MAPE.

    Parameters
    ----------
    target : Array
        Ground truth target values.
    preds : Array
        Estimated target values.
    epsilon : float, optional, default=1.17e-06
        Specifies the lower bound for target values. Any target value below epsilon
        is set to epsilon (avoids division by zero errors).

    Returns
    -------
    Tuple[Array, int]
        Sum of absolute value of percentage errors over all observations and number
        of observations.

    """
    _basic_input_array_checks(target, preds)
    _check_same_shape(target, preds)
    xp = apc.array_namespace(target, preds)

    abs_diff = xp.abs(preds - target)
    arr_sum = xp.abs(target) + xp.abs(preds)
    clamped_val = xp.where(
        arr_sum < epsilon,
        xp.asarray(epsilon, dtype=arr_sum.dtype, device=apc.device(arr_sum)),
        arr_sum,
    )
    abs_per_error = abs_diff / clamped_val

    sum_abs_per_error = 2 * xp.sum(abs_per_error, dtype=xp.float32)

    num_obs = int(apc.size(target) or 0)

    return sum_abs_per_error, num_obs


def _symmetric_mean_absolute_percentage_error_compute(
    sum_abs_per_error: Array,
    num_obs: Union[int, Array],
) -> Array:
    """Compute the Symmetric Mean Absolute Percentage Error.

    Parameters
    ----------
    sum_abs_per_error : Array
        Sum of absolute value of percentage errors over all observations.
        ``(percentage error = (target - prediction) / target)``
    num_obs : int, Array
        Total number of observations.

    Returns
    -------
    Array
        The symmetric mean absolute percentage error.

    """
    return sum_abs_per_error / num_obs  # type: ignore[no-any-return]


def symmetric_mean_absolute_percentage_error(target: Array, preds: Array) -> Array:
    """Compute the symmetric mean absolute percentage error (SMAPE).

    Parameters
    ----------
    target : Array
        Ground truth target values.
    preds : Array
        Estimated target values.

    Returns
    -------
    Array
        The symmetric mean absolute percentage error.

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
    >>> from cyclops.evaluate.metrics.experimental.functional import (
    ...     symmetric_mean_absolute_percentage_error,
    ... )
    >>> target = anp.asarray([1.0, 10.0, 1e6])
    >>> preds = anp.asarray([0.9, 15.0, 1.2e6])
    >>> symmetric_mean_absolute_percentage_error(target, preds)
    Array(0.2290271, dtype=float32)

    """
    sum_abs_per_error, num_obs = _symmetric_mean_absolute_percentage_error_update(
        target,
        preds,
    )
    return _symmetric_mean_absolute_percentage_error_compute(
        sum_abs_per_error,
        num_obs,
    )
