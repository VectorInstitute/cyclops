"""Functional interface for the Weighted Mean Absolute Percentage Error metric."""

from typing import Tuple

import array_api_compat as apc

from cyclops.evaluate.metrics.experimental.utils.types import Array
from cyclops.evaluate.metrics.experimental.utils.validation import (
    _basic_input_array_checks,
    _check_same_shape,
)


def _weighted_mean_absolute_percentage_error_update(
    target: Array,
    preds: Array,
) -> Tuple[Array, Array]:
    """Update and return variables required to compute the weighted MAPE."""
    _basic_input_array_checks(target, preds)
    _check_same_shape(target, preds)
    xp = apc.array_namespace(target, preds)

    sum_abs_error = xp.sum(xp.abs((preds - target)), dtype=xp.float32)
    sum_scale = xp.sum(xp.abs(target), dtype=xp.float32)

    return sum_abs_error, sum_scale


def _weighted_mean_absolute_percentage_error_compute(
    sum_abs_error: Array,
    sum_scale: Array,
    epsilon: float = 1.17e-06,
) -> Array:
    """Compute Weighted Absolute Percentage Error.

    Parameters
    ----------
    sum_abs_error : Array
        Sum of absolute value of errors over all observations.
    sum_scale : Array
        Sum of absolute value of target values over all observations.
    epsilon : float, optional, default=1.17e-06
        Specifies the lower bound for target values. Any target value below epsilon
        is set to epsilon (avoids division by zero errors).

    """
    xp = apc.array_namespace(sum_abs_error, sum_scale)
    clamped_sum_scale = xp.where(
        sum_scale < epsilon,
        xp.asarray(epsilon, dtype=sum_scale.dtype, device=apc.device(sum_scale)),
        sum_scale,
    )
    return sum_abs_error / clamped_sum_scale  # type: ignore[no-any-return]


def weighted_mean_absolute_percentage_error(
    target: Array,
    preds: Array,
    epsilon: float = 1.17e-06,
) -> Array:
    """Compute the weighted mean absolute percentage error (`WMAPE`).

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
    Array
        The weighted mean absolute percentage error.

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
    >>> preds = anp.asarray([1.24, 2.3, 3.4, 4.5, 5.6, 6.7])
    >>> target = anp.asarray([1.2, 2.4, 3.6, 4.8, 6.0, 7.2])
    >>> weighted_mean_absolute_percentage_error(target, preds)
    Array(0.06111111, dtype=float32)

    """
    sum_abs_error, sum_scale = _weighted_mean_absolute_percentage_error_update(
        target,
        preds,
    )
    return _weighted_mean_absolute_percentage_error_compute(
        sum_abs_error,
        sum_scale,
        epsilon=epsilon,
    )
