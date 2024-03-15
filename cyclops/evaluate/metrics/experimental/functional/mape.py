"""Functional interface for the mean absolute percentage error metric."""

from typing import Tuple, Union

import array_api_compat as apc

from cyclops.evaluate.metrics.experimental.utils.types import Array
from cyclops.evaluate.metrics.experimental.utils.validation import (
    _basic_input_array_checks,
    _check_same_shape,
)


def _mean_absolute_percentage_error_update(
    target: Array,
    preds: Array,
    epsilon: float = 1.17e-06,
) -> Tuple[Array, int]:
    """Update and return variables required to compute the Mean Percentage Error.

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
    abs_target = xp.abs(target)
    clamped_abs_target = xp.where(
        abs_target < epsilon,
        xp.asarray(epsilon, dtype=abs_target.dtype, device=apc.device(abs_target)),
        abs_target,
    )
    abs_per_error = abs_diff / clamped_abs_target

    sum_abs_per_error = xp.sum(abs_per_error, dtype=xp.float32)

    num_obs = int(apc.size(target) or 0)

    return sum_abs_per_error, num_obs


def _mean_absolute_percentage_error_compute(
    sum_abs_per_error: Array,
    num_obs: Union[int, Array],
) -> Array:
    """Compute the Mean Absolute Percentage Error.

    Parameters
    ----------
    sum_abs_per_error : Array
        Sum of absolute value of percentage errors over all observations.
        ``(percentage error = (target - prediction) / target)``
    num_obs : int, Array
        Number of observations.

    Returns
    -------
    Array
        The mean absolute percentage error.

    """
    return sum_abs_per_error / num_obs  # type: ignore[no-any-return]


def mean_absolute_percentage_error(target: Array, preds: Array) -> Array:
    """Compute the mean absolute percentage error.

    Parameters
    ----------
    target : Array
        Ground truth target values.
    preds : Array
        Estimated target values.

    Returns
    -------
    Array
        The mean absolute percentage error.

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

    Notes
    -----
    The epsilon value is taken from `scikit-learn's implementation of MAPE`.

    Examples
    --------
    >>> import numpy.array_api as anp
    >>> from cyclops.evaluate.metrics.experimental.functional import (
    ...     mean_absolute_percentage_error,
    ... )
    >>> target = anp.asarray([1.0, 10.0, 1e6])
    >>> preds = anp.asarray([0.9, 15.0, 1.2e6])
    >>> mean_absolute_percentage_error(target, preds)
    Array(0.26666668, dtype=float32)

    """
    sum_abs_per_error, num_obs = _mean_absolute_percentage_error_update(target, preds)
    return _mean_absolute_percentage_error_compute(sum_abs_per_error, num_obs)
