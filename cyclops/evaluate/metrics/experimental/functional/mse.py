"""Functional interface for the mean squared error metric."""

from typing import Tuple, Union

import array_api_compat as apc

from cyclops.evaluate.metrics.experimental.utils.ops import flatten, squeeze_all
from cyclops.evaluate.metrics.experimental.utils.types import Array
from cyclops.evaluate.metrics.experimental.utils.validation import (
    _basic_input_array_checks,
    _check_same_shape,
)


def _mean_squared_error_update(
    target: Array,
    preds: Array,
    num_outputs: int,
) -> Tuple[Array, int]:
    """Update and returns variables required to compute the Mean Squared Error.

    Parameters
    ----------
    target : Array
        Ground truth target values.
    preds : Array
        Estimated target values.
    num_outputs : int
        Number of outputs in multioutput setting.

    Returns
    -------
    Tuple[Array, int]
        Sum of square of errors over all observations and number of observations.

    """
    _basic_input_array_checks(target, preds)
    _check_same_shape(target, preds)
    xp = apc.array_namespace(target, preds)

    if num_outputs == 1:
        target = flatten(target, copy=False)
        preds = flatten(preds, copy=False)

    diff = preds - target
    sum_squared_error = xp.sum(diff * diff, axis=0, dtype=xp.float32)
    return sum_squared_error, target.shape[0]


def _mean_squared_error_compute(
    sum_squared_error: Array,
    num_obs: Union[int, Array],
    squared: bool = True,
) -> Array:
    """Compute Mean Squared Error.

    Parameters
    ----------
    sum_squared_error : Array
        Sum of square of errors over all observations.
    num_obs : Array
        Number of predictions or observations.
    squared : bool, optional, default=True
        Whether to return MSE or RMSE. If set to False, returns RMSE.

    Returns
    -------
    Array
        The mean squared error or root mean squared error.

    """
    xp = apc.array_namespace(sum_squared_error)
    return squeeze_all(
        sum_squared_error / num_obs
        if squared
        else xp.sqrt(sum_squared_error / num_obs),
    )


def mean_squared_error(
    target: Array,
    preds: Array,
    squared: bool = True,
    num_outputs: int = 1,
) -> Array:
    """Compute mean squared error.

    Parameters
    ----------
    target : Array
        Ground truth target values.
    preds : Array
        Estimated target values.
    squared : bool, optional, default=True
        Whether to return mean squared error or root mean squared error. If set
        to `False`, returns the root mean squared error.
    num_outputs : int, optional, default=1
        Number of outputs in multioutput setting.

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

    Returns
    -------
    Array
        The mean squared error or root mean squared error.

    Examples
    --------
    >>> import numpy.array_api as anp
    >>> from cyclops.evaluate.metrics.experimental.functional import mean_squared_error
    >>> target = anp.asarray([0.0, 1.0, 2.0, 3.0])
    >>> preds = anp.asarray([0.025, 1.0, 2.0, 2.44])
    >>> mean_squared_error(target, preds)
    Array(0.07855625, dtype=float32)
    >>> mean_squared_error(target, preds, squared=False)
    Array(0.2802789, dtype=float32)
    >>> target = anp.asarray([[0.0, 1.0], [2.0, 3.0]])
    >>> preds = anp.asarray([[0.025, 1.0], [2.0, 2.44]])
    >>> mean_squared_error(target, preds, num_outputs=2)
    Array([0.0003125, 0.1568   ], dtype=float32)
    >>> mean_squared_error(target, preds, squared=False, num_outputs=2)
    Array([0.01767767, 0.3959798 ], dtype=float32)


    """
    sum_squared_error, num_obs = _mean_squared_error_update(
        preds,
        target,
        num_outputs=num_outputs,
    )
    return _mean_squared_error_compute(sum_squared_error, num_obs, squared=squared)
