"""Helper functions for testing metrics."""
from functools import partial
from typing import Any, Callable, Dict

import numpy as np
from sklearn.utils._testing import assert_allclose


def _functional_test(  # pylint: disable=too-many-arguments
    target: np.ndarray,
    preds: np.ndarray,
    metric_functional: Callable,
    sk_metric: Callable,
    metric_args: Dict[str, Any] = None,
    atol: float = 1e-8,
    **kwargs_update: Any,
) -> None:
    """Test functional metric against sklearn metric.

    Arguments
    ---------
        target: np.ndarray
            The target.
        preds: np.ndarray
            The predictions.
        metric_functional: Callable
            The functional metric function.
        sk_metric: Callable
            The sklearn metric function.
        metric_args: Dict[str, Any]
            The arguments to pass to the metric function.
        atol: float
            The absolute tolerance.
        **kwargs_update: Any
            The keyword arguments pass to the metric update function.

    Returns
    -------
        None

    Raises
    ------
        AssertionError
            If the metric values are not equal.

    """
    assert preds.shape[0] == target.shape[0]
    num_batches = preds.shape[0]

    if not metric_args:
        metric_args = {}

    metric = partial(metric_functional, **metric_args)

    for i in range(num_batches):
        result = metric(target[i], preds[i], **kwargs_update)
        sk_result = sk_metric(target[i], preds[i], **kwargs_update)

        # assert its the same
        assert_allclose(result, sk_result, atol=atol)
