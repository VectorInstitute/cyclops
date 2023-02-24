"""Helper functions for testing metrics."""

from functools import partial
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np


class MetricTester:
    """Helper class for testing metrics."""

    @staticmethod
    def run_functional_test(  # pylint: disable=too-many-arguments
        target: np.ndarray,
        preds: np.ndarray,
        metric_functional: Callable,
        sk_metric: Callable,
        metric_args: Optional[Dict[str, Any]] = None,
        atol: float = 1e-8,
        **kwargs_update: Any,
    ) -> None:
        """Test functional metric against sklearn metric.

        Parameters
        ----------
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
            If predictions and targets do not have the same number of
            samples.
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
            _assert_allclose(result, sk_result, atol=atol)

    @staticmethod
    def run_class_test(
        target: np.ndarray,
        preds: np.ndarray,
        metric_class: Callable,
        sk_metric: Callable,
        metric_args: Optional[Dict[str, Any]] = None,
        atol: float = 1e-8,
        **kwargs_update: Any,
    ) -> None:
        """Test metric wrapper class against sklearn metric.

        Parameters
        ----------
        target: np.ndarray
            The target.
        preds: np.ndarray
            The predictions.
        metric_class: Callable
            The wrapper class for the metric.
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
            If predictions and targets do not have the same number of
            samples.
        AssertionError
            If the metric values are not equal.

        """
        assert preds.shape[0] == target.shape[0]
        num_batches = preds.shape[0]

        if not metric_args:
            metric_args = {}

        metric = metric_class(**metric_args)

        for i in range(num_batches):
            result = metric(target[i], preds[i])  # update state and compute on batch
            sk_result = sk_metric(target[i], preds[i], **kwargs_update)

            # assert its the same
            _assert_allclose(result, sk_result, atol=atol)

        cyclops_global_result = metric.compute()

        all_target = np.concatenate(list(target))
        all_preds = np.concatenate(list(preds))
        sk_global_result = sk_metric(all_target, all_preds, **kwargs_update)

        _assert_allclose(cyclops_global_result, sk_global_result, atol=atol)


def _assert_allclose(data_a: Any, data_b: Any, atol: float = 1e-8):
    """Assert allclose."""
    if isinstance(data_a, (np.ndarray, np.ScalarType)):
        np.allclose(data_a, data_b, atol=atol)
    elif isinstance(data_a, Sequence):
        for element_a, element_b in zip(data_a, data_b):
            _assert_allclose(element_a, element_b, atol=atol)
    elif isinstance(data_a, Mapping):
        assert data_a.keys() == data_b.keys()
        for value_a, value_b in zip(data_a.values(), data_b.values()):
            _assert_allclose(value_a, value_b, atol=atol)
    else:
        raise ValueError(
            f"Unknown format for comparison: {type(data_a)} and" f" {type(data_b)}"
        )
