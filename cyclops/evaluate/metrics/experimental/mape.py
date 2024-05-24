"""Mean Absolute Percentage Error (MAPE) metric."""

from typing import Any

from cyclops.evaluate.metrics.experimental.functional.mape import (
    _mean_absolute_percentage_error_compute,
    _mean_absolute_percentage_error_update,
)
from cyclops.evaluate.metrics.experimental.metric import Metric
from cyclops.evaluate.metrics.experimental.utils.types import Array


class MeanAbsolutePercentageError(Metric):
    """Mean Absolute Percentage Error.

    Parameters
    ----------
    epsilon : float, optional, default=1.17e-06
        Specifies the lower bound for target values. Any target value below epsilon
        is set to epsilon (avoids division by zero errors).
    **kwargs : Any
        Keyword arguments to pass to the `Metric` base class.

    Examples
    --------
    >>> import numpy.array_api as anp
    >>> from cyclops.evaluate.metrics.experimental import MeanAbsolutePercentageError
    >>> target = anp.asarray([0.009, 1.05, 2.0, 3.0])
    >>> preds = anp.asarray([0.0, 1.0, 2.0, 2.0])
    >>> metric = MeanAbsolutePercentageError()
    >>> metric(target, preds)
    Array(0.34523812, dtype=float32)

    """

    name: str = "Mean Absolute Percentage Error"

    def __init__(self, epsilon: float = 1.17e-6, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not isinstance(epsilon, float):
            raise TypeError(f"Expected `epsilon` to be a float. Got {type(epsilon)}")
        self.epsilon = epsilon

        self.add_state_default_factory(
            "sum_abs_per_error",
            lambda xp: xp.asarray(0.0, dtype=xp.float32, device=self.device),  # type: ignore
            dist_reduce_fn="sum",
        )
        self.add_state_default_factory(
            "num_obs",
            lambda xp: xp.asarray(0.0, dtype=xp.float32, device=self.device),  # type: ignore
            dist_reduce_fn="sum",
        )

    def _update_state(self, target: Array, preds: Array) -> None:
        """Update state of metric."""
        sum_abs_per_error, num_obs = _mean_absolute_percentage_error_update(
            target,
            preds,
            self.epsilon,
        )
        self.sum_abs_per_error += sum_abs_per_error  # type: ignore
        self.num_obs += num_obs  # type: ignore

    def _compute_metric(self) -> Array:
        """Compute the Mean Absolute Percentage Error."""
        return _mean_absolute_percentage_error_compute(
            self.sum_abs_per_error,  # type: ignore
            self.num_obs,  # type: ignore
        )
