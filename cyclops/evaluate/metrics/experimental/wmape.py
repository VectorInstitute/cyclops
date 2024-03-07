"""Weighted Mean Absolute Percentage Error metric."""

from types import ModuleType
from typing import Any

from cyclops.evaluate.metrics.experimental.functional.wmape import (
    _weighted_mean_absolute_percentage_error_compute,
    _weighted_mean_absolute_percentage_error_update,
)
from cyclops.evaluate.metrics.experimental.metric import Metric
from cyclops.evaluate.metrics.experimental.utils.types import Array


class WeightedMeanAbsolutePercentageError(Metric):
    """Weighted Mean Absolute Percentage Error.

    Parameters
    ----------
    epsilon : float, optional, default=1.17e-6
        Specifies the lower bound for target values. Any target value below epsilon
        is set to epsilon (avoids division by zero errors).
    **kwargs : Any
        Keyword arguments to pass to the `Metric` base class.

    Examples
    --------
    >>> import numpy.array_api as anp
    >>> from cyclops.evaluate.metrics.experimental import (
    ...     WeightedMeanAbsolutePercentageError,
    ... )
    >>> target = anp.asarray([0.009, 1.05, 2.0, 3.0])
    >>> preds = anp.asarray([0.0, 1.0, 2.0, 2.0])
    >>> metric = WeightedMeanAbsolutePercentageError()
    >>> metric(target, preds)
    Array(0.17478132, dtype=float32)

    """

    name: str = "Weighted Mean Absolute Percentage Error"

    def __init__(self, epsilon: float = 1.17e-6, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not isinstance(epsilon, float):
            raise TypeError(f"Expected `epsilon` to be a float. Got {type(epsilon)}")
        self.epsilon = epsilon

        def default_factory(*, xp: ModuleType) -> Array:
            return xp.asarray(0.0, dtype=xp.float32, device=self.device)  # type: ignore[no-any-return]

        self.add_state_default_factory(
            "sum_abs_error",
            default_factory=default_factory,  # type: ignore
            dist_reduce_fn="sum",
        )
        self.add_state_default_factory(
            "sum_scale",
            default_factory=default_factory,  # type: ignore
            dist_reduce_fn="sum",
        )

    def _update_state(self, target: Array, preds: Array) -> None:
        """Update state of metric."""
        sum_abs_error, sum_scale = _weighted_mean_absolute_percentage_error_update(
            target,
            preds,
        )
        self.sum_abs_error += sum_abs_error  # type: ignore
        self.sum_scale += sum_scale  # type: ignore

    def _compute_metric(self) -> Array:
        """Compute the Weighted Mean Absolute Percentage Error."""
        return _weighted_mean_absolute_percentage_error_compute(
            self.sum_abs_error,  # type: ignore
            self.sum_scale,  # type: ignore
            self.epsilon,
        )
