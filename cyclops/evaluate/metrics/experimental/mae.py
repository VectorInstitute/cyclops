"""Mean Absolute Error metric."""

from typing import Any

from cyclops.evaluate.metrics.experimental.functional.mae import (
    _mean_absolute_error_compute,
    _mean_absolute_error_update,
)
from cyclops.evaluate.metrics.experimental.metric import Metric
from cyclops.evaluate.metrics.experimental.utils.types import Array


class MeanAbsoluteError(Metric):
    """Mean Absolute Error.

    Parameters
    ----------
    **kwargs : Any
        Keyword arguments to pass to the `Metric` base class.

    Examples
    --------
    >>> import numpy.array_api as anp
    >>> from cyclops.evaluate.metrics.experimental import MeanAbsoluteError
    >>> target = anp.asarray([0.009, 1.05, 2.0, 3.0])
    >>> preds = anp.asarray([0.0, 1.0, 2.0, 2.0])
    >>> metric = MeanAbsoluteError()
    >>> metric(target, preds)
    Array(0.26475, dtype=float32)

    """

    name: str = "Mean Absolute Error"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state_default_factory(
            "sum_abs_error",
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
        sum_abs_error, num_obs = _mean_absolute_error_update(target, preds)
        self.sum_abs_error += sum_abs_error  # type: ignore
        self.num_obs += num_obs  # type: ignore

    def _compute_metric(self) -> Array:
        """Compute the Mean Absolute Error."""
        return _mean_absolute_error_compute(self.sum_abs_error, self.num_obs)  # type: ignore
