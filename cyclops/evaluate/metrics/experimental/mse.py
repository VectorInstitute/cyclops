"""Mean Squared Error metric."""

from typing import Any

from cyclops.evaluate.metrics.experimental.functional.mse import (
    _mean_squared_error_compute,
    _mean_squared_error_update,
)
from cyclops.evaluate.metrics.experimental.metric import Metric
from cyclops.evaluate.metrics.experimental.utils.types import Array


class MeanSquaredError(Metric):
    """Mean Squared Error.

    Parameters
    ----------
    squared : bool, optional, default=True
        Whether to return mean squared error or root mean squared error. If set
        to `False`, returns the root mean squared error.
    num_outputs : int, optional, default=1
        Number of outputs in multioutput setting.
    **kwargs : Any
        Additional keyword arguments to pass to the `Metric` base class.

    Examples
    --------
    >>> import numpy.array_api as anp
    >>> from cyclops.evaluate.metrics.experimental import MeanSquaredError
    >>> target = anp.asarray([0.009, 1.05, 2.0, 3.0])
    >>> preds = anp.asarray([0.0, 1.0, 2.0, 2.0])
    >>> metric = MeanSquaredError()
    >>> metric(target, preds)
    Array(0.25064525, dtype=float32)
    >>> metric = MeanSquaredError(squared=False)
    >>> metric(target, preds)
    Array(0.50064486, dtype=float32)
    >>> metric = MeanSquaredError(num_outputs=2)
    >>> target = anp.asarray([[0.009, 1.05], [2.0, 3.0]])
    >>> preds = anp.asarray([[0.0, 1.0], [2.0, 2.0]])
    >>> metric(target, preds)
    Array([4.0500e-05, 5.0125e-01], dtype=float32)
    >>> metric = MeanSquaredError(squared=False, num_outputs=2)
    >>> metric(target, preds)
    Array([0.00636396, 0.7079901 ], dtype=float32)

    """

    name: str = "Mean Squared Error"

    def __init__(
        self,
        squared: bool = True,
        num_outputs: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(squared, bool):
            raise TypeError(f"Expected `squared` to be a boolean. Got {type(squared)}")
        if not isinstance(num_outputs, int) and num_outputs > 0:
            raise TypeError(
                f"Expected `num_outputs` to be a positive integer. Got {type(num_outputs)}",
            )
        self.num_outputs = num_outputs
        self.squared = squared

        self.add_state_default_factory(
            "sum_squared_error",
            lambda xp: xp.zeros(num_outputs, dtype=xp.float32, device=self.device),  # type: ignore
            dist_reduce_fn="sum",
        )
        self.add_state_default_factory(
            "num_obs",
            lambda xp: xp.asarray(0.0, dtype=xp.float32, device=self.device),  # type: ignore
            dist_reduce_fn="sum",
        )

    def _update_state(self, target: Array, preds: Array) -> None:
        """Update state of metric."""
        sum_squared_error, num_obs = _mean_squared_error_update(
            target,
            preds,
            self.num_outputs,
        )
        self.sum_squared_error += sum_squared_error  # type: ignore
        self.num_obs += num_obs  # type: ignore

    def _compute_metric(self) -> Array:
        """Compute the Mean Squared Error."""
        return _mean_squared_error_compute(
            self.sum_squared_error,  # type: ignore
            self.num_obs,  # type: ignore
            self.squared,
        )
