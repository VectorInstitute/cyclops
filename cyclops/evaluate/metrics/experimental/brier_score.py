"""Brier score metric."""

from typing import Any, Optional

from cyclops.evaluate.metrics.experimental.functional.brier_score import (
    _binary_brier_score_compute,
    _binary_brier_score_format_arrays,
    _binary_brier_score_update,
    _binary_brier_score_validate_args,
    _binary_brier_score_validate_arrays,
    _multiclass_brier_score_format_arrays,
    _multiclass_brier_score_update,
    _multiclass_brier_score_validate_args,
    _multiclass_brier_score_validate_arrays,
)
from cyclops.evaluate.metrics.experimental.metric import Metric
from cyclops.evaluate.metrics.experimental.utils.types import Array


class BinaryBrierScore(Metric):
    """Brier score for binary classification tasks.

    The Brier score is the mean squared error between predicted
    probabilities and the (binary) target, and is a proper scoring rule
    for probabilistic predictions - it rewards models whose predicted
    probabilities are well-calibrated, not just well-ranked, which matters
    for clinical risk scores that are acted on directly.

    Parameters
    ----------
    ignore_index : int, optional, default=None
        Values in the target array to ignore when computing the metric.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> import numpy.array_api as anp
    >>> from cyclops.evaluate.metrics.experimental import BinaryBrierScore
    >>> target = anp.asarray([0, 1, 1, 0])
    >>> preds = anp.asarray([0.1, 0.9, 0.8, 0.3])
    >>> metric = BinaryBrierScore()
    >>> metric(target, preds)
    Array(0.0375, dtype=float32)

    """

    name: str = "Brier Score"

    def __init__(self, ignore_index: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        _binary_brier_score_validate_args(ignore_index=ignore_index)
        self.ignore_index = ignore_index

        self.add_state_default_factory(
            "sum_squared_error",
            lambda xp: xp.asarray(0.0, dtype=xp.float32, device=self.device),  # type: ignore
            dist_reduce_fn="sum",
        )
        self.add_state_default_factory(
            "num_obs",
            lambda xp: xp.asarray(0.0, dtype=xp.float32, device=self.device),  # type: ignore
            dist_reduce_fn="sum",
        )

    def _update_state(self, target: Array, preds: Array) -> None:
        """Update the state of the metric."""
        xp = _binary_brier_score_validate_arrays(
            target,
            preds,
            ignore_index=self.ignore_index,
        )
        target, preds = _binary_brier_score_format_arrays(
            target,
            preds,
            self.ignore_index,
            xp=xp,
        )
        sum_squared_error, num_obs = _binary_brier_score_update(target, preds)
        self.sum_squared_error += sum_squared_error  # type: ignore
        self.num_obs += num_obs  # type: ignore

    def _compute_metric(self) -> Array:
        """Compute the binary Brier score."""
        return _binary_brier_score_compute(
            self.sum_squared_error,  # type: ignore
            self.num_obs,  # type: ignore
        )


class MulticlassBrierScore(Metric):
    """Brier score for multiclass classification tasks.

    Computed as the mean squared error between the predicted probability
    vector for each sample and the one-hot encoded target.

    Parameters
    ----------
    num_classes : int
        The number of classes in the classification task.
    ignore_index : int, optional, default=None
        Values in the target array to ignore when computing the metric.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> import numpy.array_api as anp
    >>> from cyclops.evaluate.metrics.experimental import MulticlassBrierScore
    >>> target = anp.asarray([0, 1, 2])
    >>> preds = anp.asarray(
    ...     [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]],
    ... )
    >>> metric = MulticlassBrierScore(num_classes=3)
    >>> metric(target, preds)
    Array(0.14666666, dtype=float32)

    """

    name: str = "Brier Score"

    def __init__(
        self,
        num_classes: int,
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        _multiclass_brier_score_validate_args(num_classes, ignore_index=ignore_index)
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.add_state_default_factory(
            "sum_squared_error",
            lambda xp: xp.asarray(0.0, dtype=xp.float32, device=self.device),  # type: ignore
            dist_reduce_fn="sum",
        )
        self.add_state_default_factory(
            "num_obs",
            lambda xp: xp.asarray(0.0, dtype=xp.float32, device=self.device),  # type: ignore
            dist_reduce_fn="sum",
        )

    def _update_state(self, target: Array, preds: Array) -> None:
        """Update the state of the metric."""
        xp = _multiclass_brier_score_validate_arrays(target, preds, self.num_classes)
        target, preds = _multiclass_brier_score_format_arrays(
            target,
            preds,
            self.ignore_index,
            self.num_classes,
            xp=xp,
        )
        sum_squared_error, num_obs = _multiclass_brier_score_update(target, preds)
        self.sum_squared_error += sum_squared_error  # type: ignore
        self.num_obs += num_obs  # type: ignore

    def _compute_metric(self) -> Array:
        """Compute the multiclass Brier score."""
        return _binary_brier_score_compute(
            self.sum_squared_error,  # type: ignore
            self.num_obs,  # type: ignore
        )
