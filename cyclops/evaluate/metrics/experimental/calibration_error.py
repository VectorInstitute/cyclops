"""Calibration error metric."""

from typing import Any, Literal, Optional

from cyclops.evaluate.metrics.experimental.functional.brier_score import (
    _binary_brier_score_format_arrays,
)
from cyclops.evaluate.metrics.experimental.functional.calibration_error import (
    _binary_calibration_error_compute,
    _binary_calibration_error_update,
    _binary_calibration_error_validate_args,
    _binary_calibration_error_validate_arrays,
)
from cyclops.evaluate.metrics.experimental.metric import Metric
from cyclops.evaluate.metrics.experimental.utils.types import Array


class BinaryCalibrationError(Metric):
    """Calibration error for binary classification tasks.

    Groups predicted probabilities into equal-width bins and measures,
    within each bin, the gap between the average predicted probability
    (confidence) and the observed event rate (accuracy). The default
    `"l1"` norm gives the Expected Calibration Error (ECE), the most
    commonly reported calibration metric.

    A well-calibrated clinical risk model should have a low calibration
    error: among patients given, say, a 30% predicted risk, roughly 30%
    should actually experience the event. This matters even for models
    with good discrimination (e.g. high AUROC), since discrimination
    alone doesn't guarantee predicted probabilities can be trusted at
    face value - which is often how clinical risk scores are actually
    used.

    Parameters
    ----------
    n_bins : int, optional, default=15
        Number of equal-width bins to group predicted probabilities into.
    norm : {'l1', 'l2', 'max'}, optional, default='l1'
        Norm used to aggregate the per-bin calibration gaps. `'l1'` gives
        the Expected Calibration Error (ECE), `'max'` gives the Maximum
        Calibration Error (MCE).
    ignore_index : int, optional, default=None
        Values in the target array to ignore when computing the metric.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> import numpy.array_api as anp
    >>> from cyclops.evaluate.metrics.experimental import BinaryCalibrationError
    >>> target = anp.asarray([0, 1, 1, 0])
    >>> preds = anp.asarray([0.1, 0.9, 0.8, 0.3])
    >>> metric = BinaryCalibrationError(n_bins=2)
    >>> metric(target, preds)
    Array(0.17499998, dtype=float32)

    """

    name: str = "Calibration Error"

    def __init__(
        self,
        n_bins: int = 15,
        norm: Literal["l1", "l2", "max"] = "l1",
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        _binary_calibration_error_validate_args(
            n_bins=n_bins,
            norm=norm,
            ignore_index=ignore_index,
        )
        self.n_bins = n_bins
        self.norm = norm
        self.ignore_index = ignore_index

        self.add_state_default_factory(
            "bin_confidence_sums",
            lambda xp: xp.zeros(n_bins, dtype=xp.float32, device=self.device),  # type: ignore
            dist_reduce_fn="sum",
        )
        self.add_state_default_factory(
            "bin_correct_sums",
            lambda xp: xp.zeros(n_bins, dtype=xp.float32, device=self.device),  # type: ignore
            dist_reduce_fn="sum",
        )
        self.add_state_default_factory(
            "bin_counts",
            lambda xp: xp.zeros(n_bins, dtype=xp.int64, device=self.device),  # type: ignore
            dist_reduce_fn="sum",
        )

    def _update_state(self, target: Array, preds: Array) -> None:
        """Update the state of the metric."""
        xp = _binary_calibration_error_validate_arrays(
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
        bin_confidence_sums, bin_correct_sums, bin_counts = (
            _binary_calibration_error_update(target, preds, self.n_bins, xp=xp)
        )
        self.bin_confidence_sums += bin_confidence_sums  # type: ignore
        self.bin_correct_sums += bin_correct_sums  # type: ignore
        self.bin_counts += bin_counts  # type: ignore

    def _compute_metric(self) -> Array:
        """Compute the binary calibration error."""
        return _binary_calibration_error_compute(
            self.bin_confidence_sums,  # type: ignore
            self.bin_correct_sums,  # type: ignore
            self.bin_counts,  # type: ignore
            norm=self.norm,
        )
