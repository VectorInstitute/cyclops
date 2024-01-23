"""Classes for computing area under the Average Precision (AUPRC)."""

from cyclops.evaluate.metrics.experimental.functional.average_precision import (
    _binary_average_precision_compute,
)
from cyclops.evaluate.metrics.experimental.precision_recall_curve import (
    BinaryPrecisionRecallCurve,
)
from cyclops.evaluate.metrics.experimental.utils.ops import dim_zero_cat
from cyclops.evaluate.metrics.experimental.utils.types import Array


class BinaryAveragePrecision(
    BinaryPrecisionRecallCurve,
    registry_key="binary_average_precision",
):
    """Compute average precision for binary input.

    Parameters
    ----------
    thresholds : Union[int, List[float], Array], optional, default=None
        The thresholds to use for computing the precision and recall. Can be one
        of the following:
        - `None`: use all unique values in `preds` as thresholds.
        - `int`: use `int` (larger than 1) uniformly spaced thresholds in the range
          [0, 1].
        - `List[float]`: use the values in the list as bins for the thresholds.
        - `Array`: use the values in the Array as bins for the thresholds. The
          array must be 1D.
    ignore_index : int, optional, default=None
        The value in `target` that should be ignored when computing the precision
        and recall. If `None`, all values in `target` are used.
    **kwargs : Any
        Additional keyword arguments to pass to the `Metric` base class.

    Examples
    --------
    >>> import numpy.array_api as anp
    >>> from cyclops.evaluate.metrics.experimental import BinaryAveragePrecision
    >>> target = anp.asarray([0, 1, 0, 1])
    >>> preds = anp.asarray([0.1, 0.4, 0.35, 0.8])
    >>> metric = BinaryAveragePrecision(thresholds=3)
    >>> metric(target, preds)
    Array(0.75, dtype=float32)
    >>> metric.reset()
    >>> target = [[0, 1, 0, 1], [1, 1, 0, 0]]
    >>> preds = [[0.1, 0.4, 0.35, 0.8], [0.6, 0.3, 0.1, 0.7]]
    >>> for t, p in zip(target, preds):
    ...     metric.update(anp.asarray(t), anp.asarray(p))
    >>> metric.compute()
    Array(0.5833334, dtype=float32)

    """

    name: str = "Average Precision"

    def _compute_metric(self) -> Array:  # type: ignore[override]
        """Compute the metric."""
        state = (
            (dim_zero_cat(self.target), dim_zero_cat(self.preds))  # type: ignore[attr-defined]
            if self.thresholds is None
            else self.confmat  # type: ignore[attr-defined]
        )

        return _binary_average_precision_compute(
            state,
            self.thresholds,  # type: ignore
            pos_label=1,
        )
