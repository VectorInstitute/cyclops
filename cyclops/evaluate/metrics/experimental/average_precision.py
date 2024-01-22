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
    thresholds : int or list of floats or numpy.ndarray of floats, default=None
        Thresholds used for computing the precision and recall scores.
        If int, then the number of thresholds to use.
        If list or numpy.ndarray, then the thresholds to use.
        If None, then the thresholds are automatically determined by the
        unique values in ``preds``.
    pos_label : int
        The label of the positive class.

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
    >>> target = anp.asarray([[0, 1, 0, 1], [1, 1, 0, 0]])
    >>> preds = anp.asarray([[0.1, 0.4, 0.35, 0.8], [0.6, 0.3, 0.1, 0.7]])
    >>> for t, p in zip(target, preds):
    ...     metric.update(t, p)
    >>> metric.compute()
    Array(0.5833333333333333, dtype=float32)

    """

    name: str = "Average Precision"

    def _compute_metric(self) -> Array:
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
