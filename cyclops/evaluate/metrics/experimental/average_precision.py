"""Classes for computing area under the Average Precision (AUPRC)."""

from typing import Any, List, Literal, Optional, Tuple, Union

from cyclops.evaluate.metrics.experimental.functional.average_precision import (
    _binary_average_precision_compute,
    _multiclass_average_precision_compute,
    _multiclass_average_precision_validate_args,
    _multilabel_average_precision_compute,
    _multilabel_average_precision_validate_args,
)
from cyclops.evaluate.metrics.experimental.precision_recall_curve import (
    BinaryPrecisionRecallCurve,
    MulticlassPrecisionRecallCurve,
    MultilabelPrecisionRecallCurve,
)
from cyclops.evaluate.metrics.experimental.utils.ops import dim_zero_cat
from cyclops.evaluate.metrics.experimental.utils.types import Array


class BinaryAveragePrecision(
    BinaryPrecisionRecallCurve,
    registry_key="binary_average_precision",
):
    """A summary of the precision-recall curve via a weighted mean of the points.

    The average precision score summarizes a precision-recall curve as the weighted
    mean of precisions achieved at each threshold, with the increase in recall from
    the previous threshold used as the weight.

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
        The value in `target` that should be ignored when computing the average
        precision. If `None`, all values in `target` are used.
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

    name: str = "Average Precision Score"

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


class MulticlassAveragePrecision(
    MulticlassPrecisionRecallCurve,
    registry_key="multiclass_average_precision",
):
    """A summary of the precision-recall curve via a weighted mean of the points.

    The average precision score summarizes a precision-recall curve as the weighted
    mean of precisions achieved at each threshold, with the increase in recall from
    the previous threshold used as the weight.

    Parameters
    ----------
    num_classes : int
        The number of classes in the classification problem.
    thresholds : Union[int, List[float], Array], optional, default=None
        The thresholds to use for computing the average precision score. Can be one
        of the following:
        - `None`: use all unique values in `preds` as thresholds.
        - `int`: use `int` (larger than 1) uniformly spaced thresholds in the range
          [0, 1].
        - `List[float]`: use the values in the list as bins for the thresholds.
        - `Array`: use the values in the Array as bins for the thresholds. The
          array must be 1D.
    average : {"macro", "weighted", "none"}, optional, default="macro"
        The type of averaging to use for computing the average precision score. Can
        be one of the following:
        - `"macro"`: compute the average precision score for each class and average
            over the classes.
        - `"weighted"`: computes the average of the precision for each class and
            average over the classwise scores using the support of each class as
            weights.
        - `"none"`: do not average over the classwise scores.
    ignore_index : int or Tuple[int], optional, default=None
        The value(s) in `target` that should be ignored when computing the average
        precision score. If `None`, all values in `target` are used.
    **kwargs : Any
        Additional keyword arguments to pass to the `Metric` base class.

    Examples
    --------
    >>> import numpy.array_api as anp
    >>> from cyclops.evaluate.metrics.experimental import MulticlassAveragePrecision
    >>> target = anp.asarray([0, 1, 2, 0, 1, 2])
    >>> preds = anp.asarray(
    ...     [
    ...         [0.11, 0.22, 0.67],
    ...         [0.84, 0.73, 0.12],
    ...         [0.33, 0.92, 0.44],
    ...         [0.11, 0.22, 0.67],
    ...         [0.84, 0.73, 0.12],
    ...         [0.33, 0.92, 0.44],
    ...     ]
    ... )
    >>> metric = MulticlassAveragePrecision(
    ...     num_classes=3,
    ...     thresholds=None,
    ...     average=None,
    ... )
    >>> metric(target, preds)
    Array([0.33333334, 0.5       , 0.5       ], dtype=float32)

    """

    name: str = "Average Precision Score"

    def __init__(
        self,
        num_classes: int,
        thresholds: Optional[Union[int, List[float], Array]] = None,
        average: Optional[Literal["macro", "weighted", "none"]] = "macro",
        ignore_index: Optional[Union[int, Tuple[int]]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a `MulticlassAveragePrecision` instance."""
        super().__init__(num_classes, thresholds, ignore_index=ignore_index, **kwargs)
        _multiclass_average_precision_validate_args(
            num_classes,
            thresholds=thresholds,
            average=average,
            ignore_index=ignore_index,
        )
        self.average = average  # type: ignore[assignment]

    def _compute_metric(self) -> Array:  # type: ignore[override]
        """Compute the metric."""
        state = (
            (dim_zero_cat(self.target), dim_zero_cat(self.preds))  # type: ignore[attr-defined]
            if self.thresholds is None
            else self.confmat  # type: ignore[attr-defined]
        )

        return _multiclass_average_precision_compute(
            state,
            self.num_classes,
            thresholds=self.thresholds,  # type: ignore[arg-type]
            average=self.average,  # type: ignore[arg-type]
        )


class MultilabelAveragePrecision(
    MultilabelPrecisionRecallCurve,
    registry_key="multilabel_average_precision",
):
    """A summary of the precision-recall curve via a weighted mean of the points.

    The average precision score summarizes a precision-recall curve as the weighted
    mean of precisions achieved at each threshold, with the increase in recall from
    the previous threshold used as the weight.

    Parameters
    ----------
    num_labels : int
        The number of labels in the multilabel classification problem.
    thresholds : Union[int, List[float], Array], optional, default=None
        The thresholds to use for computing the average precision score. Can be one
        of the following:
        - `None`: use all unique values in `preds` as thresholds.
        - `int`: use `int` (larger than 1) uniformly spaced thresholds in the range
          [0, 1].
        - `List[float]`: use the values in the list as bins for the thresholds.
        - `Array`: use the values in the Array as bins for the thresholds. The
          array must be 1D.
    average : {"micro", "macro", "weighted", "none"}, optional, default="macro"
        The type of averaging to use for computing the average precision score. Can
        be one of the following:
        - `"micro"`: computes the average precision score globally by summing over
            the average precision scores for each label.
        - `"macro"`: compute the average precision score for each label and average
            over the labels.
        - `"weighted"`: computes the average of the precision for each label and
            average over the labelwise scores using the support of each label as
            weights.
        - `"none"`: do not average over the labelwise scores.
    ignore_index : int, optional, default=None
        The value in `target` that should be ignored when computing the average
        precision score. If `None`, all values in `target` are used.
    **kwargs : Any
        Additional keyword arguments to pass to the `Metric` base class.

    Examples
    --------
    >>> import numpy.array_api as anp
    >>> from cyclops.evaluate.metrics.experimental import MultilabelAveragePrecision
    >>> target = anp.asarray([[0, 1, 0], [1, 1, 0], [0, 0, 1]])
    >>> preds = anp.asarray(
    ...     [[0.11, 0.22, 0.67], [0.84, 0.73, 0.12], [0.33, 0.92, 0.44]],
    ... )
    >>> metric = MultilabelAveragePrecision(
    ...     num_labels=3,
    ...     thresholds=None,
    ...     average=None,
    ... )
    >>> metric(target, preds)
    Array([1.       , 0.5833334, 0.5      ], dtype=float32)
    """

    name: str = "Average Precision Score"

    def __init__(
        self,
        num_labels: int,
        thresholds: Optional[Union[int, List[float], Array]] = None,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a `MultilabelAveragePrecision` instance."""
        super().__init__(
            num_labels,
            thresholds=thresholds,
            ignore_index=ignore_index,
            **kwargs,
        )
        _multilabel_average_precision_validate_args(
            num_labels,
            thresholds=thresholds,
            average=average,
            ignore_index=ignore_index,
        )
        self.average = average

    def _compute_metric(self) -> Array:  # type: ignore[override]
        """Compute the metric."""
        state = (
            (dim_zero_cat(self.target), dim_zero_cat(self.preds))  # type: ignore[attr-defined]
            if self.thresholds is None
            else self.confmat  # type: ignore[attr-defined]
        )

        return _multilabel_average_precision_compute(
            state,
            self.num_labels,
            thresholds=self.thresholds,  # type: ignore[arg-type]
            average=self.average,
            ignore_index=self.ignore_index,
        )
