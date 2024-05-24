"""Classes for computing the area under the ROC curve."""

from typing import Any, List, Literal, Optional, Tuple, Union

from cyclops.evaluate.metrics.experimental.functional.auroc import (
    _binary_auroc_compute,
    _binary_auroc_validate_args,
    _multiclass_auroc_compute,
    _multiclass_auroc_validate_args,
    _multilabel_auroc_compute,
    _multilabel_auroc_validate_args,
)
from cyclops.evaluate.metrics.experimental.precision_recall_curve import (
    BinaryPrecisionRecallCurve,
    MulticlassPrecisionRecallCurve,
    MultilabelPrecisionRecallCurve,
)
from cyclops.evaluate.metrics.experimental.utils.ops import dim_zero_cat
from cyclops.evaluate.metrics.experimental.utils.types import Array


class BinaryAUROC(BinaryPrecisionRecallCurve, registry_key="binary_auroc"):
    """Area under the Receiver Operating Characteristic (ROC) curve.

    Parameters
    ----------
    max_fpr : float, optional, default=None
        If not `None`, computes the maximum area under the curve up to the given
        false positive rate value. Must be a float in the range (0, 1].
    thresholds : Union[int, List[float], Array], optional, default=None
        The thresholds to use for computing the ROC curve. Can be one of the following:
        - `None`: use all unique values in `preds` as thresholds.
        - `int`: use `int` (larger than 1) uniformly spaced thresholds in the range
          [0, 1].
        - `List[float]`: use the values in the list as bins for the thresholds.
        - `Array`: use the values in the Array as bins for the thresholds. The
          array must be 1D.
    ignore_index : int, optional, default=None
        The value in `target` that should be ignored when computing the AUROC.
        If `None`, all values in `target` are used.
    **kwargs : Any
        Additional keyword arguments that are common to all metrics.

    Examples
    --------
    >>> import numpy.array_api as anp
    >>> from cyclops.evaluate.metrics.experimental import BinaryAUROC
    >>> target = anp.asarray([0, 1, 1, 0, 1, 0, 0, 1])
    >>> preds = anp.asarray([0.1, 0.4, 0.35, 0.8, 0.2, 0.6, 0.7, 0.3])
    >>> auroc = BinaryAUROC(thresholds=None)
    >>> auroc(target, preds)
    Array(0.25, dtype=float32)
    >>> auroc = BinaryAUROC(thresholds=5)
    >>> auroc(target, preds)
    Array(0.21875, dtype=float32)
    """

    name: str = "AUC ROC Curve"

    def __init__(
        self,
        max_fpr: Optional[float] = None,
        thresholds: Optional[Union[int, List[float], Array]] = None,
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the BinaryAUROC metric."""
        super().__init__(thresholds=thresholds, ignore_index=ignore_index, **kwargs)
        _binary_auroc_validate_args(
            max_fpr=max_fpr,
            thresholds=thresholds,
            ignore_index=ignore_index,
        )
        self.max_fpr = max_fpr

    def _compute_metric(self) -> Array:  # type: ignore[override]
        """Compute the AUROC."""
        state = (
            (dim_zero_cat(self.target), dim_zero_cat(self.preds))  # type: ignore[attr-defined]
            if self.thresholds is None
            else self.confmat  # type: ignore[attr-defined]
        )
        return _binary_auroc_compute(
            state,
            thresholds=self.thresholds,  # type: ignore
            max_fpr=self.max_fpr,
        )


class MulticlassAUROC(MulticlassPrecisionRecallCurve, registry_key="multiclass_auroc"):
    """Area under the Receiver Operating Characteristic (ROC) curve.

    Parameters
    ----------
    num_classes : int
        The number of classes in the classification problem.
    thresholds : Union[int, List[float], Array], optional, default=None
        The thresholds to use for computing the ROC curve. Can be one of the following:
        - `None`: use all unique values in `preds` as thresholds.
        - `int`: use `int` (larger than 1) uniformly spaced thresholds in the range
          [0, 1].
        - `List[float]`: use the values in the list as bins for the thresholds.
        - `Array`: use the values in the Array as bins for the thresholds. The
          array must be 1D.
    average : {"macro", "weighted", "none"}, optional, default="macro"
        The type of averaging to use for computing the AUROC. Can be one of
        the following:
        - `"macro"`: interpolates the curves from each class at a combined set of
          thresholds and then average over the classwise interpolated curves.
        - `"weighted"`: average over the classwise curves weighted by the support
          (the number of true instances for each class).
        - `"none"`: do not average over the classwise curves.
    ignore_index : int or Tuple[int], optional, default=None
        The value(s) in `target` that should be ignored when computing the AUROC.
        If `None`, all values in `target` are used.
    **kwargs : Any
        Additional keyword arguments that are common to all metrics.

    Examples
    --------
    >>> import numpy.array_api as anp
    >>> from cyclops.evaluate.metrics.experimental import MulticlassAUROC
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
    >>> auroc = MulticlassAUROC(num_classes=3, average="macro", thresholds=None)
    >>> auroc(target, preds)
    Array(0.33333334, dtype=float32)
    >>> auroc = MulticlassAUROC(num_classes=3, average=None, thresholds=None)
    >>> auroc(target, preds)
    Array([0. , 0.5, 0.5], dtype=float32)
    >>> auroc = MulticlassAUROC(num_classes=3, average="macro", thresholds=5)
    >>> auroc(target, preds)
    Array(0.33333334, dtype=float32)
    >>> auroc = MulticlassAUROC(num_classes=3, average=None, thresholds=5)
    >>> auroc(target, preds)
    Array([0. , 0.5, 0.5], dtype=float32)
    """

    name: str = "AUC ROC Curve"

    def __init__(
        self,
        num_classes: int,
        thresholds: Optional[Union[int, List[float], Array]] = None,
        average: Optional[Literal["macro", "weighted", "none"]] = "macro",
        ignore_index: Optional[Union[int, Tuple[int]]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the MulticlassAUROC metric."""
        super().__init__(
            num_classes,
            thresholds=thresholds,
            ignore_index=ignore_index,
            **kwargs,
        )
        _multiclass_auroc_validate_args(
            num_classes=num_classes,
            thresholds=thresholds,
            average=average,
            ignore_index=ignore_index,
        )
        self.average = average  # type: ignore[assignment]

    def _compute_metric(self) -> Array:  # type: ignore[override]
        """Compute the AUROC."""
        state = (
            (dim_zero_cat(self.target), dim_zero_cat(self.preds))  # type: ignore[attr-defined]
            if self.thresholds is None
            else self.confmat  # type: ignore[attr-defined]
        )
        return _multiclass_auroc_compute(
            state,
            self.num_classes,
            thresholds=self.thresholds,  # type: ignore[arg-type]
            average=self.average,  # type: ignore[arg-type]
        )


class MultilabelAUROC(MultilabelPrecisionRecallCurve, registry_key="multilabel_auroc"):
    """Area under the Receiver Operating Characteristic (ROC) curve.

    Parameters
    ----------
    num_labels : int
        The number of labels in the multilabel classification problem.
    thresholds : Union[int, List[float], Array], optional, default=None
        The thresholds to use for computing the ROC curve. Can be one of the following:
        - `None`: use all unique values in `preds` as thresholds.
        - `int`: use `int` (larger than 1) uniformly spaced thresholds in the range
          [0, 1].
        - `List[float]`: use the values in the list as bins for the thresholds.
        - `Array`: use the values in the Array as bins for the thresholds. The
          array must be 1D.
    average : {"micro", "macro", "weighted", "none"}, optional, default="macro"
        The type of averaging to use for computing the AUROC. Can be one of
        the following:
        - `"micro"`: compute the AUROC globally by considering each element of the
            label indicator matrix as a label.
        - `"macro"`: compute the AUROC for each label and average them.
        - `"weighted"`: compute the AUROC for each label and average them weighted
            by the support (the number of true instances for each label).
        - `"none"`: do not average over the labelwise AUROC.
    ignore_index : int, optional, default=None
        The value in `target` that should be ignored when computing the AUROC.
        If `None`, all values in `target` are used.
    **kwargs : Any
        Additional keyword arguments that are common to all metrics.

    Examples
    --------
    >>> import numpy.array_api as anp
    >>> from cyclops.evaluate.metrics.experimental import MultilabelAUROC
    >>> target = anp.asarray([[0, 1, 0], [1, 1, 0], [0, 0, 1]])
    >>> preds = anp.asarray(
    ...     [[0.11, 0.22, 0.67], [0.84, 0.73, 0.12], [0.33, 0.92, 0.44]],
    ... )
    >>> auroc = MultilabelAUROC(num_labels=3, average="macro", thresholds=None)
    >>> auroc(target, preds)
    Array(0.5, dtype=float32)
    >>> auroc = MultilabelAUROC(num_labels=3, average=None, thresholds=None)
    >>> auroc(target, preds)
    Array([1. , 0. , 0.5], dtype=float32)
    >>> auroc = MultilabelAUROC(num_labels=3, average="macro", thresholds=5)
    >>> auroc(target, preds)
    Array(0.5, dtype=float32)
    >>> auroc = MultilabelAUROC(num_labels=3, average=None, thresholds=5)
    >>> auroc(target, preds)
    Array([1. , 0. , 0.5], dtype=float32)

    """

    name: str = "AUC ROC Curve"

    def __init__(
        self,
        num_labels: int,
        thresholds: Optional[Union[int, List[float], Array]] = None,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the MultilabelAUROC metric."""
        super().__init__(
            num_labels,
            thresholds=thresholds,
            ignore_index=ignore_index,
            **kwargs,
        )
        _multilabel_auroc_validate_args(
            num_labels=num_labels,
            thresholds=thresholds,
            average=average,
            ignore_index=ignore_index,
        )
        self.average = average

    def _compute_metric(self) -> Array:  # type: ignore[override]
        """Compute the AUROC."""
        state = (
            (dim_zero_cat(self.target), dim_zero_cat(self.preds))  # type: ignore[attr-defined]
            if self.thresholds is None
            else self.confmat  # type: ignore[attr-defined]
        )
        return _multilabel_auroc_compute(
            state,
            self.num_labels,
            thresholds=self.thresholds,  # type: ignore[arg-type]
            average=self.average,
            ignore_index=self.ignore_index,
        )
