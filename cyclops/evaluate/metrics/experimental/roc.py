"""Classes for computing the Receiver Operating Characteristic (ROC) curve."""

from typing import List, Tuple, Union

from cyclops.evaluate.metrics.experimental.functional.roc import (
    ROCCurve,
    _binary_roc_compute,
    _multiclass_roc_compute,
    _multilabel_roc_compute,
)
from cyclops.evaluate.metrics.experimental.precision_recall_curve import (
    BinaryPrecisionRecallCurve,
    MulticlassPrecisionRecallCurve,
    MultilabelPrecisionRecallCurve,
)
from cyclops.evaluate.metrics.experimental.utils.ops import dim_zero_cat
from cyclops.evaluate.metrics.experimental.utils.types import Array


class BinaryROC(BinaryPrecisionRecallCurve, registry_key="binary_roc_curve"):
    """The receiver operating characteristic (ROC) curve.

    Parameters
    ----------
    thresholds : Union[int, List[float], Array], optional, default=None
        The thresholds to use for computing the ROC curve. Can be one of the following:
        - `None`: use all unique values in `preds` as thresholds.
        - `int`: use `int` (larger than 1) uniformly spaced thresholds in the range
          [0, 1].
        - `List[float]`: use the values in the list as bins for the thresholds.
        - `Array`: use the values in the Array as bins for the thresholds. The
          array must be 1D.
    ignore_index : int, optional, default=None
        The value in `target` that should be ignored when computing the ROC curve.
        If `None`, all values in `target` are used.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> import numpy.array_api as anp
    >>> from cyclops.evaluate.metrics.experimental import BinaryROC
    >>> target = anp.asarray([0, 1, 0, 1, 0, 1])
    >>> preds = anp.asarray([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])
    >>> metric = BinaryROC(thresholds=None)
    >>> metric(target, preds)
    ROCCurve(fpr=Array([0.        , 0.        , 0.33333334, 0.33333334,
           0.6666667 , 0.6666667 , 1.        ], dtype=float32), tpr=Array([0.        , 0.33333334, 0.33333334, 0.6666667 ,
           0.6666667 , 1.        , 1.        ], dtype=float32), thresholds=Array([1.  , 0.92, 0.84, 0.73, 0.33, 0.22, 0.11], dtype=float64))
    >>> metric = BinaryROC(thresholds=5)
    >>> metric(target, preds)
    ROCCurve(fpr=Array([0.        , 0.33333334, 0.33333334, 0.6666667 ,
           1.        ], dtype=float32), tpr=Array([0.        , 0.33333334, 0.6666667 , 0.6666667 ,
           1.        ], dtype=float32), thresholds=Array([1.  , 0.75, 0.5 , 0.25, 0.  ], dtype=float32))

    """  # noqa: W505

    name: str = "ROC Curve"

    def _compute_metric(self) -> ROCCurve:  # type: ignore
        state = (
            (dim_zero_cat(self.target), dim_zero_cat(self.preds))  # type: ignore[attr-defined]
            if self.thresholds is None
            else self.confmat  # type: ignore[attr-defined]
        )
        fpr, tpr, thresholds = _binary_roc_compute(state, self.thresholds)  # type: ignore[arg-type]
        return ROCCurve(fpr, tpr, thresholds)


class MulticlassROC(
    MulticlassPrecisionRecallCurve,
    registry_key="multiclass_roc_curve",
):
    """The receiver operator characteristics (ROC) curve.

    Parameters
    ----------
    num_classes : int
        The number of classes in the classification problem.
    thresholds : Union[int, List[float], Array], optional, default=None
        The thresholds to use for computing the ROC curve. Can be one
        of the following:
        - `None`: use all unique values in `preds` as thresholds.
        - `int`: use `int` (larger than 1) uniformly spaced thresholds in the range
          [0, 1].
        - `List[float]`: use the values in the list as bins for the thresholds.
        - `Array`: use the values in the Array as bins for the thresholds. The
          array must be 1D.
    average : {"macro", "micro", "none"}, optional, default=None
        The type of averaging to use for computing the ROC curve. Can be one of
        the following:
        - `"macro"`: interpolates the curves from each class at a combined set of
          thresholds and then average over the classwise interpolated curves.
        - `"micro"`: one-hot encodes the targets and flattens the predictions,
          considering all classes jointly as a binary problem.
        - `"none"`: do not average over the classwise curves.
    ignore_index : int or Tuple[int], optional, default=None
        The value(s) in `target` that should be ignored when computing the ROC curve.
        If `None`, all values in `target` are used.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> import numpy.array_api as anp
    >>> from cyclops.evaluate.metrics.experimental import MulticlassROC
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
    >>> metric = MulticlassROC(num_classes=3, thresholds=None)
    >>> metric(target, preds)
    ([Array([0. , 0.5, 1. , 1. ], dtype=float32),
    Array([0. , 0.5, 0.5, 1. ], dtype=float32),
    Array([0. , 0.5, 0.5, 1. ], dtype=float32)],
    [Array([0., 0., 0., 1.], dtype=float32),
    Array([0., 0., 1., 1.], dtype=float32),
    Array([0., 0., 1., 1.], dtype=float32)],
    [Array([1.  , 0.84, 0.33, 0.11], dtype=float64),
    Array([1.  , 0.92, 0.73, 0.22], dtype=float64),
    Array([1.  , 0.67, 0.44, 0.12], dtype=float64)])
    >>> metric = MulticlassROC(num_classes=3, thresholds=5)
    >>> metric(target, preds)
    (Array([[0. , 0.5, 0.5, 1. , 1. ],
           [0. , 0.5, 0.5, 0.5, 1. ],
           [0. , 0. , 0.5, 0.5, 1. ]], dtype=float32), Array([[0., 0., 0., 0., 1.],
           [0., 0., 1., 1., 1.],
           [0., 0., 0., 1., 1.]], dtype=float32), Array([1.  , 0.75, 0.5 , 0.25, 0.  ], dtype=float32))

    """  # noqa: W505

    name: str = "ROC Curve"

    def _compute_metric(
        self,
    ) -> Union[
        Tuple[Array, Array, Array],
        Tuple[List[Array], List[Array], List[Array]],
    ]:
        state = (
            (dim_zero_cat(self.target), dim_zero_cat(self.preds))  # type: ignore[attr-defined]
            if self.thresholds is None
            else self.confmat  # type: ignore[attr-defined]
        )
        return _multiclass_roc_compute(
            state,
            self.num_classes,
            self.thresholds,  # type: ignore[arg-type]
            self.average,
        )


class MultilabelROC(
    MultilabelPrecisionRecallCurve,
    registry_key="multilabel_roc_curve",
):
    """The receiver operator characteristics (ROC) curve.

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
    ignore_index : int, optional, default=None
        The value in `target` that should be ignored when computing the ROC Curve.
        If `None`, all values in `target` are used.
    **kwargs
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> import numpy.array_api as anp
    >>> from cyclops.evaluate.metrics.experimental import MultilabelROC
    >>> target = anp.asarray([[0, 1, 0], [1, 1, 0], [0, 0, 1]])
    >>> preds = anp.asarray(
    ...     [[0.11, 0.22, 0.67], [0.84, 0.73, 0.12], [0.33, 0.92, 0.44]],
    ... )
    >>> metric = MultilabelROC(num_labels=3, thresholds=None)
    >>> metric(target, preds)
    ([Array([0. , 0. , 0.5, 1. ], dtype=float32),
    Array([0., 1., 1., 1.], dtype=float32),
    Array([0. , 0.5, 0.5, 1. ], dtype=float32)],
    [Array([0., 1., 1., 1.], dtype=float32),
    Array([0. , 0. , 0.5, 1. ], dtype=float32),
    Array([0., 0., 1., 1.], dtype=float32)],
    [Array([1.  , 0.84, 0.33, 0.11], dtype=float64),
    Array([1.  , 0.92, 0.73, 0.22], dtype=float64),
    Array([1.  , 0.67, 0.44, 0.12], dtype=float64)])
    >>> metric = MultilabelROC(num_labels=3, thresholds=5)
    >>> metric(target, preds)
    (Array([[0. , 0. , 0. , 0.5, 1. ],
           [0. , 1. , 1. , 1. , 1. ],
           [0. , 0. , 0.5, 0.5, 1. ]], dtype=float32), Array([[0. , 1. , 1. , 1. , 1. ],
           [0. , 0. , 0.5, 0.5, 1. ],
           [0. , 0. , 0. , 1. , 1. ]], dtype=float32), Array([1.  , 0.75, 0.5 , 0.25, 0.  ], dtype=float32))

    """  # noqa: W505

    name: str = "ROC Curve"

    def _compute_metric(
        self,
    ) -> Union[
        Tuple[Array, Array, Array],
        Tuple[List[Array], List[Array], List[Array]],
    ]:
        state = (
            (dim_zero_cat(self.target), dim_zero_cat(self.preds))  # type: ignore[attr-defined]
            if self.thresholds is None
            else self.confmat  # type: ignore[attr-defined]
        )
        return _multilabel_roc_compute(
            state,
            self.num_labels,
            self.thresholds,  # type: ignore[arg-type]
            self.ignore_index,
        )
