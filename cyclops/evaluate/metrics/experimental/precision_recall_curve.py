"""Classes for computing the precision-recall curve."""

from types import ModuleType
from typing import Any, List, Literal, Optional, Tuple, Union

import array_api_compat as apc

from cyclops.evaluate.metrics.experimental.functional.precision_recall_curve import (
    PRCurve,
    _binary_precision_recall_curve_compute,
    _binary_precision_recall_curve_format_arrays,
    _binary_precision_recall_curve_update,
    _binary_precision_recall_curve_validate_args,
    _binary_precision_recall_curve_validate_arrays,
    _multiclass_precision_recall_curve_compute,
    _multiclass_precision_recall_curve_format_arrays,
    _multiclass_precision_recall_curve_update,
    _multiclass_precision_recall_curve_validate_args,
    _multiclass_precision_recall_curve_validate_arrays,
    _multilabel_precision_recall_curve_compute,
    _multilabel_precision_recall_curve_format_arrays,
    _multilabel_precision_recall_curve_update,
    _multilabel_precision_recall_curve_validate_args,
    _multilabel_precision_recall_curve_validate_arrays,
)
from cyclops.evaluate.metrics.experimental.metric import Metric
from cyclops.evaluate.metrics.experimental.utils.ops import dim_zero_cat
from cyclops.evaluate.metrics.experimental.utils.types import Array


class BinaryPrecisionRecallCurve(Metric, registry_key="binary_precision_recall_curve"):
    """The precision and recall values computed at different thresholds.

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
    >>> from cyclops.evaluate.metrics.experimental import BinaryPrecisionRecallCurve
    >>> target = anp.asarray([0, 1, 0, 1, 0, 1])
    >>> preds = anp.asarray([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])
    >>> metric = BinaryPrecisionRecallCurve(thresholds=None)
    >>> metric(target, preds)
    PRCurve(precision=Array([0.5      , 0.6      , 0.5      , 0.6666667,
           0.5      , 1.       , 1.       ], dtype=float32), recall=Array([1.        , 1.        , 0.6666667 , 0.6666667 ,
           0.33333334, 0.33333334, 0.        ], dtype=float32), thresholds=Array([0.11, 0.22, 0.33, 0.73, 0.84, 0.92], dtype=float64))
    >>> metric = BinaryPrecisionRecallCurve(thresholds=5)
    >>> metric(target, preds)
    PRCurve(precision=Array([0.5      , 0.5      , 0.6666667, 0.5      ,
           0.       , 1.       ], dtype=float32), recall=Array([1.        , 0.6666667 , 0.6666667 , 0.33333334,
           0.        , 0.        ], dtype=float32), thresholds=Array([0.  , 0.25, 0.5 , 0.75, 1.  ], dtype=float32))

    """  # noqa: W505

    name: str = "Precision-Recall Curve"

    def __init__(
        self,
        thresholds: Optional[Union[int, List[float], Array]] = None,
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a `BinaryPrecisionRecallCurve` instance."""
        super().__init__(**kwargs)
        _binary_precision_recall_curve_validate_args(thresholds, ignore_index)
        self.ignore_index = ignore_index
        self.thresholds = thresholds

        if thresholds is None:
            self.add_state_default_factory(
                "preds",
                default_factory=list,  # type: ignore
                dist_reduce_fn="cat",
            )
            self.add_state_default_factory(
                "target",
                default_factory=list,  # type: ignore
                dist_reduce_fn="cat",
            )
        else:
            len_thresholds = (
                len(thresholds)
                if isinstance(thresholds, list)
                else thresholds
                if isinstance(thresholds, int)
                else thresholds.shape[0]
            )

            def default(xp: ModuleType) -> Array:
                return xp.zeros(  # type: ignore
                    (len_thresholds, 2, 2),
                    dtype=xp.int32,
                    device=self.device,
                )

            self.add_state_default_factory(
                "confmat",
                default_factory=default,  # type: ignore
                dist_reduce_fn="sum",
            )

    def _update_state(self, target: Array, preds: Array) -> None:
        """Update the state of the metric."""
        xp = _binary_precision_recall_curve_validate_arrays(
            target,
            preds,
            self.thresholds,
            self.ignore_index,
        )
        target, preds, self.thresholds = _binary_precision_recall_curve_format_arrays(
            target,
            preds,
            thresholds=self.thresholds,
            ignore_index=self.ignore_index,
            xp=xp,
        )
        state = _binary_precision_recall_curve_update(
            target,
            preds,
            thresholds=self.thresholds,
            xp=xp,
        )

        if apc.is_array_api_obj(state):
            self.confmat += state  # type: ignore[attr-defined]
        else:
            self.target.append(state[0])  # type: ignore[attr-defined]
            self.preds.append(state[1])  # type: ignore[attr-defined]

    def _compute_metric(self) -> PRCurve:
        """Compute the metric."""
        state = (
            (dim_zero_cat(self.target), dim_zero_cat(self.preds))  # type: ignore[attr-defined]
            if self.thresholds is None
            else self.confmat  # type: ignore[attr-defined]
        )
        precision, recall, thresholds = _binary_precision_recall_curve_compute(
            state,
            self.thresholds,  # type: ignore
        )
        return PRCurve(precision, recall, thresholds)


class MulticlassPrecisionRecallCurve(
    Metric,
    registry_key="multiclass_precision_recall_curve",
):
    """The precision and recall values computed at different thresholds.

    Parameters
    ----------
    num_classes : int
        The number of classes in the classification problem.
    thresholds : Union[int, List[float], Array], optional, default=None
        The thresholds to use for computing the precision and recall. Can be one
        of the following:
        - `None`: use all unique values in `preds` as thresholds.
        - `int`: use `int` (larger than 1) uniformly spaced thresholds in the range
          [0, 1].
        - `List[float]`: use the values in the list as bins for the thresholds.
        - `Array`: use the values in the Array as bins for the thresholds. The
          array must be 1D.
    average : {"macro", "micro", "none"}, optional, default=None
        The type of averaging to use for computing the precision and recall. Can
        be one of the following:
        - `"macro"`: interpolates the curves from each class at a combined set of
          thresholds and then average over the classwise interpolated curves.
        - `"micro"`: one-hot encodes the targets and flattens the predictions,
          considering all classes jointly as a binary problem.
        - `"none"`: do not average over the classwise curves.
    ignore_index : int or Tuple[int], optional, default=None
        The value(s) in `target` that should be ignored when computing the
        precision and recall. If `None`, all values in `target` are used.
    **kwargs : Any
        Additional keyword arguments to pass to the `Metric` base class.

    Examples
    --------
    >>> import numpy.array_api as anp
    >>> from cyclops.evaluate.metrics.experimental import MulticlassPrecisionRecallCurve
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
    >>> metric = MulticlassPrecisionRecallCurve(num_classes=3, thresholds=None)
    >>> metric(target, preds)
    ([Array([0.33333334, 0.        , 0.        , 1.        ], dtype=float32),
    Array([0.33333334, 0.5       , 0.        , 1.        ], dtype=float32),
    Array([0.33333334, 0.5       , 0.        , 1.        ], dtype=float32)],
    [Array([1., 0., 0., 0.], dtype=float32),
    Array([1., 1., 0., 0.], dtype=float32),
    Array([1., 1., 0., 0.], dtype=float32)],
    [Array([0.11, 0.33, 0.84], dtype=float64),
    Array([0.22, 0.73, 0.92], dtype=float64),
    Array([0.12, 0.44, 0.67], dtype=float64)])
    >>> metric = MulticlassPrecisionRecallCurve(num_classes=3, thresholds=5)
    >>> metric(target, preds)
    (Array([[0.33333334, 0.        , 0.        , 0.        ,
            0.        , 1.        ],
           [0.33333334, 0.5       , 0.5       , 0.        ,
            0.        , 1.        ],
           [0.33333334, 0.5       , 0.        , 0.        ,
            0.        , 1.        ]], dtype=float32), Array([[1., 0., 0., 0., 0., 0.],
           [1., 1., 1., 0., 0., 0.],
           [1., 1., 0., 0., 0., 0.]], dtype=float32), Array([0.  , 0.25, 0.5 , 0.75, 1.  ], dtype=float32))

    """  # noqa: W505

    name: str = "Precision-Recall Curve"

    def __init__(
        self,
        num_classes: int,
        thresholds: Optional[Union[int, List[float], Array]] = None,
        average: Optional[Literal["macro", "micro", "none"]] = None,
        ignore_index: Optional[Union[int, Tuple[int]]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a `MulticlassPrecisionRecallCurve` instance."""
        super().__init__(**kwargs)
        _multiclass_precision_recall_curve_validate_args(
            num_classes,
            thresholds=thresholds,
            average=average,
            ignore_index=ignore_index,
        )
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.average = average
        self.thresholds = thresholds

        if thresholds is None:
            self.add_state_default_factory(
                "preds",
                default_factory=list,  # type: ignore
                dist_reduce_fn="cat",
            )
            self.add_state_default_factory(
                "target",
                default_factory=list,  # type: ignore
                dist_reduce_fn="cat",
            )
        else:
            len_thresholds = (
                len(thresholds)
                if isinstance(thresholds, list)
                else thresholds
                if isinstance(thresholds, int)
                else thresholds.shape[0]
            )

            def default(xp: ModuleType) -> Array:
                return xp.zeros(  # type: ignore[no-any-return]
                    (len_thresholds, num_classes, 2, 2),
                    dtype=xp.int32,
                    device=self.device,
                )

            self.add_state_default_factory(
                "confmat",
                default_factory=default,  # type: ignore
                dist_reduce_fn="sum",
            )

    def _update_state(self, target: Array, preds: Array) -> None:
        """Update the state of the metric."""
        xp = _multiclass_precision_recall_curve_validate_arrays(
            target,
            preds,
            self.num_classes,
            ignore_index=self.ignore_index,
        )

        (
            target,
            preds,
            self.thresholds,
        ) = _multiclass_precision_recall_curve_format_arrays(
            target,
            preds,
            self.num_classes,
            thresholds=self.thresholds,
            ignore_index=self.ignore_index,
            average=self.average,
            xp=xp,
        )
        state = _multiclass_precision_recall_curve_update(
            target,
            preds,
            self.num_classes,
            thresholds=self.thresholds,
            average=self.average,
            xp=xp,
        )

        if apc.is_array_api_obj(state):
            self.confmat += state  # type: ignore[attr-defined]
        else:
            self.target.append(state[0])  # type: ignore[attr-defined]
            self.preds.append(state[1])  # type: ignore[attr-defined]

    def _compute_metric(
        self,
    ) -> Union[
        Tuple[Array, Array, Array],
        Tuple[List[Array], List[Array], List[Array]],
    ]:
        """Compute the metric."""
        state = (
            (dim_zero_cat(self.target), dim_zero_cat(self.preds))  # type: ignore[attr-defined]
            if self.thresholds is None
            else self.confmat  # type: ignore[attr-defined]
        )
        return _multiclass_precision_recall_curve_compute(
            state,
            self.num_classes,
            thresholds=self.thresholds,  # type: ignore[arg-type]
            average=self.average,
        )


class MultilabelPrecisionRecallCurve(
    Metric,
    registry_key="multilabel_precision_recall_curve",
):
    """The precision and recall values computed at different thresholds.

    Parameters
    ----------
    num_labels : int
        The number of labels in the multilabel classification problem.
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
    >>> from cyclops.evaluate.metrics.experimental import MultilabelPrecisionRecallCurve
    >>> target = anp.asarray([[0, 1, 0], [1, 1, 0], [0, 0, 1]])
    >>> preds = anp.asarray(
    ...     [[0.11, 0.22, 0.67], [0.84, 0.73, 0.12], [0.33, 0.92, 0.44]],
    ... )
    >>> metric = MultilabelPrecisionRecallCurve(num_labels=3, thresholds=None)
    >>> metric(target, preds)
    ([Array([0.33333334, 0.5       , 1.        , 1.        ], dtype=float32),
    Array([0.6666667, 0.5      , 0.       , 1.       ], dtype=float32),
    Array([0.33333334, 0.5       , 0.        , 1.        ], dtype=float32)],
    [Array([1., 1., 1., 0.], dtype=float32),
    Array([1. , 0.5, 0. , 0. ], dtype=float32),
    Array([1., 1., 0., 0.], dtype=float32)],
    [Array([0.11, 0.33, 0.84], dtype=float64),
    Array([0.22, 0.73, 0.92], dtype=float64),
    Array([0.12, 0.44, 0.67], dtype=float64)])
    >>> metric = MultilabelPrecisionRecallCurve(num_labels=3, thresholds=5)
    >>> metric(target, preds)
    (Array([[0.33333334, 0.5       , 1.        , 1.        ,
            0.        , 1.        ],
           [0.6666667 , 0.5       , 0.5       , 0.        ,
            0.        , 1.        ],
           [0.33333334, 0.5       , 0.        , 0.        ,
            0.        , 1.        ]], dtype=float32), Array([[1. , 1. , 1. , 1. , 0. , 0. ],
           [1. , 0.5, 0.5, 0. , 0. , 0. ],
           [1. , 1. , 0. , 0. , 0. , 0. ]], dtype=float32), Array([0.  , 0.25, 0.5 , 0.75, 1.  ], dtype=float32))

    """  # noqa: W505

    name: str = "Precision-Recall Curve"

    def __init__(
        self,
        num_labels: int,
        thresholds: Optional[Union[int, List[float], Array]] = None,
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a `MultilabelPrecisionRecallCurve` instance."""
        super().__init__(**kwargs)
        _multilabel_precision_recall_curve_validate_args(
            num_labels,
            thresholds=thresholds,
            ignore_index=ignore_index,
        )
        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.thresholds = thresholds

        if thresholds is None:
            self.add_state_default_factory(
                "preds",
                default_factory=list,  # type: ignore
                dist_reduce_fn="cat",
            )
            self.add_state_default_factory(
                "target",
                default_factory=list,  # type: ignore
                dist_reduce_fn="cat",
            )
        else:
            len_thresholds = (
                len(thresholds)
                if isinstance(thresholds, list)
                else thresholds
                if isinstance(thresholds, int)
                else thresholds.shape[0]
            )

            def default(xp: ModuleType) -> Array:
                return xp.zeros(  # type: ignore[no-any-return]
                    (len_thresholds, num_labels, 2, 2),
                    dtype=xp.int32,
                    device=self.device,
                )

            self.add_state_default_factory(
                "confmat",
                default_factory=default,  # type: ignore
                dist_reduce_fn="sum",
            )

    def _update_state(self, target: Array, preds: Array) -> None:
        """Update the state of the metric."""
        xp = _multilabel_precision_recall_curve_validate_arrays(
            target,
            preds,
            self.num_labels,
            thresholds=self.thresholds,
            ignore_index=self.ignore_index,
        )

        (
            target,
            preds,
            self.thresholds,
        ) = _multilabel_precision_recall_curve_format_arrays(
            target,
            preds,
            self.num_labels,
            thresholds=self.thresholds,
            ignore_index=self.ignore_index,
            xp=xp,
        )
        state = _multilabel_precision_recall_curve_update(
            target,
            preds,
            self.num_labels,
            thresholds=self.thresholds,
            xp=xp,
        )

        if apc.is_array_api_obj(state):
            self.confmat += state  # type: ignore[attr-defined]
        else:
            self.target.append(state[0])  # type: ignore[attr-defined]
            self.preds.append(state[1])  # type: ignore[attr-defined]

    def _compute_metric(
        self,
    ) -> Union[
        Tuple[Array, Array, Array],
        Tuple[List[Array], List[Array], List[Array]],
    ]:
        """Compute the metric."""
        state = (
            (dim_zero_cat(self.target), dim_zero_cat(self.preds))  # type: ignore[attr-defined]
            if self.thresholds is None
            else self.confmat  # type: ignore[attr-defined]
        )
        return _multilabel_precision_recall_curve_compute(
            state,
            self.num_labels,
            thresholds=self.thresholds,  # type: ignore[arg-type]
            ignore_index=self.ignore_index,
        )
