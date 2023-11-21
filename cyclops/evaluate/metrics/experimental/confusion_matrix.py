"""Confusion matrix."""
from typing import Any, List, Optional, Tuple, Union

from cyclops.evaluate.metrics.experimental.functional.confusion_matrix import (
    _binary_confusion_matrix_compute,
    _binary_confusion_matrix_format_inputs,
    _binary_confusion_matrix_update_state,
    _binary_confusion_matrix_validate_args,
    _binary_confusion_matrix_validate_arrays,
    _multiclass_confusion_matrix_compute,
    _multiclass_confusion_matrix_format_inputs,
    _multiclass_confusion_matrix_update_state,
    _multiclass_confusion_matrix_validate_args,
    _multiclass_confusion_matrix_validate_arrays,
    _multilabel_confusion_matrix_compute,
    _multilabel_confusion_matrix_format_inputs,
    _multilabel_confusion_matrix_update_state,
    _multilabel_confusion_matrix_validate_args,
    _multilabel_confusion_matrix_validate_arrays,
)
from cyclops.evaluate.metrics.experimental.metric import Metric
from cyclops.evaluate.metrics.experimental.utils.ops import dim_zero_cat
from cyclops.evaluate.metrics.experimental.utils.typing import Array


class _AbstractConfusionMatrix(Metric):
    """Base class defining the common interface for confusion matrix classes."""

    name: str = "Confusion Matrix"

    tp: Union[Array, List[Array]]
    fp: Union[Array, List[Array]]
    tn: Union[Array, List[Array]]
    fn: Union[Array, List[Array]]

    def _create_state(self, size: int = 1) -> None:
        """Create the state variables.

        Parameters
        ----------
        size : int
            The size of the default Array to create for the state variables

        Raises
        ------
        RuntimeError
            If ``size`` is not greater than 0.

        """
        if size <= 0:
            raise RuntimeError(
                f"Expected `size` to be greater than 0, got {size}.",
            )
        dist_reduce_fn = "sum"

        def default(xp: Any) -> Array:
            return xp.zeros(shape=size, dtype=xp.int64)

        self.add_state_default_factory("tp", default, dist_reduce_fn=dist_reduce_fn)  # type: ignore
        self.add_state_default_factory("fp", default, dist_reduce_fn=dist_reduce_fn)  # type: ignore
        self.add_state_default_factory("tn", default, dist_reduce_fn=dist_reduce_fn)  # type: ignore
        self.add_state_default_factory("fn", default, dist_reduce_fn=dist_reduce_fn)  # type: ignore

    def _update_stat_scores(self, tp: Array, fp: Array, tn: Array, fn: Array) -> None:
        """Update the stat scores."""
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def _final_state(self) -> Tuple[Array, Array, Array, Array]:
        """Return the final state variables."""
        tp = dim_zero_cat(self.tp)
        fp = dim_zero_cat(self.fp)
        tn = dim_zero_cat(self.tn)
        fn = dim_zero_cat(self.fn)
        return tp, fp, tn, fn


class BinaryConfusionMatrix(
    _AbstractConfusionMatrix,
    registry_key="binary_confusion_matrix",
):
    """Confusion matrix for binary classification tasks.

    Parameters
    ----------
    threshold : float, default=0.5
        The threshold value to use when binarizing the inputs.
    normalize : {'true', 'pred', 'all', 'none' None}, optional, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        samples or all samples. If `None` or `'none'`, confusion matrix will
        not be normalized.
    ignore_index : int, optional, default=None
        Specifies a target value that is ignored and does not contribute to
        the confusion matrix. If `None`, all values are used.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> import numpy.array_api as np
    >>> from cyclops.evaluate.metrics.experimental import BinaryConfusionMatrix
    >>> target = np.asarray([0, 1, 0, 1, 0, 1])
    >>> preds = np.asarray([0, 0, 1, 1, 0, 1])
    >>> metric = BinaryConfusionMatrix()
    >>> metric(target, preds)
    Array([[2, 1],
           [1, 2]], dtype=int64)
    >>> target = np.asarray([0, 1, 0, 1, 0, 1])
    >>> preds = np.asarray([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])
    >>> metric = BinaryConfusionMatrix()
    >>> metric(target, preds)
    Array([[2, 1],
           [1, 2]], dtype=int32)
    >>> target = np.asarray([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
    >>> preds = np.asarray([[[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]],
    ...                     [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]]])

    """

    def __init__(
        self,
        threshold: float = 0.5,
        normalize: Optional[str] = None,
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the class."""
        super().__init__(**kwargs)

        _binary_confusion_matrix_validate_args(
            threshold=threshold,
            normalize=normalize,
            ignore_index=ignore_index,
        )

        self.threshold = threshold
        self.normalize = normalize
        self.ignore_index = ignore_index

        self._create_state(size=1)

    def _update_state(self, target: Array, preds: Array) -> None:
        """Update the state variables."""
        _binary_confusion_matrix_validate_arrays(
            target,
            preds,
            ignore_index=self.ignore_index,
        )
        target, preds = _binary_confusion_matrix_format_inputs(
            target,
            preds,
            threshold=self.threshold,
            ignore_index=self.ignore_index,
        )

        tn, fp, fn, tp = _binary_confusion_matrix_update_state(target, preds)
        self._update_stat_scores(tp, fp, tn, fn)

    def _compute_metric(self) -> Array:
        """Compute the confusion matrix."""
        tp, fp, tn, fn = self._final_state()
        return _binary_confusion_matrix_compute(
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            normalize=self.normalize,
        )


class MulticlassConfusionMatrix(Metric, registry_key="multiclass_confusion_matrix"):
    """Confusion matrix for multiclass classification tasks.

    Parameters
    ----------
    num_classes : int
        The number of classes.
    normalize : {'true', 'pred', 'all', 'none' None}, optional, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        samples or all samples. If `None` or `'none'`, confusion matrix will
        not be normalized.
    ignore_index : int, optional, default=None
        Specifies a target value that is ignored and does not contribute to
        the confusion matrix. If `None`, all values are used.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> import numpy.array_api as np
    >>> from cyclops.evaluate.metrics.experimental import MulticlassConfusionMatrix
    >>> target = np.asarray([2, 1, 0, 0])
    >>> preds = np.asarray([2, 1, 0, 1])
    >>> metric = MulticlassConfusionMatrix(num_classes=3)
    >>> metric(target, preds)
    Array([[1, 1, 0],
           [0, 1, 0],
           [0, 0, 1]], dtype=int64)
    >>> target = np.asarray([2, 1, 0, 0])
    >>> preds = np.asarray([[0.16, 0.26, 0.58],
    ...                     [0.22, 0.61, 0.17],
    ...                     [0.71, 0.09, 0.20],
    ...                     [0.05, 0.82, 0.13]])
    >>> metric = MulticlassConfusionMatrix(num_classes=3)
    >>> metric(target, preds)
    Array([[1, 1, 0],
           [0, 1, 0],
           [0, 0, 1]], dtype=int64)

    """

    name: str = "Confusion Matrix"
    confmat: Array

    def __init__(
        self,
        num_classes: int,
        normalize: Optional[str] = None,
        ignore_index: Optional[Union[int, Tuple[int]]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the class."""
        super().__init__(**kwargs)

        _multiclass_confusion_matrix_validate_args(
            num_classes,
            normalize=normalize,
            ignore_index=ignore_index,
        )

        self.num_classes = num_classes
        self.normalize = normalize
        self.ignore_index = ignore_index

        dist_reduce_fn = "sum"

        def default(xp: Any) -> Array:
            return xp.zeros((num_classes,) * 2, dtype=xp.int64)

        self.add_state_default_factory("confmat", default, dist_reduce_fn=dist_reduce_fn)  # type: ignore

    def _update_state(self, target: Array, preds: Array) -> None:
        """Update the state variable."""
        _multiclass_confusion_matrix_validate_arrays(
            target,
            preds,
            self.num_classes,
            ignore_index=self.ignore_index,
        )
        target, preds = _multiclass_confusion_matrix_format_inputs(
            target,
            preds,
            ignore_index=self.ignore_index,
        )
        confmat = _multiclass_confusion_matrix_update_state(
            target,
            preds,
            self.num_classes,
        )

        self.confmat += confmat

    def _compute_metric(self) -> Array:
        """Compute the confusion matrix."""
        confmat = self.confmat
        return _multiclass_confusion_matrix_compute(
            confmat,
            normalize=self.normalize,
        )


class MultilabelConfusionMatrix(
    _AbstractConfusionMatrix,
    registry_key="multilabel_confusion_matrix",
):
    """Confusion matrix for multilabel classification tasks.

    Parameters
    ----------
    num_labels : int
        The number of labels.
    threshold : float, default=0.5
        The threshold value to use when binarizing the inputs.
    normalize : {'true', 'pred', 'all', 'none' None}, optional, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        samples or all samples. If `None` or `'none'`, confusion matrix will
        not be normalized.
    ignore_index : int, optional, default=None
        Specifies a target value that is ignored and does not contribute to
        the confusion matrix. If `None`, all values are used.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> import numpy.array_api as np
    >>> from cyclops.evaluate.metrics.experimental import MultilabelConfusionMatrix
    >>> target = np.asarray([[0, 1, 0], [1, 0, 1]])
    >>> preds = np.asarray([[0, 0, 1], [1, 0, 1]])
    >>> metric = MultilabelConfusionMatrix(num_labels=3)
    >>> metric(target, preds)
    Array([[[1, 0],
            [0, 1]],

           [[1, 0],
            [1, 0]],

           [[0, 1],
            [0, 1]]], dtype=int64)
    >>> target = np.asarray([[0, 1, 0], [1, 0, 1]])
    >>> preds = np.asarray([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
    >>> metric = MultilabelConfusionMatrix(num_labels=3)
    >>> metric(target, preds)
    Array([[[1, 0],
            [0, 1]],

           [[1, 0],
            [1, 0]],

           [[0, 1],
            [0, 1]]], dtype=int64)

    """

    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        normalize: Optional[str] = None,
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the class."""
        super().__init__(**kwargs)

        _multilabel_confusion_matrix_validate_args(
            num_labels,
            threshold=threshold,
            normalize=normalize,
            ignore_index=ignore_index,
        )

        self.num_labels = num_labels
        self.threshold = threshold
        self.normalize = normalize
        self.ignore_index = ignore_index

        self._create_state(size=num_labels)

    def _update_state(self, target: Array, preds: Array) -> None:
        """Update the state variables."""
        _multilabel_confusion_matrix_validate_arrays(
            target,
            preds,
            self.num_labels,
            ignore_index=self.ignore_index,
        )
        target, preds = _multilabel_confusion_matrix_format_inputs(
            target,
            preds,
            threshold=self.threshold,
            ignore_index=self.ignore_index,
        )
        tn, fp, fn, tp = _multilabel_confusion_matrix_update_state(target, preds)
        self._update_stat_scores(tp, fp, tn, fn)

    def _compute_metric(self) -> Array:
        """Compute the confusion matrix."""
        tp, fp, tn, fn = self._final_state()
        return _multilabel_confusion_matrix_compute(
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            num_labels=self.num_labels,
            normalize=self.normalize,
        )
