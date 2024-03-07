"""Matthews Correlation Coefficient (MCC) metric."""

from typing import Any, Optional, Tuple, Union

from cyclops.evaluate.metrics.experimental.confusion_matrix import (
    BinaryConfusionMatrix,
    MulticlassConfusionMatrix,
    MultilabelConfusionMatrix,
)
from cyclops.evaluate.metrics.experimental.functional.confusion_matrix import (
    _binary_confusion_matrix_compute,
    _multilabel_confusion_matrix_compute,
)
from cyclops.evaluate.metrics.experimental.functional.matthews_corr_coef import (
    _mcc_reduce,
)
from cyclops.evaluate.metrics.experimental.utils.types import Array


class BinaryMCC(BinaryConfusionMatrix, registry_key="binary_mcc"):
    """A measure of the agreement between predicted and actual values.

    Parameters
    ----------
    threshold : float, default=0.5
        The threshold value to use when binarizing the inputs.
    ignore_index : int, optional, default=None
        Specifies a target value that is ignored and does not contribute to the
        metric. If `None`, all values are used.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> import numpy.array_api as anp
    >>> from cyclops.evaluate.metrics.experimental import BinaryMCC
    >>> target = anp.asarray([0, 1, 0, 1, 0, 1])
    >>> preds = anp.asarray([0, 0, 1, 1, 0, 1])
    >>> metric = BinaryMCC()
    >>> metric(target, preds)
    Array(0.33333333, dtype=float64)
    >>> target = anp.asarray([0, 1, 0, 1, 0, 1])
    >>> preds = anp.asarray([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])
    >>> metric = BinaryMCC()
    >>> metric(target, preds)
    Array(0.33333333, dtype=float64)

    """

    name: str = "Matthews Correlation Coefficient"

    def __init__(
        self,
        threshold: float = 0.5,
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the class."""
        super().__init__(threshold, normalize=None, ignore_index=ignore_index, **kwargs)

    def _compute_metric(self) -> Array:
        """Compute the confusion matrix."""
        tn, fp, fn, tp = self._final_state()
        confmat = _binary_confusion_matrix_compute(
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            normalize=self.normalize,
        )
        return _mcc_reduce(confmat)


class MulticlassMCC(MulticlassConfusionMatrix, registry_key="multiclass_mcc"):
    """A measure of the agreement between predicted and actual values.

    Parameters
    ----------
    num_classes : int
        The number of classes.
    ignore_index : int, optional, default=None
        Specifies a target value that is ignored and does not contribute to the
        metric. If `None`, all values are used.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> import numpy.array_api as anp
    >>> from cyclops.evaluate.metrics.experimental import MulticlassMCC
    >>> target = anp.asarray([2, 1, 0, 0])
    >>> preds = anp.asarray([2, 1, 0, 1])
    >>> metric = MulticlassMCC(num_classes=3)
    >>> metric(target, preds)
    Array(0.7, dtype=float64)
    >>> target = anp.asarray([2, 1, 0, 0])
    >>> preds = anp.asarray(
    ...     [
    ...         [0.16, 0.26, 0.58],
    ...         [0.22, 0.61, 0.17],
    ...         [0.71, 0.09, 0.20],
    ...         [0.05, 0.82, 0.13],
    ...     ]
    ... )
    >>> metric = MulticlassMCC(num_classes=3)
    >>> metric(target, preds)
    Array(0.7, dtype=float64)
    """

    name: str = "Matthews Correlation Coefficient"

    def __init__(
        self,
        num_classes: int,
        ignore_index: Optional[Union[int, Tuple[int]]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the class."""
        super().__init__(
            num_classes=num_classes,
            normalize=None,
            ignore_index=ignore_index,
            **kwargs,
        )

    def _compute_metric(self) -> Array:
        """Compute the confusion matrix."""
        return _mcc_reduce(self.confmat)  # type: ignore


class MultilabelMCC(MultilabelConfusionMatrix, registry_key="multilabel_mcc"):
    """A measure of the agreement between predicted and actual values.

    Parameters
    ----------
    num_labels : int
        The number of labels.
    threshold : float, default=0.5
        The threshold value to use when binarizing the inputs.
    ignore_index : int, optional, default=None
        Specifies a target value that is ignored and does not contribute to the
        metric. If `None`, all values are used.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> import numpy.array_api as anp
    >>> from cyclops.evaluate.metrics.experimental import MultilabelMCC
    >>> target = anp.asarray([[0, 1, 0], [1, 0, 1]])
    >>> preds = anp.asarray([[0, 0, 1], [1, 0, 1]])
    >>> metric = MultilabelMCC(num_labels=3)
    >>> metric(target, preds)
    Array(0.33333333, dtype=float64)
    >>> target = anp.asarray([[0, 1, 0], [1, 0, 1]])
    >>> preds = anp.asarray([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
    >>> metric = MultilabelMCC(num_labels=3)
    >>> metric(target, preds)
    Array(0.33333333, dtype=float64)

    """

    name: str = "Matthews Correlation Coefficient"

    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the class."""
        super().__init__(
            num_labels=num_labels,
            threshold=threshold,
            normalize=None,
            ignore_index=ignore_index,
            **kwargs,
        )

    def _compute_metric(self) -> Array:
        """Compute the confusion matrix."""
        tn, fp, fn, tp = self._final_state()
        confmat = _multilabel_confusion_matrix_compute(
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            num_labels=self.num_labels,
            normalize=self.normalize,
        )
        return _mcc_reduce(confmat)
