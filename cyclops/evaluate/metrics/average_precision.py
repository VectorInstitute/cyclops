"""Classes for computing area under the Average Precision (AUPRC)."""

from typing import List, Literal, Optional, Type, Union

import numpy as np
import numpy.typing as npt

from cyclops.evaluate.metrics.functional.average_precision import (
    _binary_average_precision_compute,
)
from cyclops.evaluate.metrics.metric import Metric
from cyclops.evaluate.metrics.precision_recall_curve import (
    BinaryPrecisionRecallCurve,
)


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
    >>> from cyclops.evaluate.metrics import BinaryAveragePrecision
    >>> target = [0, 1, 0, 1]
    >>> preds = [0.1, 0.4, 0.35, 0.8]
    >>> metric = BinaryAveragePrecision(thresholds=3)
    >>> metric(target, preds)
    0.75
    >>> metric.reset_state()
    >>> target = [[0, 1, 0, 1], [1, 1, 0, 0]]
    >>> preds = [[0.1, 0.4, 0.35, 0.8], [0.6, 0.3, 0.1, 0.7]]
    >>> for t, p in zip(target, preds):
    ...     metric.update_state(t, p)
    >>> metric.compute()
    0.5833333333333333

    """

    name: str = "Average Precision"

    def compute(  # type: ignore[override]
        self,
    ) -> float:
        """Compute the average precision score from the state."""
        if self.thresholds is None:
            state = (
                np.concatenate(self.target, axis=0),  # type: ignore[attr-defined]
                np.concatenate(self.preds, axis=0),  # type: ignore[attr-defined]
            )
        else:
            state = self.confmat  # type: ignore[attr-defined]

        return _binary_average_precision_compute(
            state,
            self.thresholds,
            self.pos_label,
        )


class AveragePrecision(
    Metric,
    registry_key="average_precision",
    force_register=True,
):
    """Compute the precision-recall curve for different classification tasks.

    Parameters
    ----------
    task : Literal["binary", "multiclass", "multilabel"]
        The task for which the precision-recall curve is computed.
    thresholds : int or list of floats or numpy.ndarray of floats, default=None
        Thresholds used for computing the precision and recall scores. If int,
        then the number of thresholds to use. If list or array, then the
        thresholds to use. If None, then the thresholds are automatically
        determined by the sunique values in ``preds``
    pos_label : int, default=1
        Label to consider as positive for binary classification tasks.
    num_classes : int, optional
        The number of classes in the dataset. Required if ``task`` is
        ``"multiclass"``.
    num_labels : int, optional
        The number of labels in the dataset. Required if ``task`` is
        ``"multilabel"``.

    Examples
    --------
    >>> # (binary)
    >>> from cyclops.evaluate.metrics import PrecisionRecallCurve
    >>> target = [1, 1, 1, 0]
    >>> preds = [0.6, 0.2, 0.3, 0.8]
    >>> metric = AveragePrecision(task="binary", thresholds=None)
    >>> metric(target, preds)
    0.6388888888888888
    >>> metric.reset_state()
    >>> target = [[1, 0, 1, 1], [0, 0, 0, 1]]
    >>> preds = [[0.5, 0.4, 0.1, 0.3], [0.9, 0.6, 0.45, 0.8]]
    >>> for t, p in zip(target, preds):
    ...     metric.update_state(t, p)
    >>> metric.compute()
    0.48214285714285715

    """

    name: str = "Average Precision"

    def __new__(  # type: ignore # mypy expects a subclass of AveragePrecision
        cls: Type[Metric],
        task: Literal["binary", "multiclass", "multilabel"],
        thresholds: Optional[Union[int, List[float], npt.NDArray[np.float_]]] = None,
        pos_label: int = 1,
        num_classes: Optional[int] = None,
        num_labels: Optional[int] = None,
    ) -> Metric:
        """Create a task-specific instance of the average precision metric."""
        if task == "binary":
            return BinaryAveragePrecision(
                thresholds=thresholds,
                pos_label=pos_label,
            )
        if task == "multiclass":
            raise NotImplementedError(
                "Multiclass average precision is not implemented."
            )
        if task == "multilabel":
            raise NotImplementedError(
                "Multilabel average precision is not implemented."
            )

        raise ValueError(
            "Expected argument `task` to be either 'binary', 'multiclass' or "
            f"'multilabel', but got {task}",
        )
