"""Classes for computing area under the ROC curve."""

from typing import List, Literal, Union

import numpy as np

from cyclops.evaluate.metrics.functional.auroc import (
    _binary_auroc_compute,
    _multiclass_auroc_compute,
    _multilabel_auroc_compute,
)
from cyclops.evaluate.metrics.metric import Metric
from cyclops.evaluate.metrics.precision_recall_curve import (
    BinaryPrecisionRecallCurve,
    MulticlassPrecisionRecallCurve,
    MultilabelPrecisionRecallCurve,
)
from cyclops.evaluate.metrics.utils import _check_average_arg

# mypy: ignore-errors


class BinaryAUROC(BinaryPrecisionRecallCurve):
    """Compute the area under the ROC curve for binary classification tasks.

    Parameters
    ----------
        max_fpr : float, default=None
            The maximum value of the false positive rate. If not None, then
            the partial AUCROC in the range [0, max_fpr] is returned.
        thresholds : Union[int, List[float], numpy.ndarray], default=None
            Thresholds used for binarizing the values of ``preds``.
            If int, then the number of thresholds to use.
            If list or array, then the thresholds to use.
            If None, then the thresholds are automatically determined by the
            unique values in ``preds``.

    Examples
    --------
        >>> from cyclops.evaluation.metrics import BinaryAUROC
        >>> target = [0, 0, 1, 1]
        >>> preds = [0.1, 0.4, 0.35, 0.8]
        >>> metric = BinaryAUROC()
        >>> metric(target, preds)
        0.75
        >>> metric.reset_state()
        >>> target = [[0, 1, 0], [1, 0, 1]]
        >>> preds = [[0.1, 0.9, 0.8], [0.7, 0.2, 0.1]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        0.6111111111111112

    """

    def __init__(
        self,
        max_fpr: float = None,
        thresholds: Union[int, List[float], np.ndarray] = None,
    ) -> None:
        super().__init__(thresholds=thresholds)
        self.max_fpr = max_fpr

    def compute(self) -> float:  # type: ignore # super().compute() returns Tuple
        """Compute the area under the ROC curve from the state variables."""
        # pylint: disable=no-member # attributes are set with setattr
        if self.thresholds is None:
            state = [
                np.concatenate(self.target, axis=0),
                np.concatenate(self.preds, axis=0),
            ]
        else:
            state = self.confmat

        return _binary_auroc_compute(
            state, thresholds=self.thresholds, max_fpr=self.max_fpr
        )


class MulticlassAUROC(MulticlassPrecisionRecallCurve):
    """Compute the area under the ROC curve for multiclass classification tasks.

    Parameters
    ----------
        num_classes : int
            Number of classes.
        thresholds : Union[int, List[float], numpy.ndarray], default=None
            Thresholds used for binarizing the values of ``preds``.
            If int, then the number of thresholds to use.
            If list or array, then the thresholds to use.
            If None, then the thresholds are automatically determined by the
            unique values in ``preds``.
        average : Literal["macro", "weighted"], default=None
            If ``None``, then the scores for each class are returned. Otherwise,
            this determines the type of averaging performed on the scores. One of
            - `macro`: Calculate metrics for each class, and find their unweighted
              mean. This does not take class imbalance into account.
            - `weighted`: Calculate metrics for each class, and find their average,
              weighted by support (the number of true instances for each class).

    Examples
    --------
        >>> from cyclops.evaluation.metrics import MulticlassAUROC
        >>> target = [0, 1, 2, 0]
        >>> preds = [[0.9, 0.05, 0.05], [0.05, 0.89, 0.06],
        ...         [0.05, 0.01, 0.94], [0.9, 0.05, 0.05]]
        >>> metric = MulticlassAUROC(num_classes=3)
        >>> metric(target, preds)
        array([1., 1., 1.])
        >>> metric.reset_state()
        >>> target = [[0, 1, 0], [1, 0, 1]]
        >>> preds = [[[0.1, 0.9, 0.0], [0.7, 0.2, 0.1], [0.2, 0.3, 0.5]],
        ...         [[0.1, 0.1, 0.8], [0.7, 0.2, 0.1], [0.2, 0.3, 0.5]]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        array([0.5       , 0.22222222, 0.        ])

    """

    def __init__(
        self,
        num_classes: int,
        thresholds: Union[int, List[float], np.ndarray] = None,
        average: Literal["macro", "weighted"] = None,
    ) -> None:
        super().__init__(num_classes=num_classes, thresholds=thresholds)
        _check_average_arg(average)
        self.average = average

    def compute(self) -> Union[float, np.ndarray]:  # type: ignore
        """Compute the area under the ROC curve from the state variables."""
        # pylint: disable=no-member # attributes are set with setattr
        if self.thresholds is None:
            state = [
                np.concatenate(self.target, axis=0),
                np.concatenate(self.preds, axis=0),
            ]
        else:
            state = self.confmat

        return _multiclass_auroc_compute(
            state=state,
            num_classes=self.num_classes,
            thresholds=self.thresholds,
            average=self.average,
        )


class MultilabelAUROC(MultilabelPrecisionRecallCurve):
    """Compute the area under the ROC curve for multilabel classification tasks.

    Parameters
    ----------
        num_labels : int
            The number of labels in the multilabel classification task.
        thresholds : Union[int, List[float], numpy.ndarray], default=None
            Thresholds used for binarizing the values of ``preds``.
            If int, then the number of thresholds to use.
            If list or array, then the thresholds to use.
            If None, then the thresholds are automatically determined by the
            unique values in ``preds``.
        average : Literal["micro", "macro", "weighted"], default=None
            If ``None``, then the scores for each label are returned. Otherwise,
            this determines the type of averaging performed on the scores. One of
            - `micro`: Calculate metrics globally.
            - `macro`: Calculate metrics for each label, and find their unweighted
              mean. This does not take label imbalance into account.
            - `weighted``: Calculate metrics for each label, and find their average,
              weighted by support (the number of true instances for each label).

    Examples
    --------
        >>> from cyclops.evaluation.metrics import MultilabelAUROC
        >>> target = [[0, 1], [1, 1], [1, 0]]
        >>> preds = [[0.9, 0.05], [0.05, 0.89], [0.05, 0.01]]
        >>> metric = MultilabelAUROC(num_labels=2)
        >>> metric(target, preds)
        array([1., 1.])
        >>> metric.reset_state()
        >>> target = [[[0, 1], [1, 0]], [[1, 1], [1, 0]]]
        >>> preds = [[[0.9, 0.05], [0.05, 0.89]], [[0.05, 0.89], [0.05, 0.01]]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        array([1.   , 0.625])

    """

    def __init__(
        self,
        num_labels: int,
        thresholds: Union[int, List[float], np.ndarray] = None,
        average: Literal["micro", "macro", "weighted"] = None,
    ) -> None:
        super().__init__(num_labels=num_labels, thresholds=thresholds)
        _check_average_arg(average)
        self.average = average

    def compute(self) -> Union[float, np.ndarray]:  # type: ignore
        """Compute the area under the ROC curve from the state variables."""
        # pylint: disable=no-member # attributes are set with setattr
        if self.thresholds is None:
            state = [
                np.concatenate(self.target, axis=0),
                np.concatenate(self.preds, axis=0),
            ]
        else:
            state = self.confmat

        return _multilabel_auroc_compute(
            state=state,
            num_labels=self.num_labels,
            thresholds=self.thresholds,
            average=self.average,
        )


class AUROC(Metric):
    """Compute the AUROC curve for different types of classification tasks.

    Parameters
    ----------
        task : Literal["binary", "multiclass", "multilabel"]
            Task type. One of ``binary``, ``multiclass``, ``multilabel``.
        max_fpr : float, default=None
            The maximum value of the false positive rate. If not None, the
            a partial AUC in the range [0, max_fpr] is returned. Only used for
            binary classification.
        thresholds : int or list of floats or numpy.ndarray of floats, default=None
            Thresholds used for binarizing the values of ``preds``.
            If int, then the number of thresholds to use.
            If list or array, then the thresholds to use.
            If None, then the thresholds are automatically determined by the
            unique values in ``preds``.
        num_classes : int, default=None
            Number of classes. This parameter is required for the ``multiclass``
            task.
        num_labels : int, default=None
            Number of labels. This parameter is required for the ``multilabel``
            task.
        average : Literal["micro", "macro", "weighted"], default=None
            If not None, apply the method to compute the average area under the
            ROC curve. Only applicable for the ``multiclass`` and ``multilabel``
            tasks. One of:
            - ``micro``: Calculate metrics globally.
            - ``macro``: Calculate metrics for each label, and find their unweighted
              mean. This does not take label imbalance into account.
            - ``weighted``: Calculate metrics for each label, and find their average,
              weighted by support (accounting for label imbalance).

    Examples (binary)
    -----------------
        >>> from cyclops.evaluation.metrics import BinaryAUROC
        >>> target = [0, 0, 1, 1]
        >>> preds = [0.1, 0.4, 0.35, 0.8]
        >>> metric = BinaryAUROC()
        >>> metric(target, preds)
        0.75
        >>> metric.reset_state()
        >>> target = [[0, 1, 0], [1, 0, 1]]
        >>> preds = [[0.1, 0.9, 0.8], [0.7, 0.2, 0.1]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        0.6111111111111112

    Examples (multiclass)
    ---------------------
        >>> from cyclops.evaluation.metrics import MulticlassAUROC
        >>> target = [0, 1, 2, 0]
        >>> preds = [[0.9, 0.05, 0.05], [0.05, 0.89, 0.06],
        ...         [0.05, 0.01, 0.94], [0.9, 0.05, 0.05]]
        >>> metric = MulticlassAUROC(num_classes=3)
        >>> metric(target, preds)
        array([1., 1., 1.])
        >>> metric.reset_state()
        >>> target = [[0, 1, 0], [1, 0, 1]]
        >>> preds = [[[0.1, 0.9, 0.0], [0.7, 0.2, 0.1], [0.2, 0.3, 0.5]],
        ...         [[0.1, 0.1, 0.8], [0.7, 0.2, 0.1], [0.2, 0.3, 0.5]]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        array([0.5       , 0.22222222, 0.        ])


    Examples (multilabel)
    ---------------------
        >>> from cyclops.evaluation.metrics import MultilabelAUROC
        >>> target = [[0, 1], [1, 1], [1, 0]]
        >>> preds = [[0.9, 0.05], [0.05, 0.89], [0.05, 0.01]]
        >>> metric = MultilabelAUROC(num_labels=2)
        >>> metric(target, preds)
        array([1., 1.])
        >>> metric.reset_state()
        >>> target = [[[0, 1], [1, 0]], [[1, 1], [1, 0]]]
        >>> preds = [[[0.9, 0.05], [0.05, 0.89]], [[0.05, 0.89], [0.05, 0.01]]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        array([1.   , 0.625])

    """

    def __new__(  # type: ignore # mypy expects a subclass of AUROC
        cls,
        task: Literal["binary", "multiclass", "multilabel"],
        max_fpr: float = None,
        thresholds: Union[int, List[float], np.ndarray] = None,
        num_classes: int = None,
        num_labels: int = None,
        average: Literal["micro", "macro", "weighted"] = None,
    ) -> Metric:
        """Create a task-specific instance of the AUROC metric."""
        if task == "binary":
            return BinaryAUROC(max_fpr=max_fpr, thresholds=thresholds)
        if task == "multiclass":
            assert (
                isinstance(num_classes, int) and num_classes > 0
            ), "Number of classes must be a positive integer."
            return MulticlassAUROC(
                num_classes=num_classes,
                thresholds=thresholds,
                average=average,  # type: ignore
            )
        if task == "multilabel":
            assert (
                isinstance(num_labels, int) and num_labels > 0
            ), "Number of labels must be a positive integer."
            return MultilabelAUROC(
                num_labels=num_labels, thresholds=thresholds, average=average
            )
        raise ValueError(
            "Expected argument `task` to be either 'binary', 'multiclass' or "
            f"'multilabel', but got {task}"
        )
