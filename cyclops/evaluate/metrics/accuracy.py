"""Classes for computing accuracy metrics."""

from typing import Literal, Optional

from cyclops.evaluate.metrics.functional.accuracy import _accuracy_reduce
from cyclops.evaluate.metrics.metric import Metric
from cyclops.evaluate.metrics.stat_scores import (
    BinaryStatScores,
    MulticlassStatScores,
    MultilabelStatScores,
)
from cyclops.evaluate.metrics.utils import _check_average_arg


class BinaryAccuracy(BinaryStatScores):
    """Compute accuracy score for binary classification tasks.

    Parameters
    ----------
        pos_label : int, default=1
            The label of the positive class. Can be 0 or 1.
        threshold : float, default=0.5
            The threshold value for converting probability or logit scores to
            binary. A sigmoid function is first applied to logits to convert them
            to probabilities.
        zero_division : Literal["warn", 0, 1], default="warn"
            Sets the value to return when there is a zero division. If set to ``warn``,
            this acts as 0, but warnings are also raised.

    Examples
    --------
        >>> from cyclops.evaluation.metrics import BinaryAccuracy
        >>> target = [0, 1, 0, 1]
        >>> preds = [0, 1, 1, 1]
        >>> metric = BinaryAccuracy()
        >>> metric(target, preds)
        0.75
        >>> metric.reset_state()
        >>> target = [[0, 1, 0, 1], [1, 0, 1, 0]]
        >>> preds = [[0, 1, 1, 1], [1, 0, 1, 0]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        0.875

    """

    def __init__(
        self,
        threshold: float = 0.5,
        pos_label: int = 1,
        zero_division: Literal["warn", 0, 1] = "warn",
    ) -> None:
        super().__init__(threshold=threshold, pos_label=pos_label)
        self.zero_division = zero_division

    def compute(self) -> float:
        """Compute the accuracy score from the state."""
        tp, fp, tn, fn = self._final_state()
        return _accuracy_reduce(
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            task_type="binary",
            average=None,
            zero_division=self.zero_division,
        )


class MulticlassAccuracy(MulticlassStatScores):
    """Compute the accuracy score for multiclass classification problems.

    Parameters
    ----------
        num_classes : int
            Number of classes in the dataset.
        top_k : int, default=None
            Number of highest probability predictions or logits to consider when
            computing the accuracy score.
        average : Literal["micro", "macro", "weighted", None], default=None
            If not None, this determines the type of averaging performed on the data:
                - ``micro``: Calculate metrics globally.
                - ``macro``: Calculate metrics for each class, and find their
                  unweighted mean. This does not take class imbalance into account.
                - ``weighted``: Calculate metrics for each class, and find their
                  average, weighted by support (the number of true instances for
                  each class). This alters ``macro`` to account for class imbalance.
        zero_division : Literal["warn", 0, 1], default="warn"
            Sets the value to return when there is a zero division. If set to ``warn``,
            this acts as 0, but warnings are also raised.

    Examples
    --------
        >>> from cyclops.evaluation.metrics import MulticlassAccuracy
        >>> target = [0, 1, 2, 2, 2]
        >>> preds = [0, 0, 2, 2, 1]
        >>> metric = MulticlassAccuracy(num_classes=3)
        >>> metric(target, preds)
        array([1.        , 0.        , 0.66666667])
        >>> metric.reset_state()
        >>> target = [[0, 1, 2], [2, 1, 0]]
        >>> preds = [[[0.05, 0.95, 0], [0.1, 0.8, 0.1], [0.2, 0.6, 0.2]],
        ...          [[0.1, 0.8, 0.1], [0.05, 0.95, 0], [0.2, 0.6, 0.2]]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        array([0., 1., 0.])

    """

    def __init__(
        self,
        num_classes: int,
        top_k: Optional[int] = None,
        average: Literal["micro", "macro", "weighted", None] = None,
        zero_division: Literal["warn", 0, 1] = "warn",
    ) -> None:
        super().__init__(num_classes=num_classes, top_k=top_k, classwise=True)
        _check_average_arg(average)

        self.average = average
        self.zero_division = zero_division

    def compute(self) -> float:
        """Compute the accuracy score from the state."""
        tp, fp, tn, fn = self._final_state()
        return _accuracy_reduce(
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            task_type="multiclass",
            average=self.average,
            zero_division=self.zero_division,
        )


class MultilabelAccuracy(MultilabelStatScores):
    """Compute the accuracy score for multilabel-indicator targets.

    Parameters
    ----------
        num_labels : int
            Number of labels in the multilabel classification task.
        threshold : float, default=0.5
            Threshold value for binarizing the output of the classifier.
        top_k : int, optional, default=None
            The number of highest probability or logit predictions considered
            to find the correct label. Only works when ``preds`` contains
            probabilities/logits.
        average : Literal['micro', 'macro', 'weighted', None], default=None
            If None, return the accuracy score per label, otherwise this determines
            the type of averaging performed on the data:
                - ``micro``: Calculate metrics globally.
                - ``macro``: Calculate metrics for each label, and find their unweighted
                  mean. This does not take label imbalance into account.
                - ``weighted``: Calculate metrics for each label, and find their
                  average, weighted by support (the number of true instances for
                  each label).
        zero_division : Literal['warn', 0, 1], default="warn"
            Sets the value to return when there is a zero division. If set to ``warn``,
            this acts as 0, but warnings are also raised.

    Examples
    --------
        >>> from cyclops.evaluation.metrics import MultilabelAccuracy
        >>> target = [[0, 1, 1], [1, 0, 0]]
        >>> preds = [[0, 1, 0], [1, 0, 1]]
        >>> metric = MultilabelAccuracy(num_labels=3)
        >>> metric(target, preds)
        array([1., 1., 0.])
        >>> metric.reset_state()
        >>> target = [[[0, 1, 1], [1, 0, 0]], [[1, 0, 0], [0, 1, 1]]]
        >>> preds = [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]],
        ...          [[0.1, 0.8, 0.1], [0.05, 0.95, 0]]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        array([0.5, 0.5, 0.5])

    """

    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        top_k: Optional[int] = None,
        average: Literal["micro", "macro", "weighted", None] = None,
        zero_division: Literal["warn", 0, 1] = "warn",
    ) -> None:
        super().__init__(
            num_labels=num_labels, threshold=threshold, top_k=top_k, labelwise=True
        )
        _check_average_arg(average)

        self.average = average
        self.zero_division = zero_division

    def compute(self) -> float:
        """Compute the accuracy score from the state."""
        tp, fp, tn, fn = self._final_state()
        return _accuracy_reduce(
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            task_type="multilabel",
            average=self.average,
            zero_division=self.zero_division,
        )


class Accuracy(Metric):
    """Compute accuracy score for different classification tasks.

    Parameters
    ----------
        task : Literal["binary", "multiclass", "multilabel"]
            The type of task for the input data. One of 'binary', 'multiclass'
            or 'multilabel'.
        pos_label : int, default=1
            Label to consider as positive for binary classification tasks.
        num_classes : int, default=None
            Number of classes for the task. Required if ``task`` is ``"multiclass"``.
        threshold : float, default=0.5
            Threshold for deciding the positive class. Only used if ``task`` is
            ``"binary"`` or ``"multilabel"``.
        top_k : int, optional
            If given, and predictions are probabilities/logits, the precision will
            be computed only for the top k classes. Otherwise, ``top_k`` will be
            set to 1. Only used if ``task`` is ``"multiclass"`` or ``"multilabel"``.
        num_labels : int, default=None
            Number of labels for the task. Required if ``task`` is ``"multilabel"``.
        average : Literal["micro", "macro", "weighted", None], default=None
            If ``None``, return the recall score for each label/class. Otherwise,
            use one of the following options to compute the average score:
                - ``micro``: Calculate metrics globally.
                - ``macro``: Calculate metrics for each class/label, and find their
                  unweighted mean. This does not take label imbalance into account.
                - ``weighted``: Calculate metrics for each label/class, and find
                  their average weighted by support (the number of true instances
                  for each label/class). This alters ``macro`` to account for
                  label/class imbalance.
        zero_division : Literal["warn", 0, 1], default="warn"
            Sets the value to return when there is a zero division. If set to ``warn``,
            this acts as 0, but warnings are also raised.

    Examples (binary)
    -----------------
        >>> from cyclops.evaluation.metrics import Accuracy
        >>> target = [0, 0, 1, 1]
        >>> preds = [0, 1, 1, 1]
        >>> metric = Accuracy(task="binary")
        >>> metric(target, preds)
        0.75
        >>> metric.reset_state()
        >>> target = [[0, 0, 1, 1], [1, 1, 0, 0]]
        >>> preds = [[0.05, 0.95, 0, 0], [0.1, 0.8, 0.1, 0]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        0.5

    Examples (multiclass)
    ---------------------
        >>> from cyclops.evaluation.metrics import Accuracy
        >>> target = [0, 1, 2, 2, 2]
        >>> preds = [0, 0, 2, 2, 1]
        >>> metric = Accuracy(task="multiclass", num_classes=3)
        >>> metric(target, preds)
        array([1.        , 0.        , 0.66666667])
        >>> metric.reset_state()
        >>> target = [[0, 1, 2], [2, 1, 0]]
        >>> preds = [[[0.05, 0.95, 0], [0.1, 0.8, 0.1], [0.2, 0.6, 0.2]],
        ...          [[0.1, 0.8, 0.1], [0.05, 0.95, 0], [0.2, 0.6, 0.2]]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        array([0., 1., 0.])

    Examples (multilabel)
    ---------------------
        >>> from cyclops.evaluation.metrics import Accuracy
        >>> target = [[0, 1, 1], [1, 0, 0]]
        >>> preds = [[0, 1, 0], [1, 0, 1]]
        >>> metric = Accuracy(task="multilabel", num_labels=3)
        >>> metric(target, preds)
        array([1., 1., 0.])
        >>> metric.reset_state()
        >>> target = [[[0, 1, 1], [1, 0, 0]], [[1, 0, 0], [0, 1, 1]]]
        >>> preds = [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]],
        ...          [[0.1, 0.8, 0.1], [0.05, 0.95, 0]]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        array([0.5, 0.5, 0.5])

    """

    def __new__(  # type: ignore # mypy expects a subclass of Accuracy
        cls,
        task: Literal["binary", "multiclass", "multilabel"],
        pos_label: int = 1,
        num_classes: int = None,
        threshold: float = 0.5,
        top_k: Optional[int] = None,
        num_labels: int = None,
        average: Literal["micro", "macro", "weighted", None] = None,
        zero_division: Literal["warn", 0, 1] = "warn",
    ) -> Metric:
        """Create a task-specific instance of the metric."""
        if task == "binary":
            return BinaryAccuracy(
                threshold=threshold, pos_label=pos_label, zero_division=zero_division
            )
        if task == "multiclass":
            assert (
                isinstance(num_classes, int) and num_classes > 0
            ), "Number of classes must be specified for multiclass classification."
            return MulticlassAccuracy(
                num_classes=num_classes,
                top_k=top_k,
                average=average,
                zero_division=zero_division,
            )
        if task == "multilabel":
            assert (
                isinstance(num_labels, int) and num_labels > 0
            ), "Number of labels must be specified for multilabel classification."
            return MultilabelAccuracy(
                num_labels=num_labels,
                threshold=threshold,
                average=average,
                zero_division=zero_division,
            )
        raise ValueError(
            f"Task {task} is not supported, expected one of 'binary', 'multiclass'"
            " or 'multilabel'"
        )
