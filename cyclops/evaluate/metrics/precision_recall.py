"""Classes for computing precision and recall metrics."""

from typing import Literal, Optional

from cyclops.evaluate.metrics.functional.precision_recall import (
    _precision_recall_reduce,
)
from cyclops.evaluate.metrics.metric import Metric
from cyclops.evaluate.metrics.stat_scores import (
    BinaryStatScores,
    MulticlassStatScores,
    MultilabelStatScores,
)
from cyclops.evaluate.metrics.utils import _check_average_arg


class BinaryPrecision(BinaryStatScores):
    """Compute the precision score for binary classification tasks.

    Parameters
    ----------
        pos_label : int, default=1
            The label of the positive class.
        threshold : float, default=0.5
            Threshold for deciding the positive class.
        zero_division : Literal["warn", 0, 1], default="warn"
            Value to return when there is a zero division. If set to "warn", this
            acts as 0, but warnings are also raised.

    Examples
    --------
        >>> from cyclops.evaluation.metrics import BinaryPrecision
        >>> target = [0, 1, 0, 1]
        >>> preds = [0, 1, 1, 1]
        >>> metric = BinaryPrecision()
        >>> metric(target, preds)
        0.6666666666666666
        >>> metric.reset_state()
        >>> target = [[0, 1, 0, 1], [0, 0, 1, 1]]
        >>> preds = [[0.1, 0.9, 0.8, 0.2], [0.2, 0.3, 0.6, 0.1]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        0.6666666666666666

    """

    def __init__(
        self,
        pos_label: int = 1,
        threshold: float = 0.5,
        zero_division: Literal["warn", 0, 1] = "warn",
    ) -> None:
        super().__init__(threshold=threshold, pos_label=pos_label)
        self.zero_division = zero_division

    def compute(self) -> float:
        """Compute the precision score from the state."""
        tp, fp, _, fn = self._final_state()
        return _precision_recall_reduce(
            tp=tp,
            fp=fp,
            fn=fn,
            metric="precision",
            average=None,
            zero_division=self.zero_division,
        )


class MulticlassPrecision(MulticlassStatScores):
    """Compute the precision score for multiclass classification tasks.

    Parameters
    ----------
        num_classes : int
            Number of classes in the dataset.
        top_k : int, optional
            If given, and predictions are probabilities/logits, the precision will
            be computed only for the top k classes. Otherwise, ``top_k`` will be
            set to 1.
        average : Literal["micro", "macro", "weighted", None], default=None
           If ``None``, return the score for each class. Otherwise, use one of the
           following options to compute the average score:
                - ``micro``: Calculate metric globally from the total count of true
                  positives and false positives.
                - ``macro``: Calculate metric for each class, and find their
                  unweighted mean. This does not take class imbalance into account.
                - ``weighted``: Calculate metric for each class, and find their
                  average weighted by the support (the number of true instances
                  for each class). This alters "macro" to account for class
                  imbalance.
        zero_division : Literal["warn", 0, 1], default="warn"
            Value to return when there is a zero division. If set to "warn", this
            acts as 0, but warnings are also raised.

    Examples
    --------
        >>> from cyclops.evaluation.metrics import MulticlassPrecision
        >>> target = [0, 1, 2, 0]
        >>> preds = [0, 2, 1, 0]
        >>> metric = MulticlassPrecision(num_classes=3, average=None)
        >>> metric(target, preds)
        array([1. , 0. , 0.])
        >>> metric.reset_state()
        >>> target = [[0, 1, 2, 0], [2, 1, 2, 0]]
        >>> preds = [
        ...     [[0.1, 0.6, 0.3],
        ...      [0.05, 0.1, 0.85],
        ...      [0.2, 0.7, 0.1],
        ...      [0.9, 0.05, 0.05]],
        ...     [[0.1, 0.6, 0.3],
        ...      [0.05, 0.1, 0.85],
        ...      [0.2, 0.7, 0.1],
        ...      [0.9, 0.05, 0.05]]
        ... ]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        array([1., 0., 0.])

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
        """Compute the precision score from the state."""
        tp, fp, _, fn = self._final_state()
        return _precision_recall_reduce(
            tp=tp,
            fp=fp,
            fn=fn,
            metric="precision",
            average=self.average,
            zero_division=self.zero_division,
        )


class MultilabelPrecision(MultilabelStatScores):
    """Compute the precision score for multilabel classification tasks.

    Parameters
    ----------
        num_labels : int
            Number of labels for the task.
        threshold : float, default=0.5
            Threshold for deciding the positive class.
        top_k : int, optional
            If given, and predictions are probabilities/logits, the precision will
            be computed only for the top k classes. Otherwise, ``top_k`` will be
            set to 1.
        average : Literal["micro", "macro", "weighted", None], default=None
            If ``None``, return the precision score for each label. Otherwise,
            use one of the following options to compute the average precision score:
                - ``micro``: Calculate metric globally from the total count of true
                  positives and false positives.
                - ``macro``: Calculate metric for each label, and find their
                  unweighted mean. This does not take label imbalance into account.
                - ``weighted``: Calculate metric for each label, and find their
                  average weighted by the support (the number of true instances
                  for each label). This alters "macro" to account for label
                  imbalance.
        zero_division : Literal["warn", 0, 1], default="warn"
            Value to return when there is a zero division. If set to "warn", this
            acts as 0, but warnings are also raised.

    Examples
    --------
        >>> from cyclops.evaluation.metrics import MultilabelPrecision
        >>> target = [[0, 1], [1, 1]]
        >>> preds = [[0.1, 0.9], [0.2, 0.8]]
        >>> metric = MultilabelPrecision(num_labels=2, average=None)
        >>> metric(target, preds)
        array([0., 1.])
        >>> metric.reset_state()
        >>> target = [[[0, 1], [1, 1]], [[1, 1], [1, 0]]]
        >>> preds = [
        ...     [[0.1, 0.7], [0.2, 0.8]],
        ...     [[0.5, 0.9], [0.3, 0.4]]
        ... ]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        array([1., 1.])

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
            num_labels=num_labels,
            threshold=threshold,
            top_k=top_k,
            labelwise=True,
        )
        _check_average_arg(average)

        self.average = average
        self.zero_division = zero_division

    def compute(self) -> float:
        """Compute the precision score from the state."""
        tp, fp, _, fn = self._final_state()
        return _precision_recall_reduce(
            tp=tp,
            fp=fp,
            fn=fn,
            metric="precision",
            average=self.average,
            zero_division=self.zero_division,
        )


class Precision(Metric):
    """Compute the precision score for different types of classification tasks.

    This metric can be used for binary, multiclass, and multilabel classification
    tasks. It creates the appropriate metric based on the ``task`` parameter.

    Parameters
    ----------
        task : Literal["binary", "multiclass", "multilabel"]
            Type of classification task.
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
            If ``None``, return the precision score for each label/class. Otherwise,
            use one of the following options to compute the average precision score:
                - ``micro``: Calculate metrics globally by counting the total true
                  positives and false positives.
                - ``macro``: Calculate metrics for each class/label, and find their
                  unweighted mean. This does not take label/class imbalance into
                  account.
                - ``weighted``: Calculate metrics for each label/class, and find
                  their average weighted by support (the number of true instances
                  for each label/class). This alters ``macro`` to account for
                  label/class imbalance.
        zero_division : Literal["warn", 0, 1], default="warn"
            Value to return when there is a zero division. If set to "warn", this
            acts as 0, but warnings are also raised.

    Examples (binary)
    -----------------
        >>> from cyclops.evaluation.metrics import Precision
        >>> target = [0, 1, 0, 1]
        >>> preds = [0, 1, 1, 1]
        >>> metric = Precision(task="binary")
        >>> metric(target, preds)
        0.6666666666666666
        >>> metric.reset_state()
        >>> target = [[0, 1, 0, 1], [0, 0, 1, 1]]
        >>> preds = [[0.1, 0.9, 0.8, 0.2], [0.2, 0.3, 0.6, 0.1]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        0.6666666666666666

    Examples (multiclass)
    ---------------------
        >>> from cyclops.evaluation.metrics import Precision
        >>> target = [0, 1, 2, 0]
        >>> preds = [0, 2, 1, 0]
        >>> metric = Precision(task="multiclass", num_classes=3)
        >>> metric(target, preds)
        array([1. , 0. , 0.])
        >>> metric.reset_state()
        >>> target = [[0, 1, 2, 0], [2, 1, 2, 0]]
        >>> preds = [
        ...     [[0.1, 0.6, 0.3],
        ...      [0.05, 0.1, 0.85],
        ...      [0.2, 0.7, 0.1],
        ...      [0.9, 0.05, 0.05]],
        ...     [[0.1, 0.6, 0.3],
        ...      [0.05, 0.1, 0.85],
        ...      [0.2, 0.7, 0.1],
        ...      [0.9, 0.05, 0.05]]
        ... ]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        array([1., 0., 0.])

    Examples (multilabel)
    ---------------------
        >>> from cyclops.evaluation.metrics import Precision
        >>> target = [[0, 1], [1, 1]]
        >>> preds = [[0.1, 0.9], [0.2, 0.8]]
        >>> metric = Precision(task="multilabel", num_labels=2)
        >>> metric.update_state(target, preds)
        >>> metric.compute()
        array([0., 1.])
        >>> metric.reset_state()
        >>> target = [[[0, 1], [1, 1]], [[1, 1], [1, 0]]]
        >>> preds = [
        ...     [[0.1, 0.7], [0.2, 0.8]],
        ...     [[0.5, 0.9], [0.3, 0.4]]
        ... ]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        array([1., 1.])

    """

    def __new__(  # type: ignore # mypy expects a subclass of Precision
        cls,
        task: Literal["binary", "multiclass", "multilabel"],
        pos_label: int = 1,
        num_classes: Optional[int] = None,
        threshold: float = 0.5,
        top_k: Optional[int] = None,
        num_labels: Optional[int] = None,
        average: Literal["micro", "macro", "weighted", None] = None,
        zero_division: Literal["warn", 0, 1] = "warn",
    ) -> Metric:
        """Create a task-specific precision metric."""
        if task == "binary":
            return BinaryPrecision(
                threshold=threshold, pos_label=pos_label, zero_division=zero_division
            )
        if task == "multiclass":
            assert (
                isinstance(num_classes, int) and num_classes > 0
            ), "Number of classes must be specified for multiclass classification."
            return MulticlassPrecision(
                num_classes=num_classes,
                top_k=top_k,
                average=average,
                zero_division=zero_division,
            )
        if task == "multilabel":
            assert (
                isinstance(num_labels, int) and num_labels > 0
            ), "Number of labels must be specified for multilabel classification."
            return MultilabelPrecision(
                num_labels=num_labels,
                threshold=threshold,
                top_k=top_k,
                average=average,
                zero_division=zero_division,
            )
        raise ValueError(
            f"Task '{task}' not supported, expected 'binary', 'multiclass' or "
            f"'multilabel'."
        )


class BinaryRecall(BinaryStatScores):
    """Computes recall score for binary classification.

    Parameters
    ----------
        pos_label : int, default=1
            Label of the positive class.
        threshold : float, default=0.5
            Threshold for deciding the positive class.
        zero_division : Literal["warn", 0, 1], default="warn"
            Value to return when there is a zero division. If set to "warn", this
            acts as 0, but warnings are also raised.

    Examples
    --------
        >>> from cyclops.evaluation.metrics import BinaryRecall
        >>> target = [0, 1, 0, 1]
        >>> preds = [0, 1, 1, 0]
        >>> metric = Recall()
        >>> metric(target, preds)
        0.5
        >>> metric.reset_state()
        >>> target = [[0, 1, 0, 1], [0, 0, 1, 1]]
        >>> preds = [[0.1, 0.9, 0.8, 0.2], [0.2, 0.3, 0.6, 0.1]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        0.5

    """

    def __init__(
        self,
        pos_label: int = 1,
        threshold: float = 0.5,
        zero_division: Literal["warn", 0, 1] = "warn",
    ) -> None:
        super().__init__(threshold=threshold, pos_label=pos_label)
        self.zero_division = zero_division

    def compute(self) -> float:
        """Compute the recall score from the state."""
        tp, fp, _, fn = self._final_state()
        return _precision_recall_reduce(
            tp=tp,
            fp=fp,
            fn=fn,
            metric="recall",
            average=None,
            zero_division=self.zero_division,
        )


class MulticlassRecall(MulticlassStatScores):
    """Compute the recall score for multiclass classification tasks.

    Parameters
    ----------
        num_classes : int
            Number of classes in the dataset.
        top_k : int, optional
            If given, and predictions are probabilities/logits, the recall will
            be computed only for the top k classes. Otherwise, ``top_k`` will be
            set to 1.
        average : Literal["micro", "macro", "weighted", None], default=None
           If ``None``, return the recall score for each class. Otherwise,
           use one of the following options to compute the average score:
                - ``micro``: Calculate metric globally from the total count of true
                  positives and false negatives.
                - ``macro``: Calculate metric for each class, and find their
                  unweighted mean. This does not take label imbalance into account.
                - ``weighted``: Calculate metric for each class, and find their
                  average weighted by the support (the number of true instances
                  for each class). This alters "macro" to account for class
                  imbalance.
        zero_division : Literal["warn", 0, 1], default="warn"
            Value to return when there is a zero division. If set to "warn", this
            acts as 0, but warnings are also raised.

    Examples
    --------
        >>> from cyclops.evaluation.metrics import MulticlassRecall
        >>> target = [0, 1, 2, 0]
        >>> preds = [2, 0, 2, 1]
        >>> metric = MulticlassRecall(num_classes=3)
        >>> metric(target, preds)
        array([0., 0., 1.])
        >>> metric.reset_state()
        >>> target = [[0, 1, 2, 0], [2, 1, 2, 0]]
        >>> preds = [
        ...     [[0.1, 0.6, 0.3],
        ...      [0.05, 0.1, 0.85],
        ...      [0.2, 0.7, 0.1],
        ...      [0.9, 0.05, 0.05]],
        ...     [[0.1, 0.6, 0.3],
        ...      [0.05, 0.1, 0.85],
        ...      [0.2, 0.7, 0.1],
        ...      [0.9, 0.05, 0.05]]
        ... ]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        array([0.66666667, 0.        , 0.        ])

    """

    def __init__(
        self,
        num_classes: int,
        top_k: Optional[int] = None,
        average: Literal["micro", "macro", "weighted", None] = None,
        zero_division: Literal["warn", 0, 1] = "warn",
    ) -> None:
        super().__init__(num_classes=num_classes, top_k=top_k, classwise=True)
        self.average = average
        self.zero_division = zero_division

    def compute(self) -> float:
        """Compute the recall score from the state."""
        tp, fp, _, fn = self._final_state()
        return _precision_recall_reduce(
            tp=tp,
            fp=fp,
            fn=fn,
            metric="recall",
            average=self.average,
            zero_division=self.zero_division,
        )


class MultilabelRecall(MultilabelStatScores):
    """Compute the recall score for multilabel classification tasks.

    Parameters
    ----------
        num_labels : int
            Number of labels in the dataset.
        threshold : float, default=0.5
            Threshold for deciding the positive class.
        average : Literal["micro", "macro", "weighted", None], default=None
            If ``None``, return the score for each class. Otherwise,
            use one of the following options to compute the average score:
                - ``micro``: Calculate metric globally from the total count of true
                    positives and false negatives.
                - ``macro``: Calculate metric for each label, and find their
                    unweighted mean. This does not take label imbalance into account.
                - ``weighted``: Calculate metric for each label, and find their
                    average weighted by the support (the number of true instances
                    for each label). This alters "macro" to account for label
                    imbalance.
        zero_division : Literal["warn", 0, 1], default="warn"
            Value to return when there is a zero division. If set to "warn", this
            acts as 0, but warnings are also raised.

    Examples
    --------
        >>> from cyclops.evaluation.metrics import MultilabelRecall
        >>> target = [[0, 1, 0, 1], [0, 0, 1, 1]]
        >>> preds = [[0.1, 0.9, 0.8, 0.2], [0.2, 0.3, 0.6, 0.1]]
        >>> metric = MultilabelRecall(num_labels=4)
        >>> metric(target, preds)
        array([0., 1., 1. , 0. ])
        >>> metric.reset_state()
        >>> target = [[[0, 1, 0, 1], [0, 0, 1, 1]], [[0, 1, 0, 1], [0, 0, 1, 1]]]
        >>> preds = [[[0.1, 0.9, 0.8, 0.2], [0.2, 0.3, 0.6, 0.1]],
        ...          [[0.1, 0.9, 0.8, 0.2], [0.2, 0.3, 0.6, 0.1]]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        array([0., 1., 1., 0.])

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
            num_labels=num_labels,
            threshold=threshold,
            top_k=top_k,
            labelwise=True,
        )
        self.average = average
        self.zero_division = zero_division

    def compute(self) -> float:
        """Compute the recall score from the state."""
        tp, fp, _, fn = self._final_state()
        return _precision_recall_reduce(
            tp=tp,
            fp=fp,
            fn=fn,
            metric="recall",
            average=self.average,
            zero_division=self.zero_division,
        )


class Recall(Metric):
    """Compute the recall score for different types of classification tasks.

    This metric can be used for binary, multiclass, and multilabel classification
    tasks. It creates the appropriate class based on the ``task`` parameter.

    Parameters
    ----------
        task : Literal["binary", "multiclass", "multilabel"]
            Type of classification task.
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
                - ``micro``: Calculate metrics globally by counting the total true
                  positives and false negatives.
                - ``macro``: Calculate metrics for each class/label, and find their
                  unweighted mean. This does not take label imbalance into account.
                - ``weighted``: Calculate metrics for each label/class, and find
                  their average weighted by support (the number of true instances
                  for each label/class). This alters ``macro`` to account for
                  label/class imbalance.
        zero_division : Literal["warn", 0, 1], default="warn"
            Value to return when there is a zero division. If set to "warn", this
            acts as 0, but warnings are also raised.

    Examples (binary)
    -----------------
        >>> from cyclops.evaluation.metrics import Recall
        >>> target = [0, 1, 0, 1]
        >>> preds = [0, 1, 1, 1]
        >>> metric = Recall(task="binary")
        >>> metric(target, preds)
        1.
        >>> metric.reset_state()
        >>> target = [[0, 1, 0, 1], [0, 0, 1, 1]]
        >>> preds = [[0.1, 0.9, 0.8, 0.2], [0.2, 0.3, 0.6, 0.1]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        0.5

    Examples (multiclass)
    ---------------------
        >>> from cyclops.evaluation.metrics import Recall
        >>> target = [0, 1, 2, 0]
        >>> preds = [0, 2, 1, 0]
        >>> metric = Recall(task="multiclass", num_classes=3)
        >>> metric(target, preds)
        array([1. , 0. , 0.])
        >>> metric.reset_state()
        >>> target = [[0, 1, 2, 0], [2, 1, 2, 0]]
        >>> preds = [
        ...     [[0.1, 0.6, 0.3],
        ...      [0.05, 0.1, 0.85],
        ...      [0.2, 0.7, 0.1],
        ...      [0.9, 0.05, 0.05]],
        ...     [[0.1, 0.6, 0.3],
        ...      [0.05, 0.1, 0.85],
        ...      [0.2, 0.7, 0.1],
        ...      [0.9, 0.05, 0.05]]
        ... ]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
       array([0.66666667, 0.        , 0.        ])

    Examples (multilabel)
    ---------------------
        >>> from cyclops.evaluation.metrics import Recall
        >>> target = [[0, 1], [1, 1]]
        >>> preds = [[0.1, 0.9], [0.2, 0.8]]
        >>> metric = Recall(task="multilabel", num_labels=2)
        >>> metric(target, preds)
        array([0., 1.])
        >>> metric.reset_state()
        >>> target = [[[0, 1], [1, 1]], [[1, 1], [1, 0]]]
        >>> preds = [
        ...     [[0.1, 0.7], [0.2, 0.8]],
        ...     [[0.5, 0.9], [0.3, 0.4]]
        ... ]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        array([0.33333333, 1.        ])

    """

    def __new__(  # type: ignore # mypy expects a subclass of Recall
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
        """Create a task-specific metric for computing the recall score."""
        if task == "binary":
            return BinaryRecall(
                threshold=threshold, pos_label=pos_label, zero_division=zero_division
            )
        if task == "multiclass":
            assert (
                isinstance(num_classes, int) and num_classes > 0
            ), "Number of classes must be specified for multiclass classification."
            return MulticlassRecall(
                num_classes=num_classes,
                top_k=top_k,
                average=average,
                zero_division=zero_division,
            )
        if task == "multilabel":
            assert (
                isinstance(num_labels, int) and num_labels > 0
            ), "Number of labels must be specified for multilabel classification."
            return MultilabelRecall(
                num_labels=num_labels,
                threshold=threshold,
                top_k=top_k,
                average=average,
                zero_division=zero_division,
            )
        raise ValueError(
            f"Task '{task}' not supported, expected 'binary', 'multiclass' or "
            f"'multilabel'."
        )
