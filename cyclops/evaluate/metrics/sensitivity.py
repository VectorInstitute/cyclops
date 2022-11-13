"""Classes for computing sensitivity metrics."""

from typing import Literal, Optional

from cyclops.evaluate.metrics.metric import Metric
from cyclops.evaluate.metrics.precision_recall import (
    BinaryRecall,
    MulticlassRecall,
    MultilabelRecall,
)


class BinarySensitivity(BinaryRecall):
    """Computes sensitivity score for binary classification.

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
        >>> from cyclops.evaluation.metrics import BinarySensitivity
        >>> target = [0, 1, 0, 1]
        >>> preds = [0, 1, 1, 0]
        >>> metric = Sensitivity()
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
        super().__init__(
            pos_label=pos_label, threshold=threshold, zero_division=zero_division
        )


class MulticlassSensitivity(MulticlassRecall):
    """Compute the sensitivity score for multiclass classification tasks.

    Parameters
    ----------
        num_classes : int
            Number of classes in the dataset.
        top_k : int, optional
            If given, and predictions are probabilities/logits, the sensitivity will
            be computed only for the top k classes. Otherwise, ``top_k`` will be
            set to 1.
        average : Literal["micro", "macro", "weighted", None], default=None
           If ``None``, return the sensitivity score for each class. Otherwise,
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
        >>> from cyclops.evaluation.metrics import MulticlassSensitivity
        >>> target = [0, 1, 2, 0]
        >>> preds = [2, 0, 2, 1]
        >>> metric = MulticlassSensitivity(num_classes=3)
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
        super().__init__(
            num_classes=num_classes,
            top_k=top_k,
            average=average,
            zero_division=zero_division,
        )


class MultilabelSensitivity(MultilabelRecall):
    """Compute the sensitivity score for multilabel classification tasks.

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
        >>> from cyclops.evaluation.metrics import MultilabelSensitivity
        >>> target = [[0, 1, 0, 1], [0, 0, 1, 1]]
        >>> preds = [[0.1, 0.9, 0.8, 0.2], [0.2, 0.3, 0.6, 0.1]]
        >>> metric = MultilabelSensitivity(num_labels=4)
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
            average=average,
            zero_division=zero_division,
        )


class Sensitivity(Metric):
    """Compute the sensitivity score for different types of classification tasks.

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
            If ``None``, return the sensitivity score for each label/class. Otherwise,
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
        >>> from cyclops.evaluation.metrics import Sensitivity
        >>> target = [0, 1, 0, 1]
        >>> preds = [0, 1, 1, 1]
        >>> metric = Sensitivity(task="binary")
        >>> metric.update_state(target, preds)
        >>> metric.compute()
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
        >>> from cyclops.evaluation.metrics import Sensitivity
        >>> target = [0, 1, 2, 0]
        >>> preds = [0, 2, 1, 0]
        >>> metric = Sensitivity(task="multiclass", num_classes=3)
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
        >>> from cyclops.evaluation.metrics import Sensitivity
        >>> target = [[0, 1], [1, 1]]
        >>> preds = [[0.1, 0.9], [0.2, 0.8]]
        >>> metric = Sensitivity(task="multilabel", num_labels=2)
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

    def __new__(  # type: ignore # mypy expects a subclass of Sensitivity
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
        """Create a task-specific metric for computing the sensitivity score."""
        if task == "binary":
            return BinarySensitivity(
                threshold=threshold, pos_label=pos_label, zero_division=zero_division
            )
        if task == "multiclass":
            assert (
                isinstance(num_classes, int) and num_classes > 0
            ), "Number of classes must be specified for multiclass classification."
            return MulticlassSensitivity(
                num_classes=num_classes,
                top_k=top_k,
                average=average,
                zero_division=zero_division,
            )
        if task == "multilabel":
            assert (
                isinstance(num_labels, int) and num_labels > 0
            ), "Number of labels must be specified for multilabel classification."
            return MultilabelSensitivity(
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
