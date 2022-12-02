"""Classes for computing specificity metrics."""

from typing import Literal, Optional

from cyclops.evaluate.metrics.functional.specificity import _specificity_reduce
from cyclops.evaluate.metrics.metric import Metric
from cyclops.evaluate.metrics.stat_scores import (
    BinaryStatScores,
    MulticlassStatScores,
    MultilabelStatScores,
)
from cyclops.evaluate.metrics.utils import _check_average_arg


class BinarySpecificity(BinaryStatScores):
    """Compute specificity for binary classification tasks.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Estimated targets (predictions) as returned by a classifier.
        pos_label : int, default=1
            The label to use for the positive class.
        threshold : float, default=0.5
            The threshold to use for converting the predictions to binary
            values. Logits will be converted to probabilities using the sigmoid
            function.

    Examples
    --------
        >>> from cyclops.evaluation.metrics import BinarySpecificity
        >>> target = [0, 1, 1, 0]
        >>> preds = [0, 1, 0, 0]
        >>> metric = BinarySpecificity()
        >>> metric(target, preds)
        1.0
        >>> metric.reset_state()
        >>> target = [[0, 1, 1, 0], [1, 1, 0, 0]]
        >>> preds = [[0, 1, 0, 0], [1, 0, 0, 0]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        1.0

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
        """Compute the specificity score from the state."""
        # pylint: disable=invalid-name # for tp, tn, fp, fn
        tp, fp, tn, fn = self._final_state()
        return _specificity_reduce(
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            average=None,
            zero_division=self.zero_division,
        )


class MulticlassSpecificity(MulticlassStatScores):
    """Compute specificity for multiclass classification tasks.

    Parameters
    ----------
        num_classes : int
            The number of classes in the dataset.
        top_k : int, optional
            Number of highest probability or logit score predictions considered
            to find the correct label. Only works when ``preds`` contain
            probabilities/logits.
        average : Literal["micro", "macro", "weighted", None], default=None
            If None, return the specificity for each class, otherwise return the
            average specificity. Average options are:
                - ``micro``: Calculate metrics globally.
                - ``macro``: Calculate metrics for each class, and find their unweighted
                  mean. This does not take class imbalance into account.
                - ``weighted``: Calculate metrics for each class, and find their
                  average, weighted by support (the number of true instances for each
                  label).
        zero_division : Literal["warn", 0, 1], default="warn"
            Sets the value to return when there is a zero division. If set to ``warn``,
            this acts as 0, but warnings are also raised.

    Examples
    --------
        >>> from cyclops.evaluation.metrics import MulticlassSpecificity
        >>> target = [0, 1, 2, 0, 1, 2]
        >>> preds = [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.2, 0.75],
        ...          [0.35, 0.5, 0.15], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]
        >>> metric = MulticlassSpecificity(num_classes=3)
        >>> metric(target, preds)
        array([1.  , 0.75, 1.  ])
        >>> metric.reset_state()
        >>> target = [[0, 1, 2, 0, 1, 2], [1, 1, 2, 0, 0, 1]]
        >>> preds = [[0, 2, 1, 2, 0, 1], [1, 0, 1, 2, 2, 0]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        array([0.625     , 0.57142857, 0.55555556])

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
        """Compute the specificity score from the state."""
        tp, fp, tn, fn = self._final_state()
        return _specificity_reduce(
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            average=self.average,
            zero_division=self.zero_division,
        )


class MultilabelSpecificity(MultilabelStatScores):
    """Compute specificity for multilabel classification tasks.

    Parameters
    ----------
        num_labels : int
            The number of labels in the dataset.
        threshold : float, default=0.5
            The threshold value for converting probability or logit scores to
            binary. A sigmoid function is first applied to logits to convert them
            to probabilities.
        top_k : int, optional
            Number of highest probability or logit score predictions considered
            to find the correct label. Only works when ``preds`` contains
            probabilities/logits.
        average : Literal["micro", "macro", "weighted", None], default=None
            If None, return the specificity for each class, otherwise return the
            average specificity. Average options are:
            - ``micro``: Calculate metrics globally.
            - ``macro``: Calculate metrics for each label, and find their unweighted
              mean. This does not take label imbalance into account.
            - ``weighted``: Calculate metrics for each label, and find their average,
              weighted by support (the number of true instances for each label).
        zero_division : Literal["warn", 0, 1], default="warn"
            Sets the value to return when there is a zero division. If set to ``warn``,
            this acts as 0, but warnings are also raised.

    Examples
    --------
        >>> from cyclops.evaluation.metrics import MultilabelSpecificity
        >>> target = [[0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0]]
        >>> preds = [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.2, 0.75],
        ...          [0.35, 0.5, 0.15], [0.05, 0.9, 0.05]]
        >>> metric = MultilabelSpecificity(num_labels=3)
        >>> metric(target, preds)
        array([0.5, 0. , 0.5])
        >>> metric.reset_state()
        >>> target = [[[0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0]],
        ...           [[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0]]]
        >>> preds = [[[1, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 0]],
        ...          [[0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0]]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        array([0.5       , 0.66666667, 0.6       ])

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
        """Compute the specificity score from the state."""
        tp, fp, tn, fn = self._final_state()
        return _specificity_reduce(
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            average=self.average,
            zero_division=self.zero_division,
        )


class Specificity(Metric):
    """Compute specificity score for different classification tasks.

    The specificity is the ratio of true negatives to the sum of true negatives and
    false positives. It is also the recall of the negative class.

    Parameters
    ----------
        task : Literal["binary", "multiclass", "multilabel"]
            Type of classification task.
        pos_label : int, default=1
            Label to consider as positive for binary classification tasks.
        num_classes : int
            Number of classes for the task. Required if ``task`` is ``"multiclass"``.
        threshold : float, default=0.5
            Threshold for deciding the positive class. Only used if ``task`` is
            ``"binary"`` or ``"multilabel"``.
        top_k : int, optional
            If given, and predictions are probabilities/logits, the precision will
            be computed only for the top k classes. Otherwise, ``top_k`` will be
            set to 1. Only used if ``task`` is ``"multiclass"`` or ``"multilabel"``.
        num_labels : int
            Number of labels for the task. Required if ``task`` is ``"multilabel"``.
        average : Literal["micro", "macro", "weighted", None], default=None
            If ``None``, return the score for each label/class. Otherwise,
            use one of the following options to compute the average score:
                - ``micro``: Calculate metrics globally.
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
        >>> from cyclops.evaluation.metrics import Specificity
        >>> target = [0, 1, 1, 0, 1]
        >>> preds = [0.9, 0.05, 0.05, 0.35, 0.05]
        >>> metric = Specificity(task="binary")
        >>> metric(target, preds)
        0.5
        >>> metric.reset_state()
        >>> target = [[0, 1, 1], [1, 0, 1]]
        >>> preds = [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        0.0

    Examples (multiclass)
    ---------------------
        >>> from cyclops.evaluation.metrics import Specificity
        >>> target = [0, 1, 2, 0, 1, 2]
        >>> preds = [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.2, 0.75],
        ...          [0.35, 0.5, 0.15], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]
        >>> metric = Specificity(task="multiclass", num_classes=3)
        >>> metric(target, preds)
        array([1.  , 0.75, 1.  ])
        >>> metric.reset_state()
        >>> target = [[0, 1, 1], [1, 2, 1]]
        >>> preds = [[[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.2, 0.75]],
        ...          [[0.35, 0.5, 0.15], [0.25, 0.5, 0.25], [0.5, 0.05, 0.45]]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        array([0.8, 0.5, 0.8])

    Examples (multilabel)
    ---------------------
        >>> from cyclops.evaluation.metrics import Specificity
        >>> target = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        >>> preds = [[0.9, 0.05, 0.05], [0.05, 0.2, 0.75], [0.35, 0.5, 0.15]]
        >>> metric = Specificity(task="multilabel", num_labels=3)
        >>> metric(target, preds)
        array([0., 1., 1.])
        >>> metric.reset_state()
        >>> target = [[[0, 1, 0], [1, 0, 1]], [[0, 1, 1], [1, 0, 0]]]
        >>> preds = [
        ...     [[0.1, 0.7, 0.2], [0.2, 0.8, 0.3]],
        ...     [[0.5, 0.9, 0.0], [0.3, 0.4, 0.2]],
        ... ]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        array([0.5, 0.5, 1. ])

    """

    def __new__(  # type: ignore # mypy expects a subclass of Specificity
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
        """Create task-specific instance of the metric."""
        if task == "binary":
            return BinarySpecificity(
                threshold=threshold,
                pos_label=pos_label,
                zero_division=zero_division,
            )
        if task == "multiclass":
            assert (
                isinstance(num_classes, int) and num_classes > 0
            ), "Number of classes must be specified for multiclass classification."
            return MulticlassSpecificity(
                num_classes=num_classes,
                top_k=top_k,
                average=average,
                zero_division=zero_division,
            )
        if task == "multilabel":
            assert (
                isinstance(num_labels, int) and num_labels > 0
            ), "Number of labels must be specified for multilabel classification."
            return MultilabelSpecificity(
                num_labels=num_labels,
                threshold=threshold,
                average=average,
                zero_division=zero_division,
            )
        raise ValueError(
            f"Task {task} is not supported, expected one of 'binary', 'multiclass'"
            " or 'multilabel'"
        )
