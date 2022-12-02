"""Classes for computing the F-beta score."""

from typing import Literal, Optional

from cyclops.evaluate.metrics.functional.f_beta import _check_beta, _fbeta_reduce
from cyclops.evaluate.metrics.metric import Metric
from cyclops.evaluate.metrics.stat_scores import (
    BinaryStatScores,
    MulticlassStatScores,
    MultilabelStatScores,
)
from cyclops.evaluate.metrics.utils import _check_average_arg


class BinaryFbetaScore(BinaryStatScores):
    """Compute the F-beta score for binary classification tasks.

    Parameters
    ----------
        beta : float
            Weight of precision in harmonic mean.
        pos_label : int, default=1
            The positive class label. One of [0, 1].
        threshold : float, default=0.5
            Threshold value for converting probabilities and logits to binary.
            Logits will be converted to probabilities using the sigmoid function.
        zero_division : Literal["warn", 0, 1], default="warn"
            Value to return when there are no true positives or true negatives.
            If set to ``warn``, this acts as 0, but warnings are also raised.

    Examples
    --------
        >>> from cyclops.evaluation.metrics import BinaryFbetaScore
        >>> target = [0, 1, 1, 0]
        >>> preds = [0, 1, 0, 0]
        >>> metric = BinaryFbetaScore(beta=0.5)
        >>> metric(target, preds)
        0.8333333333333334
        >>> metric.reset_state()
        >>> target = [[1, 0, 1, 0], [1, 0, 0, 1]]
        >>> preds = [[0.2, 0.8, 0.3, 0.4], [0.6, 0.3, 0.1, 0.5]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        0.625

    """

    def __init__(
        self,
        beta: float,
        pos_label: int = 1,
        threshold: float = 0.5,
        zero_division: Literal["warn", 0, 1] = "warn",
    ) -> None:
        super().__init__(threshold=threshold, pos_label=pos_label)

        _check_beta(beta=beta)

        self.beta = beta
        self.zero_division = zero_division

    def compute(self) -> float:
        """Compute the metric from the state."""
        tp, fp, _, fn = self._final_state()
        return _fbeta_reduce(
            tp=tp,
            fp=fp,
            fn=fn,
            beta=self.beta,
            average=None,
            zero_division=self.zero_division,
        )


class MulticlassFbetaScore(MulticlassStatScores):
    """Compute the F-beta score for multiclass classification tasks.

    Parameters
    ----------
        beta : float
            Weight of precision in harmonic mean.
        num_classes : int
            The number of classes in the dataset.
        top_k : int, optional
            If given, and predictions are probabilities/logits, the score will
            be computed only for the top k classes. Otherwise, ``top_k`` will be
            set to 1.
        average : Literal["micro", "macro", "weighted", None], default=None
           If ``None``, return the score for each class. Otherwise,
           use one of the following options to compute the average score:
                - ``micro``: Calculate metric globally.
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
        >>> from cyclops.evaluation.metrics import MulticlassFbetaScore
        >>> target = [0, 1, 2, 0]
        >>> preds = [0, 2, 1, 0]
        >>> metric = MulticlassFbetaScore(beta=0.5, num_classes=3)
        >>> metric(target, preds)
        array([1., 0., 0.])
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
        array([0.90909091, 0.        , 0.        ])

    """

    def __init__(
        self,
        beta: float,
        num_classes: int,
        top_k: Optional[int] = None,
        average: Literal["micro", "macro", "weighted", None] = None,
        zero_division: Literal["warn", 0, 1] = "warn",
    ) -> None:
        super().__init__(num_classes=num_classes, top_k=top_k, classwise=True)

        _check_beta(beta=beta)
        _check_average_arg(average=average)

        self.beta = beta
        self.average = average
        self.zero_division = zero_division

    def compute(self) -> float:
        """Compute the metric from the state."""
        tp, fp, _, fn = self._final_state()
        return _fbeta_reduce(
            tp=tp,
            fp=fp,
            fn=fn,
            beta=self.beta,
            average=self.average,
            zero_division=self.zero_division,
        )


class MultilabelFbetaScore(MultilabelStatScores):
    """Compute the F-beta score for multilabel classification tasks.

    Parameters
    ----------
        beta : float
            Weight of precision in harmonic mean.
        num_labels : int
            Number of labels for the task.
        threshold : float, default=0.5
            Threshold for deciding the positive class if predicitions are logits
            or probability scores. Logits will be converted to probabilities using
            the sigmoid function.
        top_k : int, optional
            If given, and predictions are probabilities/logits, the score will
            be computed only for the top k classes. Otherwise, ``top_k`` will be
            set to 1.
        average : Literal["micro", "macro", "weighted", None], default=None
            If ``None``, return the score for each label. Otherwise,
            use one of the following options to compute the average score:
                - ``micro``: Calculate metric globally.
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
        >>> from cyclops.evaluation.metrics import MultilabelFbetaScore
        >>> target = [[0, 1], [1, 1]]
        >>> preds = [[0.1, 0.9], [0.8, 0.2]]
        >>> metric = MultilabelFbetaScore(beta=0.5, num_labels=2)
        >>> metric(target, preds)
        array([1.        , 0.83333333])
        >>> metric.reset_state()
        >>> target = [[[0, 1], [1, 1]], [[1, 1], [1, 0]]]
        >>> preds = [[[0, 1], [1, 0]], [[1, 1], [1, 0]]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        array([1.        , 0.90909091])

    """

    def __init__(
        self,
        beta: float,
        num_labels: int,
        threshold: float = 0.5,
        top_k: Optional[int] = None,
        average: Literal["micro", "macro", "weighted", None] = None,
        zero_division: Literal["warn", 0, 1] = "warn",
    ) -> None:
        super().__init__(
            num_labels=num_labels, threshold=threshold, top_k=top_k, labelwise=True
        )

        _check_beta(beta=beta)
        _check_average_arg(average=average)

        self.beta = beta
        self.average = average
        self.zero_division = zero_division

    def compute(self) -> float:
        """Compute the metric from the state."""
        tp, fp, _, fn = self._final_state()
        return _fbeta_reduce(
            tp=tp,
            fp=fp,
            fn=fn,
            beta=self.beta,
            average=self.average,
            zero_division=self.zero_division,
        )


class FbetaScore(Metric):
    """Compute the F-beta score for different types of classification tasks.

    Parameters
    ----------
        beta : float
            Weight of precision in harmonic mean.
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
        >>> from cyclops.evaluation.metrics import FbetaScore
        >>> target = [0, 1, 1, 0]
        >>> preds = [0.1, 0.8, 0.4, 0.3]
        >>> metric = FbetaScore(beta=0.5, task="binary")
        >>> metric(target, preds)
        0.8333333333333334
        >>> metric.reset_state()
        >>> target = [[0, 1], [1, 1]]
        >>> preds = [[0.1, 0.9], [0.8, 0.2]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        0.9090909090909091

    Examples (multiclass)
    ---------------------
        >>> from cyclops.evaluation.metrics import FbetaScore
        >>> target = [0, 1, 2, 0]
        >>> preds = [[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]]
        >>> metric = FbetaScore(beta=0.5, task="multiclass", num_classes=3)
        >>> metric(target, preds)
        array([0.83333333, 0.        , 0.55555556])
        >>> metric.reset_state()
        >>> target = [[0, 1, 0], [0, 0, 1]]
        >>> preds = [[[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]],
        ...          [[0.1, 0.1, 0.8], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        array([0.83333333, 0.5       , 0.        ])

    Examples (multilabel)
    ---------------------
        >>> from cyclops.evaluation.metrics import FbetaScore
        >>> target = [[0, 1], [1, 1]]
        >>> preds = [[0.1, 0.9], [0.8, 0.2]]
        >>> metric = FbetaScore(beta=0.5, task="multilabel", num_labels=2)
        >>> metric(target, preds)
        array([1.        , 0.83333333])
        >>> metric.reset_state()
        >>> target = [[[0, 1], [1, 1]], [[1, 1], [1, 0]]]
        >>> preds = [[[0, 1], [1, 0]], [[1, 1], [1, 0]]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        array([1.        , 0.90909091])

    """

    def __new__(  # type: ignore # mypy expects a subclass of FbetaScore
        cls,
        beta: float,
        task: Literal["binary", "multiclass", "multilabel"],
        pos_label: int = 1,
        num_classes: int = None,
        threshold: float = 0.5,
        top_k: Optional[int] = None,
        num_labels: int = None,
        average: Literal["micro", "macro", "weighted", None] = None,
        zero_division: Literal["warn", 0, 1] = "warn",
    ) -> Metric:
        """Create a task-specific FbetaScore instance."""
        if task == "binary":
            return BinaryFbetaScore(
                beta=beta,
                threshold=threshold,
                pos_label=pos_label,
                zero_division=zero_division,
            )
        if task == "multiclass":
            assert (
                isinstance(num_classes, int) and num_classes > 0
            ), "Number of classes must be specified for multiclass classification."
            return MulticlassFbetaScore(
                beta=beta,
                num_classes=num_classes,
                top_k=top_k,
                average=average,
                zero_division=zero_division,
            )
        if task == "multilabel":
            assert (
                isinstance(num_labels, int) and num_labels > 0
            ), "Number of labels must be specified for multilabel classification."
            return MultilabelFbetaScore(
                beta=beta,
                num_labels=num_labels,
                threshold=threshold,
                average=average,
                zero_division=zero_division,
            )
        raise ValueError(
            f"Task {task} is not supported, expected one of 'binary', 'multiclass'"
            " or 'multilabel'"
        )


class BinaryF1Score(BinaryFbetaScore):
    """Compute the F1 score for binary classification tasks.

    Parameters
    ----------
        pos_label: int, default=1
            The label of the positive class.
        threshold : float, default=0.5
            Threshold value for binarizing predictions in form of logits or
            probability scores.
        zero_division : Literal["warn", 0, 1], default="warn"
            Value to return when there is a zero division. If set to "warn", this
            acts as 0, but warnings are also raised.

    Examples
    --------
        >>> from cyclops.evaluation.metrics import BinaryF1Score
        >>> target = [0, 1, 1, 0]
        >>> preds = [0.1, 0.8, 0.4, 0.3]
        >>> metric = BinaryF1Score()
        >>> metric(target, preds)
        0.6666666666666666
        >>> metric.reset_state()
        >>> target = [[0, 1], [1, 1]]
        >>> preds = [[0.1, 0.9], [0.8, 0.2]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        0.8

    """

    def __init__(
        self,
        pos_label: int = 1,
        threshold: float = 0.5,
        zero_division: Literal["warn", 0, 1] = "warn",
    ) -> None:
        super().__init__(
            beta=1.0,
            threshold=threshold,
            pos_label=pos_label,
            zero_division=zero_division,
        )


class MulticlassF1Score(MulticlassFbetaScore):
    """Compute the F1 score for multiclass classification tasks.

    Parameters
    ----------
        num_classes : int
            Number of classes in the dataset.
        top_k : int, optional
            If given, and predictions are probabilities/logits, the precision will
            be computed only for the top k classes. Otherwise, ``top_k`` will be
            set to 1.
        average : Literal["micro", "macro", "weighted", None], default=None
           If ``None``, return the score for each class. Otherwise, use one of
           the following options to compute the average score:
                - ``micro``: Calculate metric globally.
                - ``macro``: Calculate metric for each class, and find their
                  unweighted mean. This does not take class imbalance into account.
                - ``weighted``: Calculate metric for each class, and find their
                  average weighted by the support (the number of true instances
                  for each class). This alters "macro" to account for class
                  imbalance. It can result in an F-score that is not between
                  precision and recall.
        zero_division : Literal["warn", 0, 1], default="warn"
            Value to return when there is a zero division. If set to "warn", this
            acts as 0, but warnings are also raised.

    Examples
    --------
        >>> from cyclops.evaluation.metrics import MulticlassF1Score
        >>> target = [0, 1, 2, 0]
        >>> preds = [[0.1, 0.6, 0.3], [0.05, 0.95, 0], [0.1, 0.8, 0.1], [0.95, 0.05, 0]]
        >>> metric = MulticlassF1Score(num_classes=3)
        >>> metric(target, preds)
        array([0.66666667, 0.5       , 0.        ])
        >>> metric.reset_state()
        >>> target = [[0, 1], [1, 1]]
        >>> preds = [[[0.1, 0.9, 0], [0.05, 0.95, 0]],
        ...         [[0.1, 0.8, 0.1], [0.05, 0.95, 0]]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        array([0.        , 0.85714286, 0.        ])

    """

    def __init__(
        self,
        num_classes: int,
        top_k: Optional[int] = None,
        average: Literal["micro", "macro", "weighted", None] = None,
        zero_division: Literal["warn", 0, 1] = "warn",
    ) -> None:
        super().__init__(
            beta=1.0,
            num_classes=num_classes,
            top_k=top_k,
            average=average,
            zero_division=zero_division,
        )


class MultilabelF1Score(MultilabelFbetaScore):
    """Compute the F1 score for multilabel classification tasks.

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
            If ``None``, return the score for each label. Otherwise, use one of
            the following options to compute the average score:
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

    Returns
    -------
        float or numpy.ndarray
            The F1 score. If ``average`` is ``None``, a numpy.ndarray of shape
            (``num_labels``,) is returned.

    Examples
    --------
        >>> from cyclops.evaluation.metrics import MultilabelF1Score
        >>> target = [[0, 1, 1], [1, 0, 0]]
        >>> preds = [[0.1, 0.9, 0.8], [0.05, 0.1, 0.2]]
        >>> metric = MultilabelF1Score(num_labels=3)
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
        array([0. , 0.8, 0. ])

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
            beta=1.0,
            num_labels=num_labels,
            threshold=threshold,
            top_k=top_k,
            average=average,
            zero_division=zero_division,
        )


class F1Score(FbetaScore):
    """Compute the F1 score for different types of classification tasks.

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
        >>> from cyclops.evaluation.metrics import F1Score
        >>> target = [0, 1, 1, 0]
        >>> preds = [0.1, 0.8, 0.4, 0.3]
        >>> metric = F1Score(task="binary")
        >>> metric(target, preds)
        0.6666666666666666
        >>> metric.reset_state()
        >>> target = [[0, 1], [1, 1]]
        >>> preds = [[0.1, 0.9], [0.8, 0.2]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        0.8

    Examples (multiclass)
    ---------------------
        >>> from cyclops.evaluation.metrics import F1Score
        >>> target = [0, 1, 2, 0]
        >>> preds = [[0.1, 0.6, 0.3], [0.05, 0.95, 0], [0.1, 0.8, 0.1], [0.95, 0.05, 0]]
        >>> metric = F1Score(task="multiclass", num_classes=3)
        >>> metric(target, preds)
        array([0.66666667, 0.5       , 0.        ])
        >>> metric.reset_state()
        >>> target = [[0, 1], [1, 1]]
        >>> preds = [[[0.1, 0.9, 0], [0.05, 0.95, 0]],
        ...         [[0.1, 0.8, 0.1], [0.05, 0.95, 0]]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        array([0.        , 0.85714286, 0.        ])


    Examples (multilabel)
    ---------------------
        >>> from cyclops.evaluation.metrics import F1Score
        >>> target = [[0, 1, 1], [1, 0, 0]]
        >>> preds = [[0.1, 0.9, 0.8], [0.05, 0.1, 0.2]]
        >>> metric = F1Score(task="multilabel", num_labels=3)
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
        array([0. , 0.8, 0. ])

    """

    def __new__(  # type: ignore # mypy expects a subclass of F1Score
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
        """Create a task-specific F1 score instance."""
        if task == "binary":
            return BinaryF1Score(
                threshold=threshold,
                pos_label=pos_label,
                zero_division=zero_division,
            )
        if task == "multiclass":
            assert (
                isinstance(num_classes, int) and num_classes > 0
            ), "Number of classes must be specified for multiclass classification."
            return MulticlassF1Score(
                num_classes=num_classes,
                top_k=top_k,
                average=average,
                zero_division=zero_division,
            )
        if task == "multilabel":
            assert (
                isinstance(num_labels, int) and num_labels > 0
            ), "Number of labels must be specified for multilabel classification."
            return MultilabelF1Score(
                num_labels=num_labels,
                threshold=threshold,
                average=average,
                zero_division=zero_division,
            )
        raise ValueError(
            f"Task {task} is not supported, expected one of 'binary', 'multiclass'"
            " or 'multilabel'"
        )
