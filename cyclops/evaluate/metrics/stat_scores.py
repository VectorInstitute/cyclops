"""Classes for computing stat scores."""

from typing import Callable, Literal, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from cyclops.evaluate.metrics.functional.stat_scores import (
    _binary_stat_scores_args_check,
    _binary_stat_scores_format,
    _binary_stat_scores_update,
    _multiclass_stat_scores_format,
    _multiclass_stat_scores_update,
    _multilabel_stat_scores_format,
    _multilabel_stat_scores_update,
    _stat_scores_compute,
)
from cyclops.evaluate.metrics.metric import Metric

# mypy: ignore-errors


class _AbstractScores(Metric):
    """Abstract base class for classes that compute stat scores."""

    def _create_state(self, size: int = 1) -> None:
        """Create the state variables.

        For the stat scores, the state variables are the true positives (tp),
        false positives (fp), true negatives (tn), and false negatives (fn).

        Parameters
        ----------
            size : int
                The size of the default numpy.ndarray to create for the state
                variables.

        Returns
        -------
            None

        Raises
        ------
            AssertionError
                If ``size`` is not greater than 0.

        """
        assert size > 0, "``size`` must be greater than 0."
        default: Callable = lambda: np.zeros(shape=size, dtype=np.int_)

        self.add_state("tp", default())
        self.add_state("fp", default())
        self.add_state("tn", default())
        self.add_state("fn", default())

    def _update_state(
        self, tp: np.ndarray, fp: np.ndarray, tn: np.ndarray, fn: np.ndarray
    ) -> None:
        """Update the state variables.

        Parameters
        ----------
            tp : numpy.ndarray
                The true positives.
            fp : numpy.ndarray
                The false positives.
            tn : numpy.ndarray
                The true negatives.
            fn : numpy.ndarray
                The false negatives.

        Returns
        -------
            None

        """
        # pylint: disable=no-member
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def _final_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the final state variables.

        Returns
        -------
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
                The final state variables. The order is true positives (tp),
                false positives (fp), true negatives (tn), and false negatives

        """
        # pylint: disable=no-member
        tp = self.tp
        fp = self.fp
        tn = self.tn
        fn = self.fn
        return tp, fp, tn, fn


class BinaryStatScores(_AbstractScores):
    """Compute binary stat scores.

    Parameters
    ----------
        pos_label : int, default=1
            The label to use for the positive class.
        threshold : float, default=0.5
            The threshold to use for converting the predictions to binary
            values. Logits will be converted to probabilities using the sigmoid
            function.

    Examples
    --------
        >>> from cyclops.evaluation.metrics import BinaryStatScores
        >>> target = [0, 1, 1, 0]
        >>> preds = [0, 1, 0, 0]
        >>> metric = BinaryStatScores(threshold=0.5, pos_label=1)
        >>> metric(target=target, preds=preds)
        array([1, 0, 2, 1, 2])
        >>> metric.reset_state()
        >>> target = [[1, 1, 0, 1, 0, 0], [0, 0, 1, 1, 0, 0]]
        >>> preds = [[0.9, 0.8, 0.3, 0.4, 0.5, 0.2], [0.2, 0.3, 0.6, 0.9, 0.4, 0.8]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(target=t, preds=p)
        >>> metric.compute()
        array([4, 2, 5, 1, 5])

    """

    def __init__(self, pos_label: int = 1, threshold: float = 0.5) -> None:
        super(_AbstractScores, self).__init__()

        _binary_stat_scores_args_check(threshold=threshold, pos_label=pos_label)

        self.threshold = threshold
        self.pos_label = pos_label

        self._create_state(size=1)

    def update_state(  # pylint: disable=arguments-differ
        self, target: ArrayLike, preds: ArrayLike
    ) -> None:  # type: ignore
        """Update the state variables."""
        target, preds = _binary_stat_scores_format(
            target, preds, threshold=self.threshold
        )

        tp, fp, tn, fn = _binary_stat_scores_update(
            target, preds, pos_label=self.pos_label
        )
        self._update_state(tp, fp, tn, fn)

    def compute(self) -> np.ndarray:
        """Compute the binary stat scores from the state variables.

        Returns
        -------
            numpy.ndarray
                The binary stat scores. The order is true positives (tp),
                false positives (fp), true negatives (tn), false negatives
                (fn) and support (tp + fn).

        """
        tp, fp, tn, fn = self._final_state()
        return _stat_scores_compute(tp=tp, fp=fp, tn=tn, fn=fn)


class MulticlassStatScores(_AbstractScores):
    """Compute multiclass stat scores.

    Parameters
    ----------
        num_classes : int
            The total number of classes for the problem.
        top_k : Optional[int], default=None
            If given, and predictions are probabilities/logits, the score will
            be computed only for the top k classes. Otherwise, ``top_k`` will be
            set to 1.
        classwise : bool, default=True
            Whether to return the stat scores for each class or sum over all
            classes.

    Examples
    --------
        >>> from cyclops.evaluation.metrics import MulticlassStatScores
        >>> target = [0, 1, 2, 2, 2]
        >>> preds = [0, 2, 1, 2, 0]
        >>> metric = MulticlassStatScores(num_classes=3, classwise=True)
        >>> metric(target=target, preds=preds)
        array([[1, 1, 3, 0, 1],
                [0, 1, 3, 1, 1],
                [1, 1, 1, 2, 3]])
        >>> metric.reset_state()
        >>> target = [[2, 0, 2, 2, 1], [1, 1, 0, 2, 2]]
        >>> preds = [
        ...         [
        ...             [0.1, 0.2, 0.6],
        ...             [0.6, 0.1, 0.2],
        ...             [0.2, 0.6, 0.1],
        ...             [0.2, 0.6, 0.1],
        ...             [0.6, 0.2, 0.1],
        ...         ],
        ...         [
        ...             [0.05, 0.1, 0.6],
        ...             [0.1, 0.05, 0.6],
        ...             [0.6, 0.1, 0.05],
        ...             [0.1, 0.6, 0.05],
        ...             [0.1, 0.6, 0.05],
        ...         ],
        ...     ]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(target=t, preds=p)
        >>> metric.compute()
        array([[2, 1, 7, 0, 2],
                [0, 4, 3, 3, 3],
                [1, 2, 3, 4, 5]])

    """

    def __init__(
        self, num_classes: int, top_k: Optional[int] = None, classwise: bool = True
    ) -> None:
        super(_AbstractScores, self).__init__()

        assert num_classes > 1, "``num_classes`` must be greater than 1"

        self.num_classes = num_classes
        self.top_k = top_k
        self.classwise = classwise

        self._create_state(size=1 if not classwise else num_classes)

    def update_state(  # pylint: disable=arguments-differ
        self, target: ArrayLike, preds: ArrayLike
    ) -> None:  # type: ignore
        """Update the state variables."""
        target, preds = _multiclass_stat_scores_format(
            target, preds, num_classes=self.num_classes, top_k=self.top_k
        )
        tp, fp, tn, fn = _multiclass_stat_scores_update(
            target,
            preds,
            num_classes=self.num_classes,
            classwise=self.classwise,
        )
        self._update_state(tp, fp, tn, fn)

    def compute(self) -> np.ndarray:
        """Compute the multiclass stat scores from the state variables.

        Returns
        -------
            numpy.ndarray
                The multiclass stat scores. The order is true positives (tp),
                false positives (fp), true negatives (tn), false negatives
                (fn) and support (tp + fn). If ``classwise`` is ``True``, the
                shape is ``(num_classes, 5)``. Otherwise, the shape is ``(5,)``.

        """
        tp, fp, tn, fn = self._final_state()
        return _stat_scores_compute(tp=tp, fp=fp, tn=tn, fn=fn)


class MultilabelStatScores(_AbstractScores):
    """Compute stat scores for multilabel problems.

    Parameters
    ----------
        threshold : float, default=0.5
            Threshold value for binarizing predictions that are probabilities or
            logits. A sigmoid function is applied if the predictions are logits.
        top_k : int, default=None
            If given, and predictions are probabilities/logits, the score will
            be computed only for the top k classes. Otherwise, ``top_k`` will be
            set to 1.
        labelwise : bool, default=False
            Whether to return the stat scores for each label or sum over all labels.

    Examples
    --------
        >>> from cyclops.evaluation.metrics import MultilabelStatScores
        >>> target = [[0, 1, 1], [1, 0, 1]]
        >>> preds = [[0.1, 0.9, 0.8], [0.8, 0.2, 0.7]]
        >>> metric = MultilabelStatScores(num_labels=3, labelwise=True)
        >>> metric(target=target, preds=preds)
        array([[1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1],
                [2, 0, 0, 0, 2]])
        >>> metric.reset_state()
        >>> target = [[[0, 1, 1], [1, 0, 1]], [[0, 0, 1], [1, 1, 1]]]
        >>> preds = [[[0.1, 0.9, 0.8], [0.8, 0.2, 0.7]],
        ...         [[0.1, 0.9, 0.8], [0.8, 0.2, 0.7]]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(target=t, preds=p)
        >>> metric.compute()
        array([[2, 0, 2, 0, 2],
                [1, 1, 1, 1, 2],
                [4, 0, 0, 0, 4]])

    """

    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        top_k: int = None,
        labelwise: bool = False,
    ) -> None:
        super().__init__()

        _binary_stat_scores_args_check(threshold=threshold, pos_label=1)

        self.num_labels = num_labels
        self.threshold = threshold
        self.top_k = top_k
        self.labelwise = labelwise

        self._create_state(size=1 if not self.labelwise else num_labels)

    def update_state(  # pylint: disable=arguments-differ
        self, target: ArrayLike, preds: ArrayLike
    ) -> None:  # type: ignore
        """Update the state variables."""
        target, preds = _multilabel_stat_scores_format(
            target,
            preds,
            num_labels=self.num_labels,
            threshold=self.threshold,
            top_k=self.top_k,
        )
        tp, fp, tn, fn = _multilabel_stat_scores_update(
            target,
            preds,
            num_labels=self.num_labels,
            labelwise=self.labelwise,
        )
        self._update_state(tp, fp, tn, fn)

    def compute(self) -> np.ndarray:
        """Compute the multilabel stat scores from the state variables.

        Returns
        -------
            numpy.ndarray
                The multilabel stat scores. The order is true positives (tp),
                false positives (fp), true negatives (tn), false negatives
                (fn) and support (tp + fn). If ``labelwise`` is ``True``, the
                shape is ``(num_labels, 5)``. Otherwise, the shape is ``(5,)``.

        """
        tp, fp, tn, fn = self._final_state()
        return _stat_scores_compute(tp=tp, fp=fp, tn=tn, fn=fn)


class StatScores(Metric):
    """Compute stat scores for binary, multiclass and multilabel problems.

    Parameters
    ----------
        task : Literal["binary", "multiclass", "multilabel"]
            The task type. Can be either ``binary``, ``multiclass`` or
            ``multilabel``.
        pos_label : int, default=1
            The positive label to report. Only used for binary tasks.
        threshold : float, default=0.5
            The threshold to use for binarizing the predictions if logits or
            probabilities are provided. If logits are provided, a sigmoid function
            is applied prior to binarization. Used for binary and multilabel tasks.
        num_classes : int
            The number of classes for the problem. Required for multiclass tasks.
        classwise : bool, default=True
            Whether to return the stat scores for each class or sum over all
            classes. Only used for multiclass tasks.
        top_k : int, default=None
            If given, and predictions are probabilities/logits, the score will
            be computed only for the top k classes. Otherwise, ``top_k`` will be
            set to 1. Used for multiclass and multilabel tasks.
        num_labels : int
            The number of labels. Only used for multilabel tasks.
        labelwise : bool, default=False
            Whether to compute the stat scores labelwise. Only used for multilabel
            tasks.


    Examples (binary)
    -----------------
        >>> from cyclops.evaluation.metrics import StatScores
        >>> target = [0, 1, 1, 0]
        >>> preds = [0, 1, 0, 0]
        >>> metric = StatScores(task="binary", threshold=0.5, pos_label=1)
        >>> metric.update_state(target=target, preds=preds)
        >>> metric.compute()
        array([1, 0, 2, 1, 2])
        >>> metric.reset_state()
        >>> target = [[1, 1, 0, 1, 0, 0], [0, 0, 1, 1, 0, 0]]
        >>> preds = [[0.9, 0.8, 0.3, 0.4, 0.5, 0.2], [0.2, 0.3, 0.6, 0.9, 0.4, 0.8]]
        >>> for t, p in zip(target, preds):
        ...     metric(target=t, preds=p)
        >>> metric.compute()
        array([4, 2, 5, 1, 5])

    Examples (multiclass)
    ---------------------
        >>> from cyclops.evaluation.metrics import StatScores
        >>> target = [0, 1, 2, 2, 2]
        >>> preds = [0, 2, 1, 2, 0]
        >>> metric = StatScores(task="multiclass", num_classes=3, classwise=True)
        >>> metric.update(target=target, preds=preds)
        array([[1, 1, 3, 0, 1],
                [0, 1, 3, 1, 1],
                [1, 1, 1, 2, 3]])
        >>> metric.reset_state()
        >>> target = [[2, 0, 2, 2, 1], [1, 1, 0, 2, 2]]
        >>> preds = [
        ...         [
        ...             [0.1, 0.2, 0.6],
        ...             [0.6, 0.1, 0.2],
        ...             [0.2, 0.6, 0.1],
        ...             [0.2, 0.6, 0.1],
        ...             [0.6, 0.2, 0.1],
        ...         ],
        ...         [
        ...             [0.05, 0.1, 0.6],
        ...             [0.1, 0.05, 0.6],
        ...             [0.6, 0.1, 0.05],
        ...             [0.1, 0.6, 0.05],
        ...             [0.1, 0.6, 0.05],
        ...         ],
        ...     ]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(target=t, preds=p)
        >>> metric.compute()
        array([[2, 1, 7, 0, 2],
                [0, 4, 3, 3, 3],
                [1, 2, 3, 4, 5]])

    Examples (multilabel)
    ---------------------
        >>> from cyclops.evaluation.metrics import StatScores
        >>> target = [[0, 1, 1], [1, 0, 1]]
        >>> preds = [[0.1, 0.9, 0.8], [0.8, 0.2, 0.7]]
        >>> metric = StatScores(task="multilabel", num_labels=3, labelwise=True)
        >>> metric(target=target, preds=preds)
        array([[1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1],
                [2, 0, 0, 0, 2]])
        >>> metric.reset_state()
        >>> target = [[[0, 1, 1], [1, 0, 1]], [[0, 0, 1], [1, 1, 1]]]
        >>> preds = [[[0.1, 0.9, 0.8], [0.8, 0.2, 0.7]],
        ...         [[0.1, 0.9, 0.8], [0.8, 0.2, 0.7]]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(target=t, preds=p)
        >>> metric.compute()
        array([[2, 0, 2, 0, 2],
                [1, 1, 1, 1, 2],
                [4, 0, 0, 0, 4]])

    """

    def __new__(  # type: ignore # mypy expects a subclass of StatScores
        cls,
        task: Literal["binary", "multiclass", "multilabel"],
        pos_label: int = 1,
        threshold: float = 0.5,
        num_classes: Optional[int] = None,
        classwise: bool = True,
        top_k: Optional[int] = None,
        num_labels: Optional[int] = None,
        labelwise: bool = False,
    ) -> Metric:
        """Create a task-specific instance of the StatScores metric."""
        if task == "binary":
            return BinaryStatScores(threshold=threshold, pos_label=pos_label)
        if task == "multiclass":
            assert (
                isinstance(num_classes, int) and num_classes > 0
            ), "Number of classes must be a positive integer."
            return MulticlassStatScores(
                num_classes=num_classes, top_k=top_k, classwise=classwise
            )
        if task == "multilabel":
            assert (
                isinstance(num_labels, int) and num_labels > 0
            ), "Number of labels must be a positive integer."
            return MultilabelStatScores(
                num_labels=num_labels,
                threshold=threshold,
                top_k=top_k,
                labelwise=labelwise,
            )
        raise ValueError(
            f"Unsupported task: {task}, expected one of 'binary', 'multiclass' or "
            f"'multilabel'."
        )
