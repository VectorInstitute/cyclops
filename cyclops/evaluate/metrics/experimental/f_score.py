"""Classes for computing the F-score for classification tasks."""

from typing import Any, Literal, Optional, Tuple, Union

from cyclops.evaluate.metrics.experimental._stat_scores import (
    _AbstractBinaryStatScores,
    _AbstractMulticlassStatScores,
    _AbstractMultilabelStatScores,
)
from cyclops.evaluate.metrics.experimental.functional.f_score import (
    _binary_fbeta_compute,
    _binary_fbeta_validate_args,
    _fbeta_compute,
    _multiclass_fbeta_validate_args,
    _multilabel_fbeta_validate_args,
)
from cyclops.evaluate.metrics.experimental.utils.types import Array


class BinaryFBetaScore(_AbstractBinaryStatScores, registry_key="binary_fbeta_score"):
    """The weighted harmonic mean of precision and recall.

    Parameters
    ----------
    beta : float
        The weight to trade off the importance of precision and recall. A value
        of `beta < 1` favors precision, while a value of `beta > 1` favors recall.
    threshold : float, default=0.5
        Threshold for converting probabilities into binary values.
    ignore_index : int, optional
        Values in the target array to ignore when computing the metric.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental import BinaryFBetaScore
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([0, 1, 0, 1])
    >>> preds = anp.asarray([0, 1, 1, 1])
    >>> metric = BinaryFBetaScore(beta=0.5)
    >>> metric(target, preds)
    Array(0.71428573, dtype=float32)
    >>> metric.reset()
    >>> target = [[0, 1, 0, 1], [1, 0, 1, 0]]
    >>> preds = [[0, 1, 1, 1], [1, 0, 1, 0]]
    >>> for t, p in zip(target, preds):
    ...     metric.update(anp.asarray(t), anp.asarray(p))
    >>> metric.compute()
    Array(0.8333333, dtype=float32)

    """

    name: str = "F-beta Score"

    def __init__(
        self,
        beta: float,
        threshold: float = 0.5,
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(threshold=threshold, ignore_index=ignore_index, **kwargs)
        _binary_fbeta_validate_args(
            beta,
            threshold=threshold,
            ignore_index=ignore_index,
        )
        self.beta = beta

    def _compute_metric(self) -> Array:
        """Compute the F-beta score."""
        tn, fp, fn, tp = self._final_state()
        return _binary_fbeta_compute(self.beta, fp=fp, fn=fn, tp=tp)


class MulticlassFBetaScore(
    _AbstractMulticlassStatScores,
    registry_key="multiclass_fbeta_score",
):
    """The weighted harmonic mean of precision and recall.

    Parameters
    ----------
    beta : float
        The weight to trade off the importance of precision and recall. A value
        of `beta < 1` favors precision, while a value of `beta > 1` favors recall.
    num_classes : int
        The number of classes in the classification task.
    top_k : int, default=1
        The number of highest probability or logit score predictions to consider
        when computing the F-beta score. By default, only the top prediction is
        considered. This parameter is ignored if `preds` contains integer values.
    average : {'micro', 'macro', 'weighted', 'none'}, optional, default='micro'
        Specifies the type of averaging to apply to the F-beta scores. Should be one
        of the following:
        - `'micro'`: Compute the F-beta score globally by considering all predictions
            and all targets.
        - `'macro'`: Compute the F-beta score for each class individually and then
            take the unweighted mean of the F-beta scores.
        - `'weighted'`: Compute the F-beta score for each class individually and then
            take the mean of the F-beta scores weighted by the support (the number of
            true positives + the number of false negatives) for each class.
        - `'none'` or `None`: Compute the F-beta score for each class individually
            and return the scores as an array.
    ignore_index : int or tuple of int, optional, default=None
        Specifies a target class that is ignored when computing the F-beta score.
        Ignoring a target class means that the corresponding predictions do not
        contribute to the F-beta score.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental import MulticlassFBetaScore
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([0, 1, 2, 2, 2])
    >>> preds = anp.asarray([0, 0, 2, 2, 1])
    >>> metric = MulticlassFBetaScore(beta=0.5, num_classes=3)
    >>> metric(target, preds)
    Array(0.6, dtype=float32)
    >>> metric.reset()
    >>> target = [[0, 1, 2], [2, 1, 0]]
    >>> preds = [
    ...     [[0.05, 0.95, 0], [0.1, 0.8, 0.1], [0.2, 0.6, 0.2]],
    ...     [[0.1, 0.8, 0.1], [0.05, 0.95, 0], [0.2, 0.6, 0.2]],
    ... ]
    >>> for t, p in zip(target, preds):
    ...     metric.update(anp.asarray(t), anp.asarray(p))
    >>> metric.compute()
    Array(0.33333334, dtype=float32)

    """

    name: str = "F-beta Score"

    def __init__(
        self,
        beta: float,
        num_classes: int,
        top_k: int = 1,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
        ignore_index: Optional[Union[int, Tuple[int]]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            num_classes,
            top_k=top_k,
            average=average,
            ignore_index=ignore_index,
            **kwargs,
        )
        _multiclass_fbeta_validate_args(
            beta,
            num_classes,
            top_k=top_k,
            average=average,
            ignore_index=ignore_index,
        )
        self.beta = beta

    def _compute_metric(self) -> Array:
        """Compute the F-beta score."""
        tn, fp, fn, tp = self._final_state()
        return _fbeta_compute(
            self.beta,
            tn=tn,
            fp=fp,
            fn=fn,
            tp=tp,
            average=self.average,
            is_multilabel=False,
        )


class MultilabelFBetaScore(
    _AbstractMultilabelStatScores,
    registry_key="multilabel_fbeta_score",
):
    """The weighted harmonic mean of precision and recall.

    Parameters
    ----------
    beta : float
        The weight to trade off the importance of precision and recall. A value
        of `beta < 1` favors precision, while a value of `beta > 1` favors recall.
    num_labels : int
        The number of labels in the classification task.
    threshold : float, optional, default=0.5
        The threshold used to convert probabilities to binary values.
    top_k : int, optional, default=1
        The number of highest probability predictions to assign the value `1`
        (all other predictions are assigned the value `0`). By default, only the
        highest probability prediction is considered. This parameter is ignored
        if `preds` does not contain floating point values.
    average : {'micro', 'macro', 'weighted', 'none'}, optional, default='macro'
        Specifies the type of averaging to apply to the F-beta scores. Should be one
        of the following:
        - `'micro'`: Compute the F-beta score globally by considering all predictions
            and all targets.
        - `'macro'`: Compute the F-beta score for each label individually and then
            take the unweighted mean of the F-beta scores.
        - `'weighted'`: Compute the F-beta score for each label individually and then
            take the mean of the F-beta scores weighted by the support (the number of
            true positives + the number of false negatives) for each label.
        - `'none'` or `None`: Compute the F-beta score for each label individually
            and return the scores as an array.
    ignore_index : int, optional, default=None
        Specifies a value in the target array(s) that is ignored when computing
        the F-beta score.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental import MultilabelFBetaScore
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([[0, 1, 1], [1, 0, 0]])
    >>> preds = anp.asarray([[0, 1, 0], [1, 0, 1]])
    >>> metric = MultilabelFBetaScore(beta=0.5, num_labels=3)
    >>> metric(target, preds)
    Array(0.6666667, dtype=float32)
    >>> metric.reset()
    >>> target = [[[0, 1, 1], [1, 0, 0]], [[1, 0, 0], [0, 1, 1]]]
    >>> preds = [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]], [[0.1, 0.8, 0.1], [0.05, 0.95, 0]]]
    >>> for t, p in zip(target, preds):
    ...     metric.update(anp.asarray(t), anp.asarray(p))
    >>> metric.compute()
    Array(0.1851852, dtype=float32)

    """

    name: str = "F-beta Score"

    def __init__(
        self,
        beta: float,
        num_labels: int,
        threshold: float = 0.5,
        top_k: int = 1,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(
            num_labels,
            threshold=threshold,
            top_k=top_k,
            average=average,
            ignore_index=ignore_index,
            **kwargs,
        )
        _multilabel_fbeta_validate_args(
            beta,
            num_labels,
            threshold=threshold,
            top_k=top_k,
            average=average,
            ignore_index=ignore_index,
        )
        self.beta = beta

    def _compute_metric(self) -> Array:
        """Compute the F-beta score."""
        tn, fp, fn, tp = self._final_state()
        return _fbeta_compute(
            self.beta,
            tn=tn,
            fp=fp,
            fn=fn,
            tp=tp,
            average=self.average,
            is_multilabel=True,
        )


class BinaryF1Score(BinaryFBetaScore, registry_key="binary_f1_score"):
    """The weighted harmonic mean of precision and recall.

    Parameters
    ----------
    threshold : float, default=0.5
        Threshold for converting probabilities into binary values.
    ignore_index : int, optional
        Values in the target array to ignore when computing the metric.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental import BinaryF1Score
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([0, 1, 0, 1])
    >>> preds = anp.asarray([0, 1, 1, 1])
    >>> metric = BinaryF1Score()
    >>> metric(target, preds)
    Array(0.8, dtype=float32)
    >>> metric.reset()
    >>> target = [[0, 1, 0, 1], [1, 0, 1, 0]]
    >>> preds = [[0, 1, 1, 1], [1, 0, 1, 0]]
    >>> for t, p in zip(target, preds):
    ...     metric.update(anp.asarray(t), anp.asarray(p))
    >>> metric.compute()
    Array(0.8888889, dtype=float32)
    """

    name: str = "F1 Score"

    def __init__(
        self,
        threshold: float = 0.5,
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(
            beta=1.0,
            threshold=threshold,
            ignore_index=ignore_index,
            **kwargs,
        )


class MulticlassF1Score(MulticlassFBetaScore, registry_key="multiclass_f1_score"):
    """The harmonic mean of precision and recall.

    Parameters
    ----------
    num_classes : int
        The number of classes in the classification task.
    top_k : int, default=1
        The number of highest probability or logit score predictions to consider
        when computing the F1 score. By default, only the top prediction is
        considered. This parameter is ignored if `preds` contains integer values.
    average : {'micro', 'macro', 'weighted', 'none'}, optional, default='micro'
        Specifies the type of averaging to apply to the F1 scores. Should be one
        of the following:
        - `'micro'`: Compute the F1 score globally by considering all predictions
            and all targets.
        - `'macro'`: Compute the F1 score for each class individually and then
            take the unweighted mean of the F1 scores.
        - `'weighted'`: Compute the F1 score for each class individually and then
            take the mean of the F1 scores weighted by the support (the number of
            true positives + the number of false negatives) for each class.
        - `'none'` or `None`: Compute the F1 score for each class individually
            and return the scores as an array.
    ignore_index : int or tuple of int, optional, default=None
        Specifies a target class that is ignored when computing the F1 score.
        Ignoring a target class means that the corresponding predictions do not
        contribute to the F1 score.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental import MulticlassF1Score
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([0, 1, 2, 2, 2])
    >>> preds = anp.asarray([0, 0, 2, 2, 1])
    >>> metric = MulticlassF1Score(num_classes=3)
    >>> metric(target, preds)
    Array(0.6, dtype=float32)
    >>> metric.reset()
    >>> target = [[0, 1, 2], [2, 1, 0]]
    >>> preds = [
    ...     [[0.05, 0.95, 0], [0.1, 0.8, 0.1], [0.2, 0.6, 0.2]],
    ...     [[0.1, 0.8, 0.1], [0.05, 0.95, 0], [0.2, 0.6, 0.2]],
    ... ]
    >>> for t, p in zip(target, preds):
    ...     metric.update(anp.asarray(t), anp.asarray(p))
    >>> metric.compute()
    Array(0.33333334, dtype=float32)
    """

    name: str = "F1 Score"

    def __init__(
        self,
        num_classes: int,
        top_k: int = 1,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
        ignore_index: Optional[Union[int, Tuple[int]]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            beta=1.0,
            num_classes=num_classes,
            top_k=top_k,
            average=average,
            ignore_index=ignore_index,
            **kwargs,
        )


class MultilabelF1Score(
    MultilabelFBetaScore,
    registry_key="multilabel_f1_score",
):
    """The harmonic mean of precision and recall.

    Parameters
    ----------
    num_labels : int
        The number of labels in the classification task.
    threshold : float, optional, default=0.5
        The threshold used to convert probabilities to binary values.
    top_k : int, optional, default=1
        The number of highest probability predictions to assign the value `1`
        (all other predictions are assigned the value `0`). By default, only the
        highest probability prediction is considered. This parameter is ignored
        if `preds` does not contain floating point values.
    average : {'micro', 'macro', 'weighted', 'none'}, optional, default='macro'
        Specifies the type of averaging to apply to the F1 scores. Should be one
        of the following:
        - `'micro'`: Compute the F1 score globally by considering all predictions
            and all targets.
        - `'macro'`: Compute the F1 score for each label individually and then
            take the unweighted mean of the F1 scores.
        - `'weighted'`: Compute the F1 score for each label individually and then
            take the mean of the F1 scores weighted by the support (the number of
            true positives + the number of false negatives) for each label.
        - `'none'` or `None`: Compute the F1 score for each label individually
            and return the scores as an array.
    ignore_index : int, optional, default=None
        Specifies a value in the target array(s) that is ignored when computing
        the F1 score.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental import MultilabelF1Score
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([[0, 1, 1], [1, 0, 0]])
    >>> preds = anp.asarray([[0, 1, 0], [1, 0, 1]])
    >>> metric = MultilabelF1Score(num_labels=3)
    >>> metric(target, preds)
    Array(0.6666667, dtype=float32)
    >>> metric.reset()
    >>> target = [[[0, 1, 1], [1, 0, 0]], [[1, 0, 0], [0, 1, 1]]]
    >>> preds = [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]], [[0.1, 0.8, 0.1], [0.05, 0.95, 0]]]
    >>> for t, p in zip(target, preds):
    ...     metric.update(anp.asarray(t), anp.asarray(p))
    >>> metric.compute()
    Array(0.22222222, dtype=float32)

    """

    name: str = "F1 Score"

    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        top_k: int = 1,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(
            beta=1.0,
            num_labels=num_labels,
            threshold=threshold,
            top_k=top_k,
            average=average,
            ignore_index=ignore_index,
            **kwargs,
        )
