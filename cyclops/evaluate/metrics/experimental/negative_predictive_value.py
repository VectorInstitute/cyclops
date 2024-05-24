"""Classes for computing the negative predictive value for classification tasks."""

from cyclops.evaluate.metrics.experimental._stat_scores import (
    _AbstractBinaryStatScores,
    _AbstractMulticlassStatScores,
    _AbstractMultilabelStatScores,
)
from cyclops.evaluate.metrics.experimental.functional.negative_predictive_value import (
    _binary_negative_predictive_value_compute,
    _negative_predictive_value_compute,
)
from cyclops.evaluate.metrics.experimental.utils.types import Array


class BinaryNPV(_AbstractBinaryStatScores, registry_key="binary_npv"):
    """The proportion of negative predictions that are true negatives.

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
    >>> from cyclops.evaluate.metrics.experimental import BinaryNPV
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([0, 1, 0, 1])
    >>> preds = anp.asarray([0, 1, 1, 1])
    >>> metric = BinaryNPV()
    >>> metric(target, preds)
    Array(1., dtype=float32)
    >>> metric.reset()
    >>> target = [[0, 1, 0, 1], [1, 0, 1, 0]]
    >>> preds = [[0, 1, 1, 1], [1, 0, 1, 0]]
    >>> for t, p in zip(target, preds):
    ...     metric.update(anp.asarray(t), anp.asarray(p))
    >>> metric.compute()
    Array(1., dtype=float32)

    """

    name: str = "Negative Predictive Value"

    def _compute_metric(self) -> Array:
        """Compute the negative predictive value."""
        tn, _, fn, _ = self._final_state()
        return _binary_negative_predictive_value_compute(fn=fn, tn=tn)


class MulticlassNPV(
    _AbstractMulticlassStatScores,
    registry_key="multiclass_npv",
):
    """The proportion of negative predictions that are true negatives.

    Parameters
    ----------
    num_classes : int
        The number of classes in the classification task.
    top_k : int, default=1
        The number of highest probability or logit score predictions to consider
        when computing the negative predictive value. By default, only the top
        prediction is considered. This parameter is ignored if `preds` contains
        integer values.
    average : {'micro', 'macro', 'weighted', 'none'}, optional, default='micro'
        Specifies the type of averaging to apply to the negative predictive values.
        Should be one of the following:
        - `'micro'`: Compute the negative predictive value globally by considering all
            predictions and all targets.
        - `'macro'`: Compute the negative predictive value for each class individually
            and then take the unweighted mean of the negative predictive values.
        - `'weighted'`: Compute the negative predictive value for each class
            individually and then take the mean of the negative predictive values
            weighted by the support (the number of true positives + the number of
            false negatives) for each class.
        - `'none'` or `None`: Compute the negative predictive value for each class
            individually and return the scores as an array.
    ignore_index : int or tuple of int, optional, default=None
        Specifies a target class that is ignored when computing the negative
        predictive value. Ignoring a target class means that the corresponding
        predictions do not contribute to the negative predictive value.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental import MulticlassNPV
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([0, 1, 2, 2, 2])
    >>> preds = anp.asarray([0, 0, 2, 2, 1])
    >>> metric = MulticlassNPV(num_classes=3)
    >>> metric(target, preds)
    Array(0.8, dtype=float32)
    >>> metric.reset()
    >>> target = [[0, 1, 2], [2, 1, 0]]
    >>> preds = [
    ...     [[0.05, 0.95, 0], [0.1, 0.8, 0.1], [0.2, 0.6, 0.2]],
    ...     [[0.1, 0.8, 0.1], [0.05, 0.95, 0], [0.2, 0.6, 0.2]],
    ... ]
    >>> for t, p in zip(target, preds):
    ...     metric.update(anp.asarray(t), anp.asarray(p))
    >>> metric.compute()
    Array(0.6666667, dtype=float32)

    """

    name: str = "Negative predictive value"

    def _compute_metric(self) -> Array:
        """Compute the negative predictive value(s)."""
        tn, fp, fn, tp = self._final_state()
        return _negative_predictive_value_compute(
            self.average,  # type: ignore[arg-type]
            is_multilabel=False,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
        )


class MultilabelNPV(
    _AbstractMultilabelStatScores,
    registry_key="multilabel_npv",
):
    """The proportion of negative predictions that are true negatives.

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
        Specifies the type of averaging to apply to the negative predictive values.
        Should be one of the following:
        - `'micro'`: Compute the negative predictive value globally by considering all
            predictions and all targets.
        - `'macro'`: Compute the negative predictive value for each label individually
            and then take the unweighted mean of the negative predictive values.
        - `'weighted'`: Compute the negative predictive value for each label
            individually and then take the mean of the negative predictive values
            weighted by the support (the number of true positives + the number of
            false negatives) for each label.
        - `'none'` or `None`: Compute the negative predictive value for each label
            individually and return the scores as an array.
    ignore_index : int, optional, default=None
        Specifies a value in the target array(s) that is ignored when computing
        the negative predictive value.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental import MultilabelNPV
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([[0, 1, 1], [1, 0, 0]])
    >>> preds = anp.asarray([[0, 1, 0], [1, 0, 1]])
    >>> metric = MultilabelNPV(num_labels=3)
    >>> metric(target, preds)
    Array(0.6666667, dtype=float32)
    >>> metric.reset()
    >>> target = [[[0, 1, 1], [1, 0, 0]], [[1, 0, 0], [0, 1, 1]]]
    >>> preds = [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]], [[0.1, 0.8, 0.1], [0.05, 0.95, 0]]]
    >>> for t, p in zip(target, preds):
    ...     metric.update(anp.asarray(t), anp.asarray(p))
    >>> metric.compute()
    Array(0.33333334, dtype=float32)

    """

    name: str = "Negative Predictive Value"

    def _compute_metric(self) -> Array:
        """Compute the negative predictive value(s)."""
        tn, fp, fn, tp = self._final_state()
        return _negative_predictive_value_compute(
            self.average,  # type: ignore[arg-type]
            is_multilabel=True,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
        )
