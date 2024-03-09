"""Classes for computing specificity scores for classification tasks."""

from cyclops.evaluate.metrics.experimental._stat_scores import (
    _AbstractBinaryStatScores,
    _AbstractMulticlassStatScores,
    _AbstractMultilabelStatScores,
)
from cyclops.evaluate.metrics.experimental.functional.specificity import (
    _binary_specificity_compute,
    _specificity_compute,
)
from cyclops.evaluate.metrics.experimental.utils.types import Array


class BinarySpecificity(_AbstractBinaryStatScores, registry_key="binary_specificity"):
    """The proportion of actual negatives that are correctly identified.

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
    >>> from cyclops.evaluate.metrics.experimental import BinarySpecificity
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([0, 1, 0, 1])
    >>> preds = anp.asarray([0, 1, 1, 1])
    >>> metric = BinarySpecificity()
    >>> metric(target, preds)
    Array(0.5, dtype=float32)
    >>> metric.reset()
    >>> target = [[0, 1, 0, 1], [1, 0, 1, 0]]
    >>> preds = [[0, 1, 1, 1], [1, 0, 1, 0]]
    >>> for t, p in zip(target, preds):
    ...     metric.update(anp.asarray(t), anp.asarray(p))
    >>> metric.compute()
    Array(0.75, dtype=float32)

    """

    name: str = "Specificity Score"

    def _compute_metric(self) -> Array:
        """Compute the specificity score."""
        tn, fp, _, _ = self._final_state()
        return _binary_specificity_compute(fp=fp, tn=tn)


class MulticlassSpecificity(
    _AbstractMulticlassStatScores,
    registry_key="multiclass_specificity",
):
    """The proportion of actual negatives that are correctly identified.

    Parameters
    ----------
    num_classes : int
        The number of classes in the classification task.
    top_k : int, default=1
        The number of highest probability or logit score predictions to consider
        when computing the specificity score. By default, only the top prediction is
        considered. This parameter is ignored if `preds` contains integer values.
    average : {'micro', 'macro', 'weighted', 'none'}, optional, default='micro'
        Specifies the type of averaging to apply to the specificity scores. Should
        be one of the following:
        - `'micro'`: Compute the specificity score globally by considering all
            predictions and all targets.
        - `'macro'`: Compute the specificity score for each class individually and
            then take the unweighted mean of the specificity scores.
        - `'weighted'`: Compute the specificity score for each class individually
            and then take the mean of the specificity scores weighted by the support
            (the number of true positives + the number of false negatives) for
            each class.
        - `'none'` or `None`: Compute the specificity score for each class individually
            and return the scores as an array.
    ignore_index : int or tuple of int, optional, default=None
        Specifies a target class that is ignored when computing the specificity score.
        Ignoring a target class means that the corresponding predictions do not
        contribute to the specificity score.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental import MulticlassSpecificity
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([0, 1, 2, 2, 2])
    >>> preds = anp.asarray([0, 0, 2, 2, 1])
    >>> metric = MulticlassSpecificity(num_classes=3)
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

    name: str = "Specificity Score"

    def _compute_metric(self) -> Array:
        """Compute the specificity score(s)."""
        tn, fp, fn, tp = self._final_state()
        return _specificity_compute(
            self.average,  # type: ignore[arg-type]
            is_multilabel=False,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
        )


class MultilabelSpecificity(
    _AbstractMultilabelStatScores,
    registry_key="multilabel_specificity",
):
    """The proportion of actual negatives that are correctly identified.

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
        Specifies the type of averaging to apply to the specificity scores. Should
        be one of the following:
        - `'micro'`: Compute the specificity score globally by considering all
            predictions and all targets.
        - `'macro'`: Compute the specificity score for each label individually and
            then take the unweighted mean of the specificity scores.
        - `'weighted'`: Compute the specificity score for each label individually
            and then take the mean of the specificity scores weighted by the support
            (the number of true positives + the number of false negatives) for each
            label.
        - `'none'` or `None`: Compute the specificity score for each label individually
            and return the scores as an array.
    ignore_index : int, optional, default=None
        Specifies a value in the target array(s) that is ignored when computing
        the specificity score.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental import MultilabelSpecificity
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([[0, 1, 1], [1, 0, 0]])
    >>> preds = anp.asarray([[0, 1, 0], [1, 0, 1]])
    >>> metric = MultilabelSpecificity(num_labels=3)
    >>> metric(target, preds)
    Array(0.6666667, dtype=float32)
    >>> metric.reset()
    >>> target = [[[0, 1, 1], [1, 0, 0]], [[1, 0, 0], [0, 1, 1]]]
    >>> preds = [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]], [[0.1, 0.8, 0.1], [0.05, 0.95, 0]]]
    >>> for t, p in zip(target, preds):
    ...     metric.update(anp.asarray(t), anp.asarray(p))
    >>> metric.compute()
     Array(0.6666667, dtype=float32)

    """

    name: str = "Specificity Score"

    def _compute_metric(self) -> Array:
        """Compute the specificity score(s)."""
        tn, fp, fn, tp = self._final_state()
        return _specificity_compute(
            self.average,  # type: ignore[arg-type]
            is_multilabel=True,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
        )


# Aliases
class BinaryTNR(BinarySpecificity, registry_key="binary_tnr"):
    """The proportion of actual negatives that are correctly identified.

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
    >>> from cyclops.evaluate.metrics.experimental import BinaryTNR
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([0, 1, 0, 1])
    >>> preds = anp.asarray([0, 1, 1, 1])
    >>> metric = BinaryTNR()
    >>> metric(target, preds)
    Array(0.5, dtype=float32)
    >>> metric.reset()
    >>> target = [[0, 1, 0, 1], [1, 0, 1, 0]]
    >>> preds = [[0, 1, 1, 1], [1, 0, 1, 0]]
    >>> for t, p in zip(target, preds):
    ...     metric.update(anp.asarray(t), anp.asarray(p))
    >>> metric.compute()
    Array(0.75, dtype=float32)

    """

    name: str = "True Negative Rate"


class MulticlassTNR(MulticlassSpecificity, registry_key="multiclass_tnr"):
    """The proportion of actual negatives that are correctly identified.

    Parameters
    ----------
    num_classes : int
        The number of classes in the classification task.
    top_k : int, default=1
        The number of highest probability or logit score predictions to consider
        when computing the true negative rate. By default, only the top prediction is
        considered. This parameter is ignored if `preds` contains integer values.
    average : {'micro', 'macro', 'weighted', 'none'}, optional, default='micro'
        Specifies the type of averaging to apply to the true negative rates. Should
        be one of the following:
        - `'micro'`: Compute the true negative rate globally by considering all
            predictions and all targets.
        - `'macro'`: Compute the true negative rate for each class individually and
            then take the unweighted mean of the true negative rates.
        - `'weighted'`: Compute the true negative rate for each class individually
            and then take the mean of the true negative rates weighted by the support
            (the number of true positives + the number of false negatives) for
            each class.
        - `'none'` or `None`: Compute the true negative rate for each class individually
            and return the scores as an array.
    ignore_index : int or tuple of int, optional, default=None
        Specifies a target class that is ignored when computing the true negative rate.
        Ignoring a target class means that the corresponding predictions do not
        contribute to the true negative rate.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental import MulticlassTNR
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([0, 1, 2, 2, 2])
    >>> preds = anp.asarray([0, 0, 2, 2, 1])
    >>> metric = MulticlassTNR(num_classes=3)
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

    name: str = "True Negative Rate"


class MultilabelTNR(MultilabelSpecificity, registry_key="multilabel_tnr"):
    """The proportion of actual negatives that are correctly identified.

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
        Specifies the type of averaging to apply to the true negative rates. Should
        be one of the following:
        - `'micro'`: Compute the true negative rate globally by considering all
            predictions and all targets.
        - `'macro'`: Compute the true negative rate for each label individually and
            then take the unweighted mean of the true negative rates.
        - `'weighted'`: Compute the true negative rate for each label individually
            and then take the mean of the true negative rates weighted by the support
            (the number of true positives + the number of false negatives) for each
            label.
        - `'none'` or `None`: Compute the true negative rate for each label individually
            and return the scores as an array.
    ignore_index : int, optional, default=None
        Specifies a value in the target array(s) that is ignored when computing
        the true negative rate.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental import MultilabelTNR
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([[0, 1, 1], [1, 0, 0]])
    >>> preds = anp.asarray([[0, 1, 0], [1, 0, 1]])
    >>> metric = MultilabelTNR(num_labels=3)
    >>> metric(target, preds)
    Array(0.6666667, dtype=float32)
    >>> metric.reset()
    >>> target = [[[0, 1, 1], [1, 0, 0]], [[1, 0, 0], [0, 1, 1]]]
    >>> preds = [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]], [[0.1, 0.8, 0.1], [0.05, 0.95, 0]]]
    >>> for t, p in zip(target, preds):
    ...     metric.update(anp.asarray(t), anp.asarray(p))
    >>> metric.compute()
     Array(0.6666667, dtype=float32)

    """

    name: str = "True Negative Rate"
