"""Classes for computing precision and recall scores for classification tasks."""

from cyclops.evaluate.metrics.experimental._stat_scores import (
    _AbstractBinaryStatScores,
    _AbstractMulticlassStatScores,
    _AbstractMultilabelStatScores,
)
from cyclops.evaluate.metrics.experimental.functional.precision_recall import (
    _binary_precision_recall_compute,
    _precision_recall_compute,
)
from cyclops.evaluate.metrics.experimental.utils.types import Array


class BinaryPrecision(_AbstractBinaryStatScores, registry_key="binary_precision"):
    """The proportion of positive predictions that are classified correctly.

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
    >>> from cyclops.evaluate.metrics.experimental import BinaryPrecision
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([0, 1, 0, 1])
    >>> preds = anp.asarray([0, 1, 1, 1])
    >>> metric = BinaryPrecision()
    >>> metric(target, preds)
    Array(0.6666667, dtype=float32)
    >>> metric.reset()
    >>> target = [[0, 1, 0, 1], [1, 0, 1, 0]]
    >>> preds = [[0, 1, 1, 1], [1, 0, 1, 0]]
    >>> for t, p in zip(target, preds):
    ...     metric.update(anp.asarray(t), anp.asarray(p))
    >>> metric.compute()
    Array(0.8, dtype=float32)

    """

    name: str = "Precision Score"

    def _compute_metric(self) -> Array:
        """Compute the precision score."""
        _, fp, fn, tp = self._final_state()
        return _binary_precision_recall_compute("precision", tp=tp, fp=fp, fn=fn)


class BinaryPPV(BinaryPrecision, registry_key="binary_ppv"):
    """The proportion of positive predictions that are classified correctly.

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
    >>> from cyclops.evaluate.metrics.experimental import BinaryPPV
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([0, 1, 0, 1])
    >>> preds = anp.asarray([0, 1, 1, 1])
    >>> metric = BinaryPPV()
    >>> metric(target, preds)
    Array(0.6666667, dtype=float32)
    >>> metric.reset()
    >>> target = [[0, 1, 0, 1], [1, 0, 1, 0]]
    >>> preds = [[0, 1, 1, 1], [1, 0, 1, 0]]
    >>> for t, p in zip(target, preds):
    ...     metric.update(anp.asarray(t), anp.asarray(p))
    >>> metric.compute()
    Array(0.8, dtype=float32)

    """

    name: str = "Positive Predictive Value"


class MulticlassPrecision(
    _AbstractMulticlassStatScores,
    registry_key="multiclass_precision",
):
    """The proportion of predicted classes that match the target classes.

    Parameters
    ----------
    num_classes : int
        The number of classes in the classification task.
    top_k : int, default=1
        The number of highest probability or logit score predictions to consider
        when computing the precision score. By default, only the top prediction is
        considered. This parameter is ignored if `preds` contains integer values.
    average : {'micro', 'macro', 'weighted', 'none'}, optional, default='micro'
        Specifies the type of averaging to apply to the precision scores. Should be one
        of the following:
        - `'micro'`: Compute the precision score globally by considering all predictions
            and all targets.
        - `'macro'`: Compute the precision score for each class individually and then
            take the unweighted mean of the precision scores.
        - `'weighted'`: Compute the precision score for each class individually and then
            take the mean of the precision scores weighted by the support (the number of
            true positives + the number of false negatives) for each class.
        - `'none'` or `None`: Compute the precision score for each class individually
            and return the scores as an array.
    ignore_index : int or tuple of int, optional, default=None
        Specifies a target class that is ignored when computing the precision score.
        Ignoring a target class means that the corresponding predictions do not
        contribute to the precision score.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental import MulticlassPrecision
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([0, 1, 2, 2, 2])
    >>> preds = anp.asarray([0, 0, 2, 2, 1])
    >>> metric = MulticlassPrecision(num_classes=3)
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

    name: str = "Precision Score"

    def _compute_metric(self) -> Array:
        """Compute the precision score(s)."""
        _, fp, fn, tp = self._final_state()
        return _precision_recall_compute(
            "precision",
            self.average,  # type: ignore[arg-type]
            is_multilabel=False,
            tp=tp,
            fp=fp,
            fn=fn,
        )


class MulticlassPPV(MulticlassPrecision, registry_key="multiclass_ppv"):
    """The proportion of predicted classes that match the target classes.

    Parameters
    ----------
    num_classes : int
        The number of classes in the classification task.
    top_k : int, default=1
        The number of highest probability or logit score predictions to consider
        when computing the positive predictive value. By default, only the top
        prediction is considered. This parameter is ignored if `preds` contains
        integer values.
    average : {'micro', 'macro', 'weighted', 'none'}, optional, default='micro'
        Specifies the type of averaging to apply to the positive predictive values.
        Should be one of the following:
        - `'micro'`: Compute the positive predictive value globally by considering
            all predictions and all targets.
        - `'macro'`: Compute the positive predictive value for each class individually
            and then take the unweighted mean of the positive predictive values.
        - `'weighted'`: Compute the positive predictive value for each class
            individually and then take the mean of the positive predictive values
            weighted by the support (the number of true positives + the number of
            false negatives) for each class.
        - `'none'` or `None`: Compute the positive predictive value for each class
            individually and return the scores as an array.
    ignore_index : int or tuple of int, optional, default=None
        Specifies a target class that is ignored when computing the positive predictive
        value. Ignoring a target class means that the corresponding predictions do not
        contribute to the positive predictive value.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental import MulticlassPPV
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([0, 1, 2, 2, 2])
    >>> preds = anp.asarray([0, 0, 2, 2, 1])
    >>> metric = MulticlassPPV(num_classes=3)
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

    name: str = "Positive Predictive Value"


class MultilabelPrecision(
    _AbstractMultilabelStatScores,
    registry_key="multilabel_precision",
):
    """The proportion of positive predictions that are classified correctly.

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
        Specifies the type of averaging to apply to the precision scores. Should be one
        of the following:
        - `'micro'`: Compute the precision score globally by considering all predictions
            and all targets.
        - `'macro'`: Compute the precision score for each label individually and then
            take the unweighted mean of the precision scores.
        - `'weighted'`: Compute the precision score for each label individually and then
            take the mean of the precision scores weighted by the support (the number of
            true positives + the number of false negatives) for each label.
        - `'none'` or `None`: Compute the precision score for each label individually
            and return the scores as an array.
    ignore_index : int, optional, default=None
        Specifies a value in the target array(s) that is ignored when computing
        the precision score.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental import MultilabelPrecision
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([[0, 1, 1], [1, 0, 0]])
    >>> preds = anp.asarray([[0, 1, 0], [1, 0, 1]])
    >>> metric = MultilabelPrecision(num_labels=3)
    >>> metric(target, preds)
    Array(0.6666667, dtype=float32)
    >>> metric.reset()
    >>> target = [[[0, 1, 1], [1, 0, 0]], [[1, 0, 0], [0, 1, 1]]]
    >>> preds = [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]], [[0.1, 0.8, 0.1], [0.05, 0.95, 0]]]
    >>> for t, p in zip(target, preds):
    ...     metric.update(anp.asarray(t), anp.asarray(p))
    >>> metric.compute()
    Array(0.16666667, dtype=float32)

    """

    name: str = "Precision Score"

    def _compute_metric(self) -> Array:
        """Compute the precision score(s)."""
        _, fp, fn, tp = self._final_state()
        return _precision_recall_compute(
            "precision",
            self.average,  # type: ignore[arg-type]
            is_multilabel=True,
            tp=tp,
            fp=fp,
            fn=fn,
        )


class MultilabelPPV(MultilabelPrecision, registry_key="multilabel_ppv"):
    """The proportion of positive predictions that are classified correctly.

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
        Specifies the type of averaging to apply to the positive predictive values.
        Should be one of the following:
        - `'micro'`: Compute the positive predictive value globally by considering all
            predictions and all targets.
        - `'macro'`: Compute the positive predictive value for each label individually
            and then take the unweighted mean of the positive predictive values.
        - `'weighted'`: Compute the positive predictive value for each label
            individually and then take the mean of the positive predictive values
            weighted by the support (the number of true positives + the number of
            false negatives) for each label.
        - `'none'` or `None`: Compute the positive predictive value for each label
            individually and return the scores as an array.
    ignore_index : int, optional, default=None
        Specifies a value in the target array(s) that is ignored when computing
        the positive predictive value.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental import MultilabelPPV
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([[0, 1, 1], [1, 0, 0]])
    >>> preds = anp.asarray([[0, 1, 0], [1, 0, 1]])
    >>> metric = MultilabelPPV(num_labels=3)
    >>> metric(target, preds)
    Array(0.6666667, dtype=float32)
    >>> metric.reset()
    >>> target = [[[0, 1, 1], [1, 0, 0]], [[1, 0, 0], [0, 1, 1]]]
    >>> preds = [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]], [[0.1, 0.8, 0.1], [0.05, 0.95, 0]]]
    >>> for t, p in zip(target, preds):
    ...     metric.update(anp.asarray(t), anp.asarray(p))
    >>> metric.compute()
    Array(0.16666667, dtype=float32)

    """

    name: str = "Positive Predictive Value"


class BinaryRecall(_AbstractBinaryStatScores, registry_key="binary_recall"):
    """The proportion of positive predictions that are classified correctly.

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
    >>> from cyclops.evaluate.metrics.experimental import BinaryRecall
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([0, 1, 0, 1])
    >>> preds = anp.asarray([0, 1, 1, 1])
    >>> metric = BinaryRecall()
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

    name: str = "Recall Score"

    def _compute_metric(self) -> Array:
        """Compute the recall score."""
        _, fp, fn, tp = self._final_state()
        return _binary_precision_recall_compute("recall", tp=tp, fp=fp, fn=fn)


class BinarySensitivity(BinaryRecall, registry_key="binary_sensitivity"):
    """The proportion of positive predictions that are classified correctly.

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
    >>> from cyclops.evaluate.metrics.experimental import BinarySensitivity
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([0, 1, 0, 1])
    >>> preds = anp.asarray([0, 1, 1, 1])
    >>> metric = BinarySensitivity()
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

    name: str = "Sensitivity Score"


class BinaryTPR(BinaryRecall, registry_key="binary_tpr"):
    """The proportion of positive predictions that are classified correctly.

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
    >>> from cyclops.evaluate.metrics.experimental import BinaryTPR
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([0, 1, 0, 1])
    >>> preds = anp.asarray([0, 1, 1, 1])
    >>> metric = BinaryTPR()
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

    name: str = "True Positive Rate"


class MulticlassRecall(_AbstractMulticlassStatScores, registry_key="multiclass_recall"):
    """The proportion of predicted classes that match the target classes.

    Parameters
    ----------
    num_classes : int
        The number of classes in the classification task.
    top_k : int, default=1
        The number of highest probability or logit score predictions to consider
        when computing the recall score. By default, only the top prediction is
        considered. This parameter is ignored if `preds` contains integer values.
    average : {'micro', 'macro', 'weighted', 'none'}, optional, default='micro'
        Specifies the type of averaging to apply to the recall scores. Should be one
        of the following:
        - `'micro'`: Compute the recall score globally by considering all predictions
            and all targets.
        - `'macro'`: Compute the recall score for each class individually and then
            take the unweighted mean of the recall scores.
        - `'weighted'`: Compute the recall score for each class individually and then
            take the mean of the recall scores weighted by the support (the number of
            true positives + the number of false negatives) for each class.
        - `'none'` or `None`: Compute the recall score for each class individually
            and return the scores as an array.
    ignore_index : int or tuple of int, optional, default=None
        Specifies a target class that is ignored when computing the recall score.
        Ignoring a target class means that the corresponding predictions do not
        contribute to the recall score.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental import MulticlassRecall
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([0, 1, 2, 2, 2])
    >>> preds = anp.asarray([0, 0, 2, 2, 1])
    >>> metric = MulticlassRecall(num_classes=3)
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

    name: str = "Recall Score"

    def _compute_metric(self) -> Array:
        """Compute the recall score(s)."""
        _, fp, fn, tp = self._final_state()
        return _precision_recall_compute(
            "recall",
            self.average,  # type: ignore[arg-type]
            is_multilabel=False,
            tp=tp,
            fp=fp,
            fn=fn,
        )


class MulticlassSensitivity(MulticlassRecall, registry_key="multiclass_sensitivity"):
    """The proportion of predicted classes that match the target classes.

    Parameters
    ----------
    num_classes : int
        The number of classes in the classification task.
    top_k : int, default=1
        The number of highest probability or logit score predictions to consider
        when computing the sensitivity score. By default, only the top prediction is
        considered. This parameter is ignored if `preds` contains integer values.
    average : {'micro', 'macro', 'weighted', 'none'}, optional, default='micro'
        Specifies the type of averaging to apply to the sensitivity scores. Should
        be one of the following:
        - `'micro'`: Compute the sensitivity score globally by considering all
            predictions and all targets.
        - `'macro'`: Compute the sensitivity score for each class individually
            and then take the unweighted mean of the sensitivity scores.
        - `'weighted'`: Compute the sensitivity score for each class individually
            and then take the mean of the sensitivity scores weighted by the support
            (the number of true positives + the number of false negatives) for
            each class.
        - `'none'` or `None`: Compute the sensitivity score for each class individually
            and return the scores as an array.
    ignore_index : int or tuple of int, optional, default=None
        Specifies a target class that is ignored when computing the sensitivity score.
        Ignoring a target class means that the corresponding predictions do not
        contribute to the sensitivity score.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental import MulticlassSensitivity
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([0, 1, 2, 2, 2])
    >>> preds = anp.asarray([0, 0, 2, 2, 1])
    >>> metric = MulticlassSensitivity(num_classes=3)
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

    name: str = "Sensitivity Score"


class MulticlassTPR(MulticlassRecall, registry_key="multiclass_tpr"):
    """The proportion of predicted classes that match the target classes.

    Parameters
    ----------
    num_classes : int
        The number of classes in the classification task.
    top_k : int, default=1
        The number of highest probability or logit score predictions to consider
        when computing the true positive rate. By default, only the top prediction is
        considered. This parameter is ignored if `preds` contains integer values.
    average : {'micro', 'macro', 'weighted', 'none'}, optional, default='micro'
        Specifies the type of averaging to apply to the true positive rates. Should
        be one of the following:
        - `'micro'`: Compute the true positive rate globally by considering all
            predictions and all targets.
        - `'macro'`: Compute the true positive rate for each class individually
            and then take the unweighted mean of the true positive rates.
        - `'weighted'`: Compute the true positive rate for each class individually
            and then take the mean of the true positive rates weighted by the support
            (the number of true positives + the number of false negatives) for
            each class.
        - `'none'` or `None`: Compute the true positive rate for each class individually
            and return the scores as an array.
    ignore_index : int or tuple of int, optional, default=None
        Specifies a target class that is ignored when computing the true positive rate.
        Ignoring a target class means that the corresponding predictions do not
        contribute to the true positive rate.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental import MulticlassTPR
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([0, 1, 2, 2, 2])
    >>> preds = anp.asarray([0, 0, 2, 2, 1])
    >>> metric = MulticlassTPR(num_classes=3)
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

    name: str = "True Positive Rate"


class MultilabelRecall(_AbstractMultilabelStatScores, registry_key="multilabel_recall"):
    """The proportion of positive predictions that are classified correctly.

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
        Specifies the type of averaging to apply to the recall scores. Should be one
        of the following:
        - `'micro'`: Compute the recall score globally by considering all predictions
            and all targets.
        - `'macro'`: Compute the recall score for each label individually and then
            take the unweighted mean of the recall scores.
        - `'weighted'`: Compute the recall score for each label individually and then
            take the mean of the recall scores weighted by the support (the number of
            true positives + the number of false negatives) for each label.
        - `'none'` or `None`: Compute the recall score for each label individually
            and return the scores as an array.
    ignore_index : int, optional, default=None
        Specifies a value in the target array(s) that is ignored when computing
        the recall score.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental import MultilabelRecall
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([[0, 1, 1], [1, 0, 0]])
    >>> preds = anp.asarray([[0, 1, 0], [1, 0, 1]])
    >>> metric = MultilabelRecall(num_labels=3)
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

    name: str = "Recall Score"

    def _compute_metric(self) -> Array:
        """Compute the recall score(s)."""
        _, fp, fn, tp = self._final_state()
        return _precision_recall_compute(
            "recall",
            self.average,  # type: ignore[arg-type]
            is_multilabel=True,
            tp=tp,
            fp=fp,
            fn=fn,
        )


class MultilabelSensitivity(MultilabelRecall, registry_key="multilabel_sensitivity"):
    """The proportion of positive predictions that are classified correctly.

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
        Specifies the type of averaging to apply to the sensitivity scores. Should
        be one of the following:
        - `'micro'`: Compute the sensitivity score globally by considering all
            predictions and all targets.
        - `'macro'`: Compute the sensitivity score for each label individually
            and then take the unweighted mean of the sensitivity scores.
        - `'weighted'`: Compute the sensitivity score for each label individually
            and then take the mean of the sensitivity scores weighted by the support
            (the number of true positives + the number of false negatives) for each
            label.
        - `'none'` or `None`: Compute the sensitivity score for each label individually
            and return the scores as an array.
    ignore_index : int, optional, default=None
        Specifies a value in the target array(s) that is ignored when computing
        the sensitivity score.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental import MultilabelSensitivity
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([[0, 1, 1], [1, 0, 0]])
    >>> preds = anp.asarray([[0, 1, 0], [1, 0, 1]])
    >>> metric = MultilabelSensitivity(num_labels=3)
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

    name: str = "Sensitivity Score"


class MultilabelTPR(MultilabelRecall, registry_key="multilabel_tpr"):
    """The proportion of predicted classes that match the target classes.

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
        Specifies the type of averaging to apply to the true positive rates. Should
        be one of the following:
        - `'micro'`: Compute the true positive rate globally by considering all
            predictions and all targets.
        - `'macro'`: Compute the true positive rate for each label individually
            and then take the unweighted mean of the true positive rates.
        - `'weighted'`: Compute the true positive rate for each label individually
            and then take the mean of the true positive rates weighted by the support
            (the number of true positives + the number of false negatives) for each
            label.
        - `'none'` or `None`: Compute the true positive rate for each label individually
            and return the scores as an array.
    ignore_index : int, optional, default=None
        Specifies a value in the target array(s) that is ignored when computing
        the true positive rate.
    **kwargs : Any
        Additional keyword arguments common to all metrics.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental import MultilabelTPR
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([[0, 1, 1], [1, 0, 0]])
    >>> preds = anp.asarray([[0, 1, 0], [1, 0, 1]])
    >>> metric = MultilabelTPR(num_labels=3)
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

    name: str = "True Positive Rate"
