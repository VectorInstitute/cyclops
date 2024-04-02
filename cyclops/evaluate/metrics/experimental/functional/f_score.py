"""Methods for computing the F-score for classification tasks."""

from typing import Literal, Optional, Tuple, Union

import array_api_compat as apc

from cyclops.evaluate.metrics.experimental.functional._stat_scores import (
    _binary_stat_scores_format_arrays,
    _binary_stat_scores_update_state,
    _binary_stat_scores_validate_args,
    _binary_stat_scores_validate_arrays,
    _multiclass_stat_scores_format_arrays,
    _multiclass_stat_scores_update_state,
    _multiclass_stat_scores_validate_args,
    _multiclass_stat_scores_validate_arrays,
    _multilabel_stat_scores_format_arrays,
    _multilabel_stat_scores_update_state,
    _multilabel_stat_scores_validate_args,
    _multilabel_stat_scores_validate_arrays,
)
from cyclops.evaluate.metrics.experimental.utils.ops import (
    _adjust_weight_apply_average,
    safe_divide,
    squeeze_all,
)
from cyclops.evaluate.metrics.experimental.utils.types import Array


def _check_beta(beta: float) -> None:
    """Check the ``beta`` argument for F-beta score metrics."""
    if not isinstance(beta, float) and beta < 0:
        raise ValueError(
            f"Expected `beta` to be a positive float, got {beta}.",
        )


def _binary_fbeta_validate_args(
    beta: float,
    threshold: float,
    ignore_index: Optional[int],
) -> None:
    """Validate the arguments for binary F-beta score metrics."""
    _check_beta(beta)
    _binary_stat_scores_validate_args(
        threshold=threshold,
        ignore_index=ignore_index,
    )


def _binary_fbeta_compute(
    beta: float,
    *,
    fp: Array,
    fn: Array,
    tp: Array,
) -> Array:
    """Compute the F-beta score for binary classification tasks."""
    beta2 = beta**2
    xp = apc.array_namespace(fp, fn, tp)
    tp = xp.astype(tp, xp.float32, copy=False)
    fp = xp.astype(fp, xp.float32, copy=False)
    fn = xp.astype(fn, xp.float32, copy=False)
    return squeeze_all(
        safe_divide(
            numerator=(1 + beta2) * tp,
            denominator=(1 + beta2) * tp + beta2 * fn + fp,
        ),
    )


def _fbeta_compute(
    beta: float,
    tn: Array,
    fp: Array,
    fn: Array,
    tp: Array,
    average: Optional[Literal["micro", "macro", "weighted", "none"]],
    is_multilabel: bool = False,
) -> Array:
    """Compute the F-beta score for multiclass and multilabel classification tasks."""
    xp = apc.array_namespace(tn, fp, fn, tp)
    beta2 = beta**2

    if average == "micro":
        tp = xp.sum(tp, axis=0, dtype=xp.float32)
        fn = xp.sum(fn, axis=0, dtype=xp.float32)
        fp = xp.sum(fp, axis=0, dtype=xp.float32)
        return safe_divide(
            numerator=(1 + beta2) * tp,
            denominator=(1 + beta2) * tp + beta2 * fn + fp,
        )

    tp = xp.astype(tp, xp.float32, copy=False)
    fp = xp.astype(fp, xp.float32, copy=False)
    fn = xp.astype(fn, xp.float32, copy=False)
    score = safe_divide(
        numerator=(1 + beta2) * tp,
        denominator=(1 + beta2) * tp + beta2 * fn + fp,
    )

    return _adjust_weight_apply_average(
        score,
        average=average,
        is_multilabel=is_multilabel,
        tp=tp,
        fp=fp,
        fn=fn,
        xp=xp,
    )


def binary_fbeta_score(
    target: Array,
    preds: Array,
    beta: float,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
) -> Array:
    """Compute the weighted harmonic mean of precision/recall for binary tasks.

    Parameters
    ----------
    target : Array
        An array object that is compatible with the Python array API standard
        and contains the ground truth labels. The expected shape of the array
        is `(N, ...)`, where `N` is the number of samples.
    preds : Array
        An array object that is compatible with the Python array API standard and
        contains the predictions of a binary classifier. The expected shape of the
        array is `(N, ...)` where `N` is the number of samples. If `preds` contains
        floating point values that are not in the range `[0, 1]`, a sigmoid function
        will be applied to each value before thresholding.
    beta : float
        The weight to trade off the importance of precision and recall. A value
        of `beta < 1` favors precision, while a value of `beta > 1` favors recall.
    ignore_index : int, optional, default=None
        Specifies a target class that is ignored when computing the F-beta score.
        Ignoring a target class means that the corresponding predictions do not
        contribute to the F-beta score.

    Returns
    -------
    Array
        An array API compatible object containing the F-beta score.

    Raises
    ------
    TypeError
        If the arrays `target` and `preds` are not compatible with the Python
        array API standard.
    ValueError
        If `target` or `preds` are empty.
    ValueError
        If `target` or `preds` are not numeric arrays.
    ValueError
        If `target` and `preds` have different shapes.
    RuntimeError
        If `target` contains values that are not in {0, 1}.
    RuntimeError
        If `preds` contains integer values that are not in {0, 1}.
    ValueError
        If `threshold` is not a float in the range [0, 1].
    ValueError
        If `ignore_index` is not `None` or an integer.


    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental.functional import binary_fbeta_score
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([1, 1, 0, 1, 0, 1])
    >>> preds = anp.asarray([1, 0, 1, 1, 0, 1])
    >>> beta = 0.5
    >>> binary_fbeta_score(target, preds, beta)
    Array(0.75, dtype=float32)
    >>> binary_fbeta_score(target, preds, beta, ignore_index=1)
    Array(0., dtype=float32)
    >>> target = anp.asarray([1, 1, 0, 1, 0, 1])
    >>> preds = anp.asarray([0.61, 0.22, 0.84, 0.73, 0.33, 0.92])
    >>> beta = 2.0
    >>> binary_fbeta_score(target, preds, beta)
    Array(0.75, dtype=float32)
    >>> binary_fbeta_score(target, preds, beta, threshold=0.8)
    Array(0.2777778, dtype=float32)

    """
    _binary_fbeta_validate_args(beta, threshold=threshold, ignore_index=ignore_index)
    xp = _binary_stat_scores_validate_arrays(
        target,
        preds,
        ignore_index=ignore_index,
    )
    target, preds = _binary_stat_scores_format_arrays(
        target,
        preds,
        threshold=threshold,
        ignore_index=ignore_index,
        xp=xp,
    )
    tn, fp, fn, tp = _binary_stat_scores_update_state(target, preds, xp=xp)

    return _binary_fbeta_compute(beta, fp=fp, fn=fn, tp=tp)


def _multiclass_fbeta_validate_args(
    beta: float,
    num_classes: int,
    top_k: int,
    average: Optional[Literal["micro", "macro", "weighted", "none"]],
    ignore_index: Optional[Union[int, Tuple[int]]],
) -> None:
    """Validate the arguments for multiclass F-beta score metrics."""
    _check_beta(beta)
    _multiclass_stat_scores_validate_args(
        num_classes,
        top_k=top_k,
        average=average,
        ignore_index=ignore_index,
    )


def multiclass_fbeta_score(
    target: Array,
    preds: Array,
    beta: float,
    num_classes: int,
    top_k: int = 1,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
    ignore_index: Optional[Union[int, Tuple[int]]] = None,
) -> Array:
    """Compute the weighted harmonic mean of precision/recall for multiclass tasks.

    Parameters
    ----------
    target : Array
        An array object that is compatible with the Python array API standard
        and contains the ground truth labels. The expected shape of the array
        is `(N, ...)`, where `N` is the number of samples.
    preds : Array
        An array object that is compatible with the Python array API standard and
        contains the predictions of a classifier. If `preds` contains integer values
        the expected shape of the array is `(N, ...)`, where `N` is the number of
        samples. If `preds` contains floating point values the expected shape of the
        array is `(N, C, ...)` where `N` is the number of samples and `C` is the
        number of classes.
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


    Returns
    -------
    Array
        An array API compatible object containing the F-beta score(s).

    Raises
    ------
    TypeError
        If the arrays `target` and `preds` are not compatible with the Python
        array API standard.
    ValueError
        If `target` or `preds` are empty.
    ValueError
        If `target` or `preds` are not numeric arrays.
    ValueError
        If `preds` has one more dimension than `target` but `preds` does not
        contain floating point values.
    ValueError
        If `preds` has one more dimension than `target` and the second dimension
        (first dimension, if `preds` is a scalar) of `preds` is not equal to
        `num_classes`. In the multidimensional case (i.e., `preds` has more than
        two dimensions), the rest of the dimensions must be the same for `target`
        and `preds`.
    ValueError
        If `preds` and `target` have the same number of dimensions but not the
        same shape.
    RuntimeError
        If `target` or `preds` contain values that are not in
        {0, 1, ..., num_classes-1} or `target` contains more values than specified
        in `ignore_index`.
    ValueError
        If `num_classes` is not a positive integer greater than two.
    ValueError
        If `top_k` is not a positive integer.
    ValueError
        If `top_k` is greater than the number of classes.
    ValueError
        If `average` is not one of {`'micro'`, `'macro'`, `'weighted'`, `'none'`,
        `None`}.
    ValueError
        If `ignore_index` is not `None`, an integer, or a tuple of integers.


    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental.functional import (
    ...     multiclass_fbeta_score,
    ... )
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([2, 1, 0, 0])
    >>> preds = anp.asarray([2, 1, 0, 1])
    >>> beta = 0.5
    >>> multiclass_fbeta_score(target, preds, beta, num_classes=3)
    Array(0.75, dtype=float32)
    >>> target = anp.asarray([2, 1, 0, 0])
    >>> preds = anp.asarray(
    ...     [[0.1, 0.1, 0.8], [0.2, 0.7, 0.1], [0.9, 0.1, 0.0], [0.4, 0.6, 0.0]],
    ... )
    >>> beta = 2.0
    >>> multiclass_fbeta_score(target, preds, beta, num_classes=3)
    Array(0.75, dtype=float32)
    >>> multiclass_fbeta_score(target, preds, beta, num_classes=3, top_k=2)
    Array(0.8333333, dtype=float32)
    >>> multiclass_fbeta_score(target, preds, beta, num_classes=3, average=None)
    Array([0.5555556, 0.8333333, 1.       ], dtype=float32)
    >>> multiclass_fbeta_score(target, preds, beta, num_classes=3, average="macro")
    Array(0.7962963, dtype=float32)
    >>> multiclass_fbeta_score(target, preds, beta, num_classes=3, average="weighted")
    Array(0.7361111, dtype=float32)
    >>> multiclass_fbeta_score(
    ...     target,
    ...     preds,
    ...     beta,
    ...     num_classes=3,
    ...     average=None,
    ...     ignore_index=0,
    ... )
    Array([0., 1., 1.], dtype=float32)
    >>> multiclass_fbeta_score(
    ...     target,
    ...     preds,
    ...     beta,
    ...     num_classes=3,
    ...     average=None,
    ...     ignore_index=(1, 2),
    ... )
    Array([0.5555556, 0.       , 0.       ], dtype=float32)

    """
    _multiclass_fbeta_validate_args(
        beta,
        num_classes,
        top_k=top_k,
        average=average,
        ignore_index=ignore_index,
    )
    xp = _multiclass_stat_scores_validate_arrays(
        target,
        preds,
        num_classes,
        top_k=top_k,
        ignore_index=ignore_index,
    )

    target, preds = _multiclass_stat_scores_format_arrays(
        target,
        preds,
        top_k=top_k,
        xp=xp,
    )
    tn, fp, fn, tp = _multiclass_stat_scores_update_state(
        target,
        preds,
        num_classes,
        top_k=top_k,
        average=average,
        ignore_index=ignore_index,
        xp=xp,
    )
    return _fbeta_compute(
        beta,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        average=average,
        is_multilabel=False,
    )


def _multilabel_fbeta_validate_args(
    beta: float,
    num_labels: int,
    threshold: float,
    top_k: int,
    average: Optional[Literal["micro", "macro", "weighted", "none"]],
    ignore_index: Optional[int],
) -> None:
    """Validate the arguments for multilabel F-beta score metrics."""
    _check_beta(beta)
    _multilabel_stat_scores_validate_args(
        num_labels,
        threshold=threshold,
        top_k=top_k,
        average=average,
        ignore_index=ignore_index,
    )


def multilabel_fbeta_score(
    target: Array,
    preds: Array,
    beta: float,
    num_labels: int,
    threshold: float = 0.5,
    top_k: int = 1,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    ignore_index: Optional[int] = None,
) -> Array:
    """Compute the weighted harmonic mean of precision/recall for multilabel tasks.

    Parameters
    ----------
    target : Array
        An array object that is compatible with the Python array API standard
        and contains the ground truth labels. The expected shape of the array
        is `(N, L, ...)`, where `N` is the number of samples and `L` is the
        number of labels.
    preds : Array
        An array object that is compatible with the Python array API standard and
        contains the predictions of a classifier. The expected shape of the array
        is `(N, L, ...)`, where `N` is the number of samples and `L` is the
        number of labels. If `preds` contains floating point values that are not
        in the range `[0, 1]`, a sigmoid function will be applied to each value
        before thresholding.
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
        Specifies value in `target` that is ignored when computing the F-beta score.

    Returns
    -------
    Array
        An array API compatible object containing the F-beta score(s).

    Raises
    ------
    TypeError
        If the arrays `target` and `preds` are not compatible with the Python
        array API standard.
    ValueError
        If `target` or `preds` are empty.
    ValueError
        If `target` or `preds` are not numeric arrays.
    ValueError
        If `target` and `preds` have different shapes.
    ValueError
        If the second dimension of `target` and `preds` is not equal to `num_labels`.
    RuntimeError
        If `target` contains values that are not in {0, 1} or not in `ignore_index`.
    RuntimeError
        If `preds` contains integer values that are not in {0, 1}.
    ValueError
        If `num_labels` is not a positive integer greater than two.
    ValueError
        If `threshold` is not a float in the range [0, 1].
    ValueError
        If `top_k` is not a positive integer.
    ValueError
        If `top_k` is greater than the number of labels.
    ValueError
        If `average` is not one of {`'micro'`, `'macro'`, `'weighted'`, `'none'`,
        `None`}.
    ValueError
        If `ignore_index` is not `None` or an integer.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental.functional import (
    ...     multilabel_fbeta_score,
    ... )
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([[0, 1, 0], [1, 0, 1]])
    >>> preds = anp.asarray([[0, 0, 1], [1, 0, 1]])
    >>> beta = 0.5
    >>> multilabel_fbeta_score(target, preds, beta, num_labels=3)
    Array(0.51851857, dtype=float32)
    >>> target = anp.asarray([[1, 0, 1, 0], [1, 1, 0, 1]])
    >>> preds = anp.asarray([[0.11, 0.58, 0.22, 0.84], [0.73, 0.47, 0.33, 0.92]])
    >>> beta = 2.0
    >>> multilabel_fbeta_score(target, preds, beta, num_labels=4)
    Array(0.3472222, dtype=float32)
    >>> multilabel_fbeta_score(target, preds, beta, num_labels=4, threshold=0.7)
    Array(0.3472222, dtype=float32)
    >>> multilabel_fbeta_score(target, preds, beta, num_labels=4, average=None)
    Array([0.5555556, 0.       , 0.       , 0.8333333], dtype=float32)
    >>> multilabel_fbeta_score(target, preds, beta, num_labels=4, average="micro")
    Array(0.41666666, dtype=float32)
    >>> multilabel_fbeta_score(target, preds, beta, num_labels=4, average="weighted")
    Array(0.3888889, dtype=float32)
    >>> multilabel_fbeta_score(
    ...     target, preds, beta, num_labels=4, average=None, ignore_index=0
    ... )
    Array([0.5555556, 0.       , 0.       , 1.       ], dtype=float32)

    """
    _multilabel_fbeta_validate_args(
        beta,
        num_labels,
        threshold=threshold,
        top_k=top_k,
        average=average,
        ignore_index=ignore_index,
    )
    xp = _multilabel_stat_scores_validate_arrays(
        target,
        preds,
        num_labels,
        ignore_index=ignore_index,
    )
    target, preds = _multilabel_stat_scores_format_arrays(
        target,
        preds,
        top_k=top_k,
        threshold=threshold,
        ignore_index=ignore_index,
        xp=xp,
    )
    tn, fp, fn, tp = _multilabel_stat_scores_update_state(target, preds, xp=xp)
    return _fbeta_compute(
        beta,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        average=average,
        is_multilabel=True,
    )


def binary_f1_score(
    target: Array,
    preds: Array,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
) -> Array:
    """Compute the harmonic mean of precision and recall for binary classifiers.

    Parameters
    ----------
    target : Array
        An array object that is compatible with the Python array API standard
        and contains the ground truth labels. The expected shape of the array
        is `(N, ...)`, where `N` is the number of samples.
    preds : Array
        An array object that is compatible with the Python array API standard and
        contains the predictions of a binary classifier. The expected shape of the
        array is `(N, ...)` where `N` is the number of samples. If `preds` contains
        floating point values that are not in the range `[0, 1]`, a sigmoid function
        will be applied to each value before thresholding.
    ignore_index : int, optional, default=None
        Specifies a target class that is ignored when computing the F1 score.
        Ignoring a target class means that the corresponding predictions do not
        contribute to the F1 score.

    Returns
    -------
    Array
        An array API compatible object containing the F1 score.

    Raises
    ------
    TypeError
        If the arrays `target` and `preds` are not compatible with the Python
        array API standard.
    ValueError
        If `target` or `preds` are empty.
    ValueError
        If `target` or `preds` are not numeric arrays.
    ValueError
        If `target` and `preds` have different shapes.
    RuntimeError
        If `target` contains values that are not in {0, 1}.
    RuntimeError
        If `preds` contains integer values that are not in {0, 1}.
    ValueError
        If `threshold` is not a float in the range [0, 1].
    ValueError
        If `ignore_index` is not `None` or an integer.


    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental.functional import binary_f1_score
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([1, 1, 0, 1, 0, 1])
    >>> preds = anp.asarray([1, 0, 1, 1, 0, 1])
    >>> binary_f1_score(target, preds)
    Array(0.75, dtype=float32)
    >>> binary_f1_score(target, preds, ignore_index=1)
    Array(0., dtype=float32)
    >>> target = anp.asarray([1, 1, 0, 1, 0, 1])
    >>> preds = anp.asarray([0.61, 0.22, 0.84, 0.73, 0.33, 0.92])
    >>> binary_f1_score(target, preds)
    Array(0.75, dtype=float32)
    >>> binary_f1_score(target, preds, threshold=0.8)
    Array(0.33333334, dtype=float32)
    """
    return binary_fbeta_score(
        target,
        preds,
        beta=1.0,
        threshold=threshold,
        ignore_index=ignore_index,
    )


def multiclass_f1_score(
    target: Array,
    preds: Array,
    num_classes: int,
    top_k: int = 1,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
    ignore_index: Optional[Union[int, Tuple[int]]] = None,
) -> Array:
    """Compute the harmonic mean of precision and recall for multiclass tasks.

    Parameters
    ----------
    target : Array
        An array object that is compatible with the Python array API standard
        and contains the ground truth labels. The expected shape of the array
        is `(N, ...)`, where `N` is the number of samples.
    preds : Array
        An array object that is compatible with the Python array API standard and
        contains the predictions of a classifier. If `preds` contains integer values
        the expected shape of the array is `(N, ...)`, where `N` is the number of
        samples. If `preds` contains floating point values the expected shape of the
        array is `(N, C, ...)` where `N` is the number of samples and `C` is the
        number of classes.
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


    Returns
    -------
    Array
        An array API compatible object containing the F1 score(s).

    Raises
    ------
    TypeError
        If the arrays `target` and `preds` are not compatible with the Python
        array API standard.
    ValueError
        If `target` or `preds` are empty.
    ValueError
        If `target` or `preds` are not numeric arrays.
    ValueError
        If `preds` has one more dimension than `target` but `preds` does not
        contain floating point values.
    ValueError
        If `preds` has one more dimension than `target` and the second dimension
        (first dimension, if `preds` is a scalar) of `preds` is not equal to
        `num_classes`. In the multidimensional case (i.e., `preds` has more than
        two dimensions), the rest of the dimensions must be the same for `target`
        and `preds`.
    ValueError
        If `preds` and `target` have the same number of dimensions but not the
        same shape.
    RuntimeError
        If `target` or `preds` contain values that are not in
        {0, 1, ..., num_classes-1} or `target` contains more values than specified
        in `ignore_index`.
    ValueError
        If `num_classes` is not a positive integer greater than two.
    ValueError
        If `top_k` is not a positive integer.
    ValueError
        If `top_k` is greater than the number of classes.
    ValueError
        If `average` is not one of {`'micro'`, `'macro'`, `'weighted'`, `'none'`,
        `None`}.
    ValueError
        If `ignore_index` is not `None`, an integer, or a tuple of integers.


    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental.functional import multiclass_f1_score
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([2, 1, 0, 0])
    >>> preds = anp.asarray([2, 1, 0, 1])
    >>> multiclass_f1_score(target, preds, num_classes=3)
    Array(0.75, dtype=float32)
    >>> target = anp.asarray([2, 1, 0, 0])
    >>> preds = anp.asarray(
    ...     [[0.1, 0.1, 0.8], [0.2, 0.7, 0.1], [0.9, 0.1, 0.0], [0.4, 0.6, 0.0]],
    ... )
    >>> multiclass_f1_score(target, preds, num_classes=3)
    Array(0.75, dtype=float32)
    >>> multiclass_f1_score(target, preds, num_classes=3, top_k=2)
    Array(0.6666667, dtype=float32)
    >>> multiclass_f1_score(target, preds, num_classes=3, average=None)
    Array([0.6666667, 0.6666667, 1.       ], dtype=float32)
    >>> multiclass_f1_score(target, preds, num_classes=3, average="macro")
    Array(0.7777778, dtype=float32)
    >>> multiclass_f1_score(target, preds, num_classes=3, average="weighted")
    Array(0.75, dtype=float32)
    >>> multiclass_f1_score(target, preds, num_classes=3, average=None, ignore_index=0)
    Array([0., 1., 1.], dtype=float32)
    >>> multiclass_f1_score(
    ...     target,
    ...     preds,
    ...     num_classes=3,
    ...     average=None,
    ...     ignore_index=(1, 2),
    ... )
    Array([0.6666667, 0.       , 0.       ], dtype=float32)
    """
    return multiclass_fbeta_score(
        target,
        preds,
        beta=1.0,
        num_classes=num_classes,
        top_k=top_k,
        average=average,
        ignore_index=ignore_index,
    )


def multilabel_f1_score(
    target: Array,
    preds: Array,
    num_labels: int,
    threshold: float = 0.5,
    top_k: int = 1,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    ignore_index: Optional[int] = None,
) -> Array:
    """Compute the harmonic mean of precision and recall for multilabel tasks.

    Parameters
    ----------
    target : Array
        An array object that is compatible with the Python array API standard
        and contains the ground truth labels. The expected shape of the array
        is `(N, L, ...)`, where `N` is the number of samples and `L` is the
        number of labels.
    preds : Array
        An array object that is compatible with the Python array API standard and
        contains the predictions of a classifier. The expected shape of the array
        is `(N, L, ...)`, where `N` is the number of samples and `L` is the
        number of labels. If `preds` contains floating point values that are not
        in the range `[0, 1]`, a sigmoid function will be applied to each value
        before thresholding.
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
        Specifies value in `target` that is ignored when computing the F1 score.

    Returns
    -------
    Array
        An array API compatible object containing the F1 score(s).

    Raises
    ------
    TypeError
        If the arrays `target` and `preds` are not compatible with the Python
        array API standard.
    ValueError
        If `target` or `preds` are empty.
    ValueError
        If `target` or `preds` are not numeric arrays.
    ValueError
        If `target` and `preds` have different shapes.
    ValueError
        If the second dimension of `target` and `preds` is not equal to `num_labels`.
    RuntimeError
        If `target` contains values that are not in {0, 1} or not in `ignore_index`.
    RuntimeError
        If `preds` contains integer values that are not in {0, 1}.
    ValueError
        If `num_labels` is not a positive integer greater than two.
    ValueError
        If `threshold` is not a float in the range [0, 1].
    ValueError
        If `top_k` is not a positive integer.
    ValueError
        If `top_k` is greater than the number of labels.
    ValueError
        If `average` is not one of {`'micro'`, `'macro'`, `'weighted'`, `'none'`,
        `None`}.
    ValueError
        If `ignore_index` is not `None` or an integer.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental.functional import multilabel_f1_score
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([[0, 1, 0], [1, 0, 1]])
    >>> preds = anp.asarray([[0, 0, 1], [1, 0, 1]])
    >>> multilabel_f1_score(target, preds, num_labels=3)
    Array(0.5555556, dtype=float32)
    >>> target = anp.asarray([[1, 0, 1, 0], [1, 1, 0, 1]])
    >>> preds = anp.asarray([[0.11, 0.58, 0.22, 0.84], [0.73, 0.47, 0.33, 0.92]])
    >>> multilabel_f1_score(target, preds, num_labels=4)
    Array(0.33333334, dtype=float32)
    >>> multilabel_f1_score(target, preds, num_labels=4, threshold=0.7)
    Array(0.33333334, dtype=float32)
    >>> multilabel_f1_score(target, preds, num_labels=4, average=None)
    Array([0.6666667, 0.       , 0.       , 0.6666667], dtype=float32)
    >>> multilabel_f1_score(target, preds, num_labels=4, average="micro")
    Array(0.44444445, dtype=float32)
    >>> multilabel_f1_score(target, preds, num_labels=4, average="weighted")
    Array(0.40000004, dtype=float32)
    >>> multilabel_f1_score(target, preds, num_labels=4, average=None, ignore_index=0)
    Array([0.6666667, 0.       , 0.       , 1.       ], dtype=float32)

    """
    return multilabel_fbeta_score(
        target,
        preds,
        beta=1.0,
        num_labels=num_labels,
        threshold=threshold,
        top_k=top_k,
        average=average,
        ignore_index=ignore_index,
    )
