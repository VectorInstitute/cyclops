"""Functional interface for the Brier score metric."""

from types import ModuleType
from typing import Optional, Tuple, Union

import array_api_compat as apc

from cyclops.evaluate.metrics.experimental.functional._stat_scores import (
    _binary_stat_scores_validate_args,
    _binary_stat_scores_validate_arrays,
)
from cyclops.evaluate.metrics.experimental.utils.ops import (
    _to_one_hot,
    flatten,
    remove_ignore_index,
    sigmoid,
    squeeze_all,
    to_int,
)
from cyclops.evaluate.metrics.experimental.utils.types import Array
from cyclops.evaluate.metrics.experimental.utils.validation import (
    _basic_input_array_checks,
    is_floating_point,
)


def _binary_brier_score_validate_args(ignore_index: Optional[int] = None) -> None:
    """Validate arguments for binary Brier score computation."""
    _binary_stat_scores_validate_args(threshold=0.5, ignore_index=ignore_index)


def _binary_brier_score_validate_arrays(
    target: Array,
    preds: Array,
    ignore_index: Optional[int] = None,
) -> ModuleType:
    """Validate `target` and `preds` for binary Brier score computation."""
    return _binary_stat_scores_validate_arrays(target, preds, ignore_index=ignore_index)


def _binary_brier_score_format_arrays(
    target: Array,
    preds: Array,
    ignore_index: Optional[int],
    *,
    xp: ModuleType,
) -> Tuple[Array, Array]:
    """Format `target` and `preds` for binary Brier score computation.

    Unlike the stat-scores formatting used for other binary classification
    metrics, `preds` is kept as a continuous probability (not thresholded
    into a hard label), since the Brier score is a proper scoring rule
    computed directly on predicted probabilities.
    """
    target = flatten(target)
    preds = flatten(preds)

    if ignore_index is not None:
        target, preds = remove_ignore_index(target, preds, ignore_index=ignore_index)

    if not is_floating_point(preds):
        preds = xp.astype(preds, xp.float32)
    elif not xp.all(to_int(preds >= 0) * to_int(preds <= 1)):  # preds are logits
        preds = sigmoid(preds)

    return xp.astype(target, preds.dtype), preds


def _binary_brier_score_update(target: Array, preds: Array) -> Tuple[Array, int]:
    """Update and return variables required to compute the binary Brier score."""
    xp = apc.array_namespace(target, preds)
    diff = preds - target
    sum_squared_error = xp.sum(diff * diff, dtype=xp.float32)
    return sum_squared_error, target.shape[0]


def _binary_brier_score_compute(
    sum_squared_error: Array,
    num_obs: Union[int, Array],
) -> Array:
    """Compute the binary Brier score from the accumulated state."""
    return squeeze_all(sum_squared_error / num_obs)


def binary_brier_score(
    target: Array,
    preds: Array,
    ignore_index: Optional[int] = None,
) -> Array:
    """Compute the Brier score for binary classification tasks.

    The Brier score is the mean squared error between predicted
    probabilities and the (binary) target, and is a proper scoring rule
    for probabilistic predictions - it rewards models whose predicted
    probabilities are well-calibrated, not just well-ranked, which matters
    for clinical risk scores that are acted on directly (e.g. a 30%
    predicted mortality risk should correspond to an observed 30% event
    rate).

    Parameters
    ----------
    target : Array
        Ground truth binary labels (0 or 1).
    preds : Array
        Predicted probabilities (or logits, which are converted to
        probabilities via the sigmoid function) of the positive class.
    ignore_index : int, optional, default=None
        Values in `target` to ignore when computing the metric.

    Returns
    -------
    Array
        The Brier score, in the range [0, 1] (lower is better).

    Raises
    ------
    TypeError
        If `target` or `preds` is not an array object that is compatible
        with the Python array API standard.
    ValueError
        If `target` or `preds` is empty, not a numeric array, or if the
        shape of `target` and `preds` are not the same.
    RuntimeError
        If `target` contains values other than 0, 1 (and `ignore_index`,
        if specified).

    Examples
    --------
    >>> import numpy.array_api as anp
    >>> from cyclops.evaluate.metrics.experimental.functional import (
    ...     binary_brier_score,
    ... )
    >>> target = anp.asarray([0, 1, 1, 0])
    >>> preds = anp.asarray([0.1, 0.9, 0.8, 0.3])
    >>> binary_brier_score(target, preds)
    Array(0.0375, dtype=float32)

    """
    _binary_brier_score_validate_args(ignore_index=ignore_index)
    xp = _binary_brier_score_validate_arrays(target, preds, ignore_index=ignore_index)
    target, preds = _binary_brier_score_format_arrays(
        target,
        preds,
        ignore_index,
        xp=xp,
    )
    sum_squared_error, num_obs = _binary_brier_score_update(target, preds)
    return _binary_brier_score_compute(sum_squared_error, num_obs)


def _multiclass_brier_score_validate_args(
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> None:
    """Validate arguments for multiclass Brier score computation."""
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError(
            f"Expected argument `num_classes` to be an integer larger than 1, "
            f"but got {num_classes}",
        )
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(
            "Expected argument `ignore_index` to either be `None` or an integer, "
            f"but got {ignore_index}",
        )


def _multiclass_brier_score_validate_arrays(
    target: Array,
    preds: Array,
    num_classes: int,
) -> ModuleType:
    """Validate `target` and `preds` for multiclass Brier score computation."""
    _basic_input_array_checks(target, preds)
    xp = apc.array_namespace(target, preds)

    if not (preds.ndim == target.ndim + 1 and is_floating_point(preds)):
        raise ValueError(
            "Expected `preds` to be a floating point array with one more "
            "dimension than `target`, containing predicted probabilities for "
            f"each of the {num_classes} classes. Got `preds` with shape "
            f"{preds.shape} and `target` with shape {target.shape}.",
        )
    if preds.shape[-1] != num_classes:
        raise ValueError(
            "Expected the last dimension of `preds` to be equal to "
            f"`num_classes` ({num_classes}), but got {preds.shape[-1]}.",
        )
    return xp  # type: ignore[no-any-return]


def _multiclass_brier_score_format_arrays(
    target: Array,
    preds: Array,
    ignore_index: Optional[int],
    num_classes: int,
    *,
    xp: ModuleType,
) -> Tuple[Array, Array]:
    """Format `target` and `preds` for multiclass Brier score computation."""
    target = flatten(target)
    preds = xp.reshape(preds, (-1, num_classes))

    if ignore_index is not None:
        target, preds = remove_ignore_index(target, preds, ignore_index=ignore_index)

    target = _to_one_hot(to_int(target), num_classes=num_classes)
    return xp.astype(target, xp.float32), preds


def _multiclass_brier_score_update(target: Array, preds: Array) -> Tuple[Array, int]:
    """Update and return variables required to compute the multiclass Brier score."""
    xp = apc.array_namespace(target, preds)
    diff = preds - target
    sum_squared_error = xp.sum(xp.sum(diff * diff, axis=-1), dtype=xp.float32)
    return sum_squared_error, target.shape[0]


def multiclass_brier_score(
    target: Array,
    preds: Array,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> Array:
    """Compute the Brier score for multiclass classification tasks.

    Computed as the mean squared error between the predicted probability
    vector for each sample and the one-hot encoded target.

    Parameters
    ----------
    target : Array
        Ground truth class labels, shape `(N, ...)`.
    preds : Array
        Predicted probabilities for each class, shape `(N, C, ...)`. Rows
        are expected to sum to 1.
    num_classes : int
        Number of classes.
    ignore_index : int, optional, default=None
        Values in `target` to ignore when computing the metric.

    Returns
    -------
    Array
        The (multiclass) Brier score, in the range [0, 2] (lower is
        better).

    Raises
    ------
    TypeError
        If `target` or `preds` is not an array object that is compatible
        with the Python array API standard.
    ValueError
        If `num_classes` is not an integer larger than 1, if `preds` does
        not have one more dimension than `target`, or if the size of the
        last dimension of `preds` is not equal to `num_classes`.

    Examples
    --------
    >>> import numpy.array_api as anp
    >>> from cyclops.evaluate.metrics.experimental.functional import (
    ...     multiclass_brier_score,
    ... )
    >>> target = anp.asarray([0, 1, 2])
    >>> preds = anp.asarray(
    ...     [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]],
    ... )
    >>> multiclass_brier_score(target, preds, num_classes=3)
    Array(0.14666666, dtype=float32)

    """
    _multiclass_brier_score_validate_args(num_classes, ignore_index=ignore_index)
    xp = _multiclass_brier_score_validate_arrays(target, preds, num_classes)
    target, preds = _multiclass_brier_score_format_arrays(
        target,
        preds,
        ignore_index,
        num_classes,
        xp=xp,
    )
    sum_squared_error, num_obs = _multiclass_brier_score_update(target, preds)
    return _binary_brier_score_compute(sum_squared_error, num_obs)
