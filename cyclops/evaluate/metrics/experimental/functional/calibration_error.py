"""Functional interface for the (binary) calibration error metric."""

from types import ModuleType
from typing import Literal, Optional, Tuple

import array_api_compat as apc

from cyclops.evaluate.metrics.experimental.functional._stat_scores import (
    _binary_stat_scores_validate_args,
    _binary_stat_scores_validate_arrays,
)
from cyclops.evaluate.metrics.experimental.functional.brier_score import (
    _binary_brier_score_format_arrays,
)
from cyclops.evaluate.metrics.experimental.utils.ops import (
    bincount,
    safe_divide,
    to_int,
)
from cyclops.evaluate.metrics.experimental.utils.types import Array


_ALLOWED_NORMS = ("l1", "l2", "max")


def _binary_calibration_error_validate_args(
    n_bins: int = 15,
    norm: Literal["l1", "l2", "max"] = "l1",
    ignore_index: Optional[int] = None,
) -> None:
    """Validate arguments for binary calibration error computation."""
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError(
            f"Expected argument `n_bins` to be a positive integer, but got {n_bins}",
        )
    if norm not in _ALLOWED_NORMS:
        raise ValueError(
            f"Expected argument `norm` to be one of {_ALLOWED_NORMS}, but got {norm}",
        )
    _binary_stat_scores_validate_args(threshold=0.5, ignore_index=ignore_index)


def _binary_calibration_error_validate_arrays(
    target: Array,
    preds: Array,
    ignore_index: Optional[int] = None,
) -> ModuleType:
    """Validate `target` and `preds` for binary calibration error computation."""
    return _binary_stat_scores_validate_arrays(target, preds, ignore_index=ignore_index)


def _binary_calibration_error_update(
    target: Array,
    preds: Array,
    n_bins: int,
    *,
    xp: ModuleType,
) -> Tuple[Array, Array, Array]:
    """Compute per-bin confidence sum, correctness sum, and count."""
    bin_ids = to_int(xp.floor(preds * n_bins))
    # `preds == 1.0` falls in its own out-of-range bin; fold it into the last one
    bin_ids = xp.where(
        bin_ids >= n_bins,
        xp.asarray(n_bins - 1, dtype=bin_ids.dtype, device=apc.device(bin_ids)),
        bin_ids,
    )

    bin_confidence_sums = xp.astype(
        bincount(bin_ids, weights=preds, minlength=n_bins),
        xp.float32,
    )
    bin_correct_sums = xp.astype(
        bincount(bin_ids, weights=target, minlength=n_bins),
        xp.float32,
    )
    bin_counts = bincount(bin_ids, minlength=n_bins)
    return bin_confidence_sums, bin_correct_sums, bin_counts


def _binary_calibration_error_compute(
    bin_confidence_sums: Array,
    bin_correct_sums: Array,
    bin_counts: Array,
    norm: Literal["l1", "l2", "max"] = "l1",
) -> Array:
    """Compute the binary calibration error from the accumulated per-bin state."""
    xp = apc.array_namespace(bin_confidence_sums, bin_correct_sums, bin_counts)
    bin_counts = xp.astype(bin_counts, xp.float32)
    bin_confidence_sums = xp.astype(bin_confidence_sums, xp.float32)
    bin_correct_sums = xp.astype(bin_correct_sums, xp.float32)

    avg_confidence = safe_divide(bin_confidence_sums, bin_counts)
    avg_accuracy = safe_divide(bin_correct_sums, bin_counts)
    gaps = xp.abs(avg_confidence - avg_accuracy)

    if norm == "max":
        return xp.astype(xp.max(gaps), xp.float32)  # type: ignore[no-any-return]

    bin_weights = safe_divide(
        bin_counts,
        xp.sum(bin_counts, dtype=xp.float32),
    )
    if norm == "l2":
        return xp.astype(  # type: ignore[no-any-return]
            xp.sqrt(xp.sum((gaps**2) * bin_weights, dtype=xp.float32)),
            xp.float32,
        )
    # l1, i.e. the "expected calibration error" (ECE)
    return xp.sum(gaps * bin_weights, dtype=xp.float32)  # type: ignore[no-any-return]


def binary_calibration_error(
    target: Array,
    preds: Array,
    n_bins: int = 15,
    norm: Literal["l1", "l2", "max"] = "l1",
    ignore_index: Optional[int] = None,
) -> Array:
    """Compute the calibration error for binary classification tasks.

    Groups predicted probabilities into `n_bins` equal-width bins and
    measures, within each bin, the gap between the average predicted
    probability (confidence) and the observed event rate (accuracy). The
    `"l1"` norm (the default) gives the Expected Calibration Error (ECE),
    the most commonly reported calibration metric.

    A well-calibrated clinical risk model should have a low calibration
    error: among patients given, say, a 30% predicted risk, roughly 30%
    should actually experience the event. This matters even for models
    with good discrimination (e.g. high AUROC), since discrimination alone
    doesn't guarantee that predicted probabilities can be trusted at face
    value - which is often how clinical risk scores are actually used.

    Parameters
    ----------
    target : Array
        Ground truth binary labels (0 or 1).
    preds : Array
        Predicted probabilities (or logits, which are converted to
        probabilities via the sigmoid function) of the positive class.
    n_bins : int, optional, default=15
        Number of equal-width bins to group predicted probabilities into.
    norm : {'l1', 'l2', 'max'}, optional, default='l1'
        Norm used to aggregate the per-bin calibration gaps:

        - `'l1'`: the (sample-size-)weighted average absolute gap, i.e.
          the Expected Calibration Error (ECE).
        - `'l2'`: the (sample-size-)weighted root mean square gap.
        - `'max'`: the largest gap across bins, i.e. the Maximum
          Calibration Error (MCE).
    ignore_index : int, optional, default=None
        Values in `target` to ignore when computing the metric.

    Returns
    -------
    Array
        The calibration error, in the range [0, 1] (lower is better).

    Raises
    ------
    TypeError
        If `target` or `preds` is not an array object that is compatible
        with the Python array API standard.
    ValueError
        If `n_bins` is not a positive integer, if `norm` is not one of
        `'l1'`, `'l2'`, `'max'`, or if `target` or `preds` is empty, not a
        numeric array, or not the same shape.
    RuntimeError
        If `target` contains values other than 0, 1 (and `ignore_index`,
        if specified).

    Examples
    --------
    >>> import numpy.array_api as anp
    >>> from cyclops.evaluate.metrics.experimental.functional import (
    ...     binary_calibration_error,
    ... )
    >>> target = anp.asarray([0, 1, 1, 0])
    >>> preds = anp.asarray([0.1, 0.9, 0.8, 0.3])
    >>> binary_calibration_error(target, preds, n_bins=2)
    Array(0.17499998, dtype=float32)

    """
    _binary_calibration_error_validate_args(
        n_bins=n_bins,
        norm=norm,
        ignore_index=ignore_index,
    )
    xp = _binary_calibration_error_validate_arrays(
        target,
        preds,
        ignore_index=ignore_index,
    )
    target, preds = _binary_brier_score_format_arrays(
        target,
        preds,
        ignore_index,
        xp=xp,
    )
    bin_confidence_sums, bin_correct_sums, bin_counts = (
        _binary_calibration_error_update(
            target,
            preds,
            n_bins,
            xp=xp,
        )
    )
    return _binary_calibration_error_compute(
        bin_confidence_sums,
        bin_correct_sums,
        bin_counts,
        norm=norm,
    )
