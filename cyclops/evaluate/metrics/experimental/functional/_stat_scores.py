"""Methods to compute the number of true/false positives, and true/false negatives."""

# mypy: disable-error-code="no-any-return"
from types import ModuleType
from typing import Literal, Optional, Tuple, Union

import array_api_compat as apc

from cyclops.evaluate.metrics.experimental.functional.confusion_matrix import (
    _multiclass_confusion_matrix_validate_arrays,
    _multilabel_confusion_matrix_validate_arrays,
)
from cyclops.evaluate.metrics.experimental.utils.ops import (
    _select_topk,
    _to_one_hot,
    bincount,
    clone,
    flatten,
    moveaxis,
    remove_ignore_index,
    sigmoid,
    squeeze_all,
    to_int,
)
from cyclops.evaluate.metrics.experimental.utils.types import Array
from cyclops.evaluate.metrics.experimental.utils.validation import (
    _basic_input_array_checks,
    _check_same_shape,
    is_floating_point,
)


def _binary_stat_scores_validate_args(
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
) -> None:
    """Validate arguments."""
    if not (isinstance(threshold, float) and (0.0 <= threshold <= 1.0)):
        raise ValueError(
            "Expected argument `threshold` to be a float in the [0,1] range, "
            f"but got {threshold}.",
        )
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(
            "Expected argument `ignore_index` to either be `None` or an integer, "
            f"but got {ignore_index}",
        )


def _binary_stat_scores_validate_arrays(
    target: Array,
    preds: Array,
    ignore_index: Optional[int] = None,
) -> ModuleType:
    """Validate input arrays."""
    _basic_input_array_checks(target, preds)
    _check_same_shape(target, preds)

    xp = apc.array_namespace(target, preds)

    unique_values = xp.unique_values(target)
    if ignore_index is None:
        check = xp.any((unique_values != 0) & (unique_values != 1))
    else:
        check = xp.any(
            (unique_values != 0)
            & (unique_values != 1)
            & (unique_values != ignore_index),
        )
    if check:
        raise RuntimeError(
            "Expected only the following values "
            f"{[0, 1] if ignore_index is None else [ignore_index]} in `target`. "
            f"But found the following values: {unique_values}",
        )

    if not is_floating_point(preds):
        unique_values = xp.unique_values(preds)
        if xp.any((unique_values != 0) & (unique_values != 1)):
            raise RuntimeError(
                "Expected only the following values "
                f"{[0, 1] if ignore_index is None else [ignore_index]} in `preds`. "
                f"But found the following values: {unique_values}",
            )

    return xp


def _binary_stat_scores_format_arrays(
    target: Array,
    preds: Array,
    threshold: float,
    ignore_index: Optional[int],
    *,
    xp: ModuleType,
) -> Tuple[Array, Array]:
    """Format input arrays.

    Notes
    -----
    - Both `target` and `preds` are flattened.
    - If `ignore_index` is not `None`, the corresponding values in `target` and `preds`
        are removed.
    - If `preds` are logits, they are converted to probabilities using the sigmoid
        function.
    - If `preds` are probabilities, they are converted to binary values using the
        `threshold` value.
    """
    preds = flatten(preds)
    target = flatten(target)

    if ignore_index is not None:
        target, preds = remove_ignore_index(target, preds, ignore_index=ignore_index)

    if is_floating_point(preds):
        # NOTE: in the 2021.12 version of the the array API standard the `__mul__`
        # operator is only defined for numeric arrays (including float and int scalars)
        # so we convert the boolean array to an integer array first.
        if not xp.all(to_int((preds >= 0)) * to_int((preds <= 1))):  # preds are logits
            preds = sigmoid(preds)
        preds = to_int(preds > threshold)

    return target, preds


def _binary_stat_scores_update_state(
    target: Array,
    preds: Array,
    *,
    xp: ModuleType,
) -> Tuple[Array, Array, Array, Array]:
    """Compute the components of a confusion matrix for binary tasks."""
    # NOTE: in the 2021.12 version of the array API standard, the `sum` method
    # only supports numeric types, so we have to cast the boolean arrays to integers.
    # Also, the `squeeze` method in the array API standard does not support `axis=None`
    # so we define a custom method `squeeze_all` to squeeze all singleton dimensions.
    tp = squeeze_all(xp.sum(to_int((target == preds) & (target == 1))))
    fn = squeeze_all(xp.sum(to_int((target != preds) & (target == 1))))
    fp = squeeze_all(xp.sum(to_int((target != preds) & (target == 0))))
    tn = squeeze_all(xp.sum(to_int((target == preds) & (target == 0))))

    return tn, fp, fn, tp


def _multiclass_stat_scores_validate_args(
    num_classes: int,
    top_k: int,
    average: Optional[Literal["micro", "macro", "weighted", "none"]],
    ignore_index: Optional[Union[int, Tuple[int]]] = None,
) -> None:
    """Validate arguments."""
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError(
            "Expected argument `num_classes` to be an integer larger than 1, "
            f"but got {num_classes}.",
        )
    if not isinstance(top_k, int):
        raise TypeError(
            f"Expected `top_k` to be an integer, but {type(top_k)} was provided.",
        )
    if top_k < 1:
        raise ValueError(
            "Expected argument `top_k` to be an integer larger than or equal to 1, "
            "but got {top_k}",
        )
    if top_k > num_classes:
        raise ValueError(
            "Expected argument `top_k` to be smaller or equal to `num_classes`, "
            f"but got {top_k} and {num_classes}",
        )

    allowed_average = ("micro", "macro", "weighted", "none", None)
    if average not in allowed_average:
        raise ValueError(
            f"Expected argument `average` to be one of {allowed_average}, "
            f"but got {average}",
        )

    if ignore_index is not None and not (
        isinstance(ignore_index, int)
        or (
            isinstance(ignore_index, tuple)
            and all(isinstance(i, int) for i in ignore_index)
        )
    ):
        raise ValueError(
            "Expected argument `ignore_index` to either be `None`, an integer, "
            f"or a tuple of integers but got {ignore_index}",
        )


def _multiclass_stat_scores_validate_arrays(
    target: Array,
    preds: Array,
    num_classes: int,
    top_k: int,
    ignore_index: Optional[Union[int, Tuple[int]]] = None,
) -> ModuleType:
    """Validate multiclass input arrays."""
    xp = _multiclass_confusion_matrix_validate_arrays(
        target,
        preds,
        num_classes,
        ignore_index=ignore_index,
    )

    if top_k > 1 and (not is_floating_point(preds) or preds.ndim == target.ndim):
        raise ValueError(
            "Expected argument `preds` to contain floating point values of logits "
            f"or probability scores when `top_k` is larger than 1, but got {preds}",
        )

    return xp


def _multiclass_stat_scores_format_arrays(
    target: Array,
    preds: Array,
    top_k: int = 1,
    *,
    xp: ModuleType,
) -> Tuple[Array, Array]:
    """Format multiclass input arrays.

    Notes
    -----
    - If `preds` are probability scores and `top_k=1`, they are converted to
        class labels by selecting the class with the highest probability.
    - Any extra dimensions in `preds` beyond the first two are flattened.
    - Any extra dimensions in `target` beyond the first are flattened.
    """
    if preds.ndim == target.ndim + 1 and top_k == 1:
        preds = xp.argmax(preds, axis=1)

    preds = (
        xp.reshape(preds, shape=(*preds.shape[:2], -1))
        if top_k != 1
        else xp.reshape(preds, shape=(preds.shape[0], -1))
    )
    target = xp.reshape(target, shape=(target.shape[0], -1))

    return target, preds


def _compute_top_k_multiclass_stat_scores(
    target: Array,
    preds: Array,
    num_classes: int,
    top_k: int,
    ignore_index: Optional[Union[int, Tuple[int]]],
    *,
    xp: ModuleType,
) -> Tuple[Array, Array, Array, Array]:
    """Compute stat scores for multiclass tasks with `top_k` > 1.

    Notes
    -----
    - The `target` and `preds` arrays are assumed to be formatted such that
        the first dimension is the batch dimension, the second dimension is
        the class dimension, and any extra dimensions are flattened.

    """
    ignore_in = None
    if isinstance(ignore_index, int):
        ignore_in = 0 <= ignore_index <= num_classes - 1
    elif isinstance(ignore_index, tuple):
        ignore_in = all(0 <= i <= num_classes - 1 for i in ignore_index)

    if ignore_index is not None and not ignore_in:
        preds = clone(preds)
        target = clone(target)

        # NOTE (Dec. 2023): The 2021.12 version of the array API standard does
        # not define an `isin` method for arrays, so we have to use a workaround
        # here.
        if isinstance(ignore_index, int):
            idx = target == ignore_index
        else:
            idx = xp.zeros_like(target, dtype=xp.bool)
            for val in ignore_index:
                if val > num_classes - 1:
                    idx |= target == val
        target[idx] = num_classes

        # NOTE (Dec. 2023): The array API standard does not explicitly define
        # a `repeat` or `tile` method, so we use `concat` instead.
        idx = (
            xp.concat([xp.expand_dims(idx, axis=1)] * num_classes, axis=1)
            if preds.ndim > target.ndim
            else idx
        )
        preds[idx] = num_classes

    preds_oh = moveaxis(
        _select_topk(preds, top_k=top_k, axis=1),
        source=1,
        destination=-1,
    )
    target_oh = _to_one_hot(
        target,
        num_classes + 1 if ignore_index is not None and not ignore_in else num_classes,
    )

    if ignore_index is not None:
        if isinstance(ignore_index, int) and ignore_in:
            target_oh[target == ignore_index] = -1
        elif isinstance(ignore_index, tuple) and ignore_in:
            idx = xp.zeros_like(target, dtype=xp.bool)
            for val in ignore_index:
                idx |= target == val
            target_oh[idx] = -1
        else:
            preds_oh = preds_oh[..., :-1] if top_k == 1 else preds_oh
            target_oh = target_oh[..., :-1]
            target_oh[target == num_classes] = -1

    # NOTE (Dec. 2023): in the array API standard, `sum` only supports numeric
    # types, so we have to cast the boolean arrays to integers.
    sum_axis = (0, 1)
    tp = xp.sum(to_int((target_oh == preds_oh) & (target_oh == 1)), axis=sum_axis)
    fn = xp.sum(to_int((target_oh != preds_oh) & (target_oh == 1)), axis=sum_axis)
    fp = xp.sum(to_int((target_oh != preds_oh) & (target_oh == 0)), axis=sum_axis)
    tn = xp.sum(to_int((target_oh == preds_oh) & (target_oh == 0)), axis=sum_axis)

    return tn, fp, fn, tp


def _multiclass_stat_scores_update_state(
    target: Array,
    preds: Array,
    num_classes: int,
    top_k: int = 1,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    ignore_index: Optional[Union[int, Tuple[int]]] = None,
    *,
    xp: ModuleType,
) -> Tuple[Array, Array, Array, Array]:
    """Compute the components of a confusion matrix for multiclass tasks."""
    if top_k == 1:
        # get a flat *view* of the arrays if ignore_index is not None because
        # `remove_ignore_index` creates a copy of the arrays, so we can avoid
        # creating a copy of the arrays twice.
        preds = flatten(preds, copy=ignore_index is None)
        target = flatten(target, copy=ignore_index is None)

        if ignore_index is not None:
            target, preds = remove_ignore_index(
                target,
                preds,
                ignore_index=ignore_index,
            )

        if average == "micro":
            tp = xp.sum(to_int(preds == target))
            fp = xp.sum(to_int(preds != target))
            fn = xp.sum(to_int(preds != target))
            tn = num_classes * apc.size(preds) - (fp + fn + tp)
        else:
            unique_mapping = to_int(target) * num_classes + to_int(preds)
            bins = bincount(unique_mapping, minlength=num_classes**2)
            confmat = xp.reshape(bins, shape=(num_classes, num_classes))
            tp = xp.linalg.diagonal(confmat)
            fp = xp.sum(confmat, axis=0) - tp
            fn = xp.sum(confmat, axis=1) - tp
            tn = xp.sum(confmat) - (fp + fn + tp)
    else:  # top_k > 1
        tn, fp, fn, tp = _compute_top_k_multiclass_stat_scores(
            target,
            preds,
            num_classes,
            top_k=top_k,
            ignore_index=ignore_index,
            xp=xp,
        )

    return tn, fp, fn, tp


def _multilabel_stat_scores_validate_args(
    num_labels: int,
    threshold: float = 0.5,
    top_k: int = 1,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    ignore_index: Optional[Union[int, Tuple[int]]] = None,
) -> None:
    """Validate arguments."""
    if not isinstance(num_labels, int) or num_labels < 2:
        raise ValueError(
            "Expected argument `num_labels` to be an integer larger than 1, "
            f"but got {num_labels}.",
        )

    if not isinstance(top_k, int):
        raise TypeError(
            f"Expected `top_k` to be an integer, but {type(top_k)} was provided.",
        )
    if top_k < 1:
        raise ValueError(
            "Expected argument `top_k` to be an integer larger than or equal to 1, "
            f"but got {top_k}",
        )
    if top_k > num_labels:
        raise ValueError(
            "Expected argument `top_k` to be smaller or equal to `num_labels`, "
            f"but got {top_k} and {num_labels}",
        )

    if not (isinstance(threshold, float) and (0.0 <= threshold <= 1.0)):
        raise ValueError(
            "Expected argument `threshold` to be a float in the [0,1] range, "
            f"but got {threshold}.",
        )

    allowed_average = ("micro", "macro", "weighted", "none", None)
    if average not in allowed_average:
        raise ValueError(
            f"Expected argument `average` to be one of {allowed_average}, "
            f"but got {average}",
        )

    if ignore_index is not None and not (
        isinstance(ignore_index, int)
        or (
            isinstance(ignore_index, tuple)
            and all(isinstance(i, int) for i in ignore_index)
        )
    ):
        raise ValueError(
            "Expected argument `ignore_index` to either be `None`, an integer, "
            f"or a tuple of integers but got {ignore_index}",
        )


def _multilabel_stat_scores_validate_arrays(
    target: Array,
    preds: Array,
    num_labels: int,
    ignore_index: Optional[int] = None,
) -> ModuleType:
    """Validate multilabel input arrays."""
    return _multilabel_confusion_matrix_validate_arrays(
        target,
        preds,
        num_labels,
        ignore_index=ignore_index,
    )


def _multilabel_stat_scores_format_arrays(
    target: Array,
    preds: Array,
    threshold: float,
    top_k: int,
    ignore_index: Optional[int],
    *,
    xp: ModuleType,
) -> Tuple[Array, Array]:
    """Format multilabel input arrays.

    Notes
    -----
    - If `preds` are logits and `top_k=1`, they are converted to probabilities
      using the sigmoid function.
    - If `preds` are probabilities and `top_k=1`, they are converted to binary
      values using the `threshold` value.
    - If `top_k` > 1, the top `k` classes are selected for each sample.
    - Any extra dimensions in `preds` and `target` beyond the first two are
      flattened.
    - If `ignore_index` is not `None`, the corresponding values in `target` are
      set to -1.
    """
    if top_k == 1:
        if is_floating_point(preds):
            if not xp.all(to_int((preds >= 0)) * to_int((preds <= 1))):
                preds = sigmoid(preds)
            preds = to_int(preds > threshold)
    else:
        preds = _select_topk(preds, top_k=top_k, axis=1)

    preds = xp.reshape(preds, shape=(*preds.shape[:2], -1))
    target = xp.reshape(target, shape=(*target.shape[:2], -1))

    if ignore_index is not None:
        target = clone(target)
        target[target == ignore_index] = -1

    return target, preds


def _multilabel_stat_scores_update_state(
    target: Array,
    preds: Array,
    *,
    xp: ModuleType,
) -> Tuple[Array, Array, Array, Array]:
    """Compute stat scores for the given `target` and `preds` arrays."""
    sum_axis = (0, -1)
    tp = squeeze_all(xp.sum(to_int((target == preds) & (target == 1)), axis=sum_axis))
    fn = squeeze_all(xp.sum(to_int((target != preds) & (target == 1)), axis=sum_axis))
    fp = squeeze_all(xp.sum(to_int((target != preds) & (target == 0)), axis=sum_axis))
    tn = squeeze_all(xp.sum(to_int((target == preds) & (target == 0)), axis=sum_axis))

    return tn, fp, fn, tp
