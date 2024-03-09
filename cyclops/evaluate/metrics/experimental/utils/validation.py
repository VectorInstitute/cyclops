"""Utility functions for performing common input validations."""

from typing import Any, List, Literal

import array_api_compat as apc

from cyclops.evaluate.metrics.experimental.utils.types import Array


def is_floating_point(array: Array) -> bool:
    """Return `True` if the array has a floating-point datatype.

    Floating-point datatypes include:
    - `float32`
    - `float64`
    - `float16`
    - `bfloat16`

    """
    xp = apc.array_namespace(array)
    float_dtypes = _get_float_dtypes(xp)

    return array.dtype in float_dtypes


def is_numeric(*arrays: Array) -> bool:
    """Check if given arrays have numeric datatype.

    Numeric datatypes include:
    - `float32`
    - `float64`
    - `float16`
    - `bfloat16`
    - `int8`
    - `int16`
    - `int32`
    - `int64`
    - `uint8`
    - `uint16`
    - `uint32`
    - `uint64`

    Parameters
    ----------
    arrays : Array
        The arrays to check.

    Returns
    -------
    bool
        `True` if all of the arrays have a numeric datatype. `False` otherwise.

    """
    xp = apc.array_namespace(*arrays)
    numeric_dtypes = _get_int_dtypes(xp) + _get_float_dtypes(xp)

    return all(array.dtype in numeric_dtypes for array in arrays)


def _basic_input_array_checks(
    target: Array,
    preds: Array,
) -> None:
    """Perform basic validation of `target` and `preds`."""
    if not apc.is_array_api_obj(target):
        raise TypeError(
            "Expected `target` to be an array-API-compatible object, but got "
            f"{type(target)}.",
        )

    if not apc.is_array_api_obj(preds):
        raise TypeError(
            "Expected `preds` to be an array-API-compatible object, but got "
            f"{type(preds)}.",
        )

    if _is_empty(target) or _is_empty(preds):
        raise ValueError("Expected `target` and `preds` to be non-empty arrays.")

    if not is_numeric(target, preds):
        raise ValueError(
            "Expected `target` and `preds` to be numeric arrays, but got "
            f"{target.dtype} and {preds.dtype}, respectively.",
        )


def _check_average_arg(average: Literal["micro", "macro", "weighted", None]) -> None:
    """Validate the `average` argument."""
    if average not in ["micro", "macro", "weighted", None]:
        raise ValueError(
            f"Argument average has to be one of 'micro', 'macro', 'weighted', "
            f"or None, got {average}.",
        )


def _check_same_shape(target: Array, preds: Array) -> None:
    """Check if `target` and `preds` have the same shape."""
    if target.shape != preds.shape:
        raise ValueError(
            "Expected `target` and `preds` to have the same shape, but got `target` "
            f"with shape={target.shape} and `preds` with shape={preds.shape}.",
        )


def _get_float_dtypes(namespace: Any) -> List[Any]:
    """Return a list of floating-point dtypes.

    Notes
    -----
    The integer types `float16` and `bfloat16` are not defined in the API, but
    are included here as they are increasingly common in deep learning frameworks.

    """
    float_dtypes = [namespace.float32, namespace.float64]
    if hasattr(namespace, "float16"):
        float_dtypes.append(namespace.float16)
    if hasattr(namespace, "bfloat16"):
        float_dtypes.append(namespace.bfloat16)

    return float_dtypes


def _get_int_dtypes(namespace: Any) -> List[Any]:
    """Return a list of integer dtypes.

    Notes
    -----
    The integer types `uint16`, `uint32` and `uint64` are defined in the API
    standard but not in PyTorch. Although, PyTorch supports quantized integer
    types like `qint8` and `quint8`, but we omit them here because they are not
    part of the array API standard.
    The 2022.12 version of the array API standard includes a `isdtype` method
    that will eliminate the need for this function. The `array_api_compat`
    package currently (Nov. 2023) supports only the 2021.12 version of the
    standard, so we need to define this function for now.

    """
    int_dtypes = [
        namespace.int8,
        namespace.int16,
        namespace.int32,
        namespace.int64,
        namespace.uint8,
    ]

    if hasattr(namespace, "uint16"):
        int_dtypes.append(namespace.uint16)
    if hasattr(namespace, "uint32"):
        int_dtypes.append(namespace.uint32)
    if hasattr(namespace, "uint64"):
        int_dtypes.append(namespace.uint64)

    return int_dtypes


def _is_empty(array: Array) -> bool:
    """Return `True` if the array is empty."""
    numel = apc.size(array)
    return numel is not None and numel == 0
