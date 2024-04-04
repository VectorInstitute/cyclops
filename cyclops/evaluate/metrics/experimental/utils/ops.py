"""Utility functions for performing operations on array-API-compatible objects."""

# mypy: disable-error-code="no-any-return"
from collections import OrderedDict, defaultdict
from types import ModuleType
from typing import (
    Any,
    Callable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import array_api_compat as apc
import numpy as np
from array_api_compat.common._helpers import is_numpy_array, is_torch_array

from cyclops.evaluate.metrics.experimental.utils.types import Array
from cyclops.evaluate.metrics.experimental.utils.validation import (
    _get_int_dtypes,
    is_floating_point,
)


def apply_to_array_collection(  # noqa: PLR0911
    data: Any,
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Apply a function to an array or collection of arrays.

    Parameters
    ----------
    data : Any
        An array or collection of arrays.
    func : Callable[..., Any]
        A function to be applied to `data`.
    *args : Any
        Positional arguments to be passed to the function.
    **kwargs : Any
        Keyword arguments to be passed to the function.

    Returns
    -------
    Any
        The result of applying the function to the input data.

    """
    is_namedtuple = (
        isinstance(data, tuple)
        and hasattr(data, "_asdict")
        and hasattr(data, "_fields")
    )
    if apc.is_array_api_obj(data):
        return func(data, *args, **kwargs)
    if isinstance(data, list) and all(apc.is_array_api_obj(x) for x in data):
        return [func(x, *args, **kwargs) for x in data]
    if (isinstance(data, tuple) and not is_namedtuple) and all(
        apc.is_array_api_obj(x) for x in data
    ):
        return tuple(func(x, *args, **kwargs) for x in data)
    if isinstance(data, dict) and all(apc.is_array_api_obj(x) for x in data.values()):
        return {k: func(v, *args, **kwargs) for k, v in data.items()}

    elem_type = type(data)

    if isinstance(data, Mapping):
        out = []
        for k, v in data.items():
            out.append((k, apply_to_array_collection(v, func, *args, **kwargs)))
        if isinstance(data, defaultdict):
            return elem_type(data.default_factory, OrderedDict(out))
        return elem_type(OrderedDict(out))

    is_sequence = isinstance(data, Sequence) and not isinstance(data, str)
    if is_namedtuple or is_sequence:
        out = []
        for d in data:
            out.append(apply_to_array_collection(d, func, *args, **kwargs))
        return elem_type(*out) if is_namedtuple else elem_type(out)
    return data


def bincount(
    array: Array,
    weights: Optional[Array] = None,
    minlength: int = 0,
) -> Array:
    """Count the number of occurrences of each value in an array of non-negative ints.

    Parameters
    ----------
    array : Array
        The input array.
    weights : Array, optional, default=None
        An array of weights, of the same shape as `array`. Each value in `array`
        only contributes its associated weight towards the bin count (instead of 1).
        If `weights` is None, all values in `array` are counted equally.
    minlength : int, optional, default=0
        A minimum number of bins for the output array. If `minlength` is greater
        than the largest value in `array`, then the output array will have
        `minlength` bins.

    Returns
    -------
    Array
        The result of binning the input array.

    Raises
    ------
    ValueError
        If `array` is not a 1D array of non-negative integers, `weights` is not None
        and `weights` and `array` do not have the same shape, or `minlength` is
        negative.

    Examples
    --------
    >>> import numpy.array_api as np
    >>> from cyclops.evaluate.metrics.experimental.utils.ops import bincount
    >>> x = np.asarray([0, 1, 1, 2, 2, 2])
    >>> bincount(x)
    Array([1, 2, 3], dtype=int64)
    >>> bincount(x, weights=np.asarray([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
    Array([0.5, 1. , 1.5], dtype=float64)
    >>> bincount(x, minlength=5)
    Array([1, 2, 3, 0, 0], dtype=int64)

    """
    xp = apc.array_namespace(array)

    if not (isinstance(minlength, int) and minlength >= 0):
        raise ValueError(
            "Expected `min_length` to be a non-negative integer. "
            f"Got minlength={minlength}.",
        )

    if apc.size(array) == 0:
        return xp.zeros(shape=(minlength,), dtype=xp.int64, device=apc.device(array))

    if array.ndim != 1:
        raise ValueError(f"Expected `array` to be a 1D array. Got {array.ndim}D array.")

    if array.dtype not in _get_int_dtypes(namespace=xp):
        raise ValueError(
            f"Expected `array` to be an integer array. Got {array.dtype} type.",
        )

    if xp.any(array < 0):
        raise ValueError("`array` must contain only non-negative integers.")

    if weights is not None and array.shape != weights.shape:
        raise ValueError(
            "Expected `array` and `weights` to have the same shape. "
            f"Got array.shape={array.shape} and weights.shape={weights.shape}.",
        )

    size = int(xp.max(array)) + 1
    size = max(size, int(minlength))
    device = apc.device(array)

    bincount = xp.astype(
        array == xp.arange(size, device=device)[:, None],
        weights.dtype if weights is not None else xp.int32,
        copy=False,
    )
    return xp.sum(bincount * (weights if weights is not None else 1), axis=1)


def clone(array: Array) -> Array:
    """Create a copy of an array.

    Parameters
    ----------
    array : Array
        The input array.

    Returns
    -------
    Array
        A copy of the input array.

    Notes
    -----
    This method is a temporary workaround for the lack of support for the `copy`
    or `clone` method in the array API standard. The 2023 version of the standard
    may include a copy method. See: https://github.com/data-apis/array-api/issues/495

    Examples
    --------
    >>> import numpy.array_api as np
    >>> from cyclops.evaluate.metrics.experimental.utils.ops import clone
    >>> x = np.zeros((1, 2, 3))
    >>> y = x
    >>> y is x
    True
    >>> y = clone(x)
    >>> y is x
    False

    """
    xp = apc.array_namespace(array)
    return xp.asarray(array, device=apc.device(array), copy=True)


def dim_zero_cat(x: Union[Array, List[Array], Tuple[Array]]) -> Array:
    """Concatenation along the zero dimension."""
    if apc.is_array_api_obj(x) or not x:  # covers empty list/tuple
        return cast(Array, x)

    if not isinstance(x, (list, tuple)):
        raise TypeError(
            "Expected `x` to be an Array or a list/tuple of Arrays. "
            f"Got {type(x)} instead.",
        )

    xp = apc.array_namespace(x[0])
    x_ = []
    for el in x:
        if not apc.is_array_api_obj(el):
            raise TypeError(
                "Expected `x` to be an Array or a list/tuple of Arrays. "
                f"Got a list/tuple containing a {type(el)} instead.",
            )
        if apc.size(el) == 1 and el.ndim == 0:
            x_.append(xp.expand_dims(el, axis=0))
        else:
            x_.append(el)

    if not x_:  # empty list
        raise ValueError("No samples to concatenate")
    return xp.concat(x_, axis=0)


def dim_zero_max(x: Array) -> Array:
    """Max along the zero dimension."""
    xp = apc.array_namespace(x)
    return xp.max(x, axis=0)


def dim_zero_mean(x: Array) -> Array:
    """Average along the zero dimension."""
    xp = apc.array_namespace(x)
    x = x if is_floating_point(x) else xp.astype(x, xp.float32, copy=False)
    return xp.mean(x, axis=0)


def dim_zero_min(x: Array) -> Array:
    """Min along the zero dimension."""
    xp = apc.array_namespace(x)
    return xp.min(x, axis=0)


def dim_zero_sum(x: Array) -> Array:
    """Summation along the zero dimension."""
    xp = apc.array_namespace(x)
    return xp.sum(x, axis=0)


def flatten(array: Array, copy: bool = True) -> Array:
    """Flatten an array.

    Parameters
    ----------
    array : Array
        The input array.

    Returns
    -------
    Array
        The flattened array.
    copy : bool, optional, default=True
        Whether to copy the input array.

    Notes
    -----
    This method is a temporary workaround for the lack of support for the `flatten`
    method in the array API standard.

    Examples
    --------
    >>> import numpy.array_api as np
    >>> from cyclops.evaluate.metrics.experimental.utils.ops import flatten
    >>> x = np.zeros((1, 2, 3))
    >>> x.shape
    (1, 2, 3)
    >>> flatten(x).shape
    (6,)

    """
    xp = apc.array_namespace(array)
    return xp.asarray(
        xp.reshape(array, shape=(-1,)),
        device=apc.device(array),
        copy=copy if copy else None,
    )


def flatten_seq(inp: Sequence[Any]) -> List[Any]:
    """Flatten a nested sequence into a single list.

    Parameters
    ----------
    inp : Sequence
        The input sequence.

    Returns
    -------
    List[Any]
        The flattened sequence.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental.utils.ops import flatten_seq
    >>> x = [[1, 2, 3], [4, 5, 6]]
    >>> flatten_seq(x)
    [1, 2, 3, 4, 5, 6]

    """
    if not isinstance(inp, Sequence):
        raise TypeError("Input must be a Sequence")

    if len(inp) == 0:
        return []

    if isinstance(inp, str) and len(inp) == 1:
        return [inp]

    result = []
    for sublist in inp:
        if isinstance(sublist, Sequence):
            result.extend(flatten_seq(sublist))
        else:
            result.append(sublist)
    return result


def moveaxis(
    array: Array,
    source: Union[int, Tuple[int]],
    destination: Union[int, Tuple[int]],
) -> Array:
    """Move given array axes to new positions.

    Parameters
    ----------
    array : Array
        The input array.
    source : int or Tuple[int]
        Original positions of the axes to move. These must be unique.
    destination : int or Tuple[int]
        Destination positions for each of the original axes. These must also be
        unique.

    Returns
    -------
    Array
        Array with moved axes. This array is a view of the input array.

    Raises
    ------
    ValueError
        If the source and destination axes are not unique or if the number of
        elements in `source` and `destination` are not equal.

    Notes
    -----
    A similar method has been added to the array API standard in v2022.12. See:
    https://data-apis.org/array-api/draft/API_specification/generated/array_api.moveaxis.html
    The `array_api_compat` library does not yet support that version of the standard,
    so we define this method here for now.

    Examples
    --------
    >>> import numpy.array_api as np
    >>> from cyclops.evaluate.metrics.experimental.utils.ops import moveaxis
    >>> x = np.zeros((1, 2, 3))
    >>> moveaxis(x, 0, 1).shape
    (2, 1, 3)

    """
    if isinstance(source, int):
        source = (source,)
    if isinstance(destination, int):
        destination = (destination,)

    if (isinstance(source, tuple) and not isinstance(destination, tuple)) or (
        isinstance(destination, tuple) and not isinstance(source, tuple)
    ):
        raise ValueError(
            "`source` and `destination` must both be tuples or both be integers",
        )

    if len(set(source)) != len(source) or len(set(destination)) != len(destination):
        raise ValueError("`source` and `destination` must not contain duplicate values")

    if len(source) != len(destination):
        raise ValueError(
            "`source` and `destination` must have the same number of elements",
        )

    xp = apc.array_namespace(array)
    num_dims = array.ndim
    if (
        max(source) >= num_dims
        or max(destination) >= num_dims
        or abs(min(source)) > num_dims
        or abs(min(destination)) > num_dims
    ):
        raise ValueError(
            "Values in `source` and `destination` are out of bounds for `array` "
            f"with {num_dims} dimensions",
        )

    # normalize negative indices
    src_ = tuple([src % num_dims for src in source])
    dest_ = tuple([dest % num_dims for dest in destination])

    order = [n for n in range(num_dims) if n not in src_]

    for src, dest in sorted(zip(dest_, src_)):
        order.insert(src, dest)

    return xp.permute_dims(array, order)


def remove_ignore_index(
    target: Array,
    preds: Array,
    ignore_index: Optional[Union[Tuple[int, ...], int]],
) -> Tuple[Array, Array]:
    """Remove the samples at the indices where target values match `ignore_index`.

    Parameters
    ----------
    target : Array
        The target array.
    preds : Array
        The predictions array.
    ignore_index : int or Tuple[int], optional, default=None
        The index or indices to ignore. If None, no indices will be ignored.

    Returns
    -------
    Tuple[Array, Array]
        The `target` and `preds` arrays with the samples at the indices where target
        values match `ignore_index` removed.
    """
    if ignore_index is None:
        return target, preds

    if not (
        isinstance(ignore_index, int)
        or (
            isinstance(ignore_index, tuple)
            and all(isinstance(x, int) for x in ignore_index)
        )
    ):
        raise TypeError(
            "Expected `ignore_index` to be an integer or a tuple of integers. "
            f"Got {type(ignore_index)} instead.",
        )

    xp = apc.array_namespace(target, preds)

    if isinstance(ignore_index, int):
        mask = target == ignore_index
    else:
        mask = xp.zeros_like(target, dtype=xp.bool)
        for index in ignore_index:
            mask = xp.logical_or(mask, target == index)

    return clone(target[~mask]), clone(preds[~mask])


def safe_divide(
    numerator: Array,
    denominator: Array,
) -> Array:
    """Divide two arrays and return zero if denominator is zero.

    Parameters
    ----------
    numerator : Array
        The numerator array.
    denominator : Array
        The denominator array.

    Returns
    -------
    quotient : Array
        The quotient of the two arrays.

    Raises
    ------
    ValueError
        If `numerator` and `denominator` do not have the same shape.

    Examples
    --------
    >>> import numpy.array_api as np
    >>> from cyclops.evaluate.metrics.experimental.utils.ops import safe_divide
    >>> x = np.asarray([1.1, 2.0, 3.0])
    >>> y = np.asarray([1.1, 0.0, 3.0])
    >>> safe_divide(x, y)
    Array([1., 0., 1.], dtype=float64)

    """
    xp = apc.array_namespace(numerator, denominator)

    numerator = (
        numerator
        if is_floating_point(numerator)
        else xp.astype(numerator, xp.float32, copy=False)
    )
    denominator = (
        denominator
        if is_floating_point(denominator)
        else xp.astype(denominator, xp.float32, copy=False)
    )

    return xp.where(
        denominator == 0,
        xp.asarray(0, dtype=xp.float32, device=apc.device(numerator)),
        numerator / denominator,
    )


def sigmoid(array: Array) -> Array:
    """Compute the sigmoid of an array.

    Parameters
    ----------
    array : Array
        The input array.

    Returns
    -------
    Array
        The sigmoid of the input array.

    Examples
    --------
    >>> import numpy.array_api as np
    >>> from cyclops.evaluate.metrics.experimental.utils.ops import sigmoid
    >>> x = np.asarray([1.1, 2.0, 3.0])
    >>> sigmoid(x)
    Array([0.75026011, 0.88079708, 0.95257413], dtype=float64)

    """
    xp = apc.array_namespace(array)
    if apc.size(array) == 0:
        return xp.asarray([], dtype=xp.float32, device=apc.device(array))

    array = (
        array if is_floating_point(array) else xp.astype(array, xp.float32, copy=False)
    )

    exp_array = xp.exp(array)
    return xp.where(
        array >= 0,
        1 / (1 + xp.exp(-array)),
        exp_array / (1 + exp_array),
    )


def softmax(array: Array, axis: Optional[int] = None) -> Array:
    """Compute the softmax of an array.

    Parameters
    ----------
    array : Array
        The input array.
    axis : int, optional, default=None
        The axis along which to compute the softmax. If None, the softmax will be
        computed over all elements in the array.

    Returns
    -------
    Array
        The softmax of the input array.

    Examples
    --------
    >>> import numpy.array_api as np
    >>> from cyclops.evaluate.metrics.experimental.utils.ops import softmax
    >>> x = np.asarray([[1.1, 2.0, 3.0], [2.0, 1.0, 0.5]])
    >>> softmax(x, axis=1)
    Array([[0.09856589, 0.24243297, 0.65900114],
           [0.62853172, 0.2312239 , 0.14024438]], dtype=float64)

    """
    xp = apc.array_namespace(array)

    x_max = xp.max(array, axis=axis, keepdims=True)
    exp_x_shifted = xp.exp(array - x_max)

    return safe_divide(exp_x_shifted, xp.sum(exp_x_shifted, axis=axis, keepdims=True))


def squeeze_all(array: Array) -> Array:
    """Remove all singleton dimensions from an array.

    Parameters
    ----------
    array : Array
        An array to squeeze.

    Returns
    -------
    Array
        The squeezed array.

    Examples
    --------
    >>> import numpy.array_api as np
    >>> from cyclops.evaluate.metrics.experimental.utils.ops import squeeze_all
    >>> x = np.zeros((1, 2, 1, 3, 1, 4))
    >>> x.shape
    (1, 2, 1, 3, 1, 4)
    >>> squeeze_all(x).shape
    (2, 3, 4)
    """
    xp = apc.array_namespace(array)
    singleton_axes = tuple(i for i in range(array.ndim) if array.shape[i] == 1)
    if len(singleton_axes) == 0:
        return array

    return xp.squeeze(array, axis=singleton_axes)


def to_int(array: Array) -> Array:
    """Convert the data type of an array to a 64-bit integer type.

    Parameters
    ----------
    array : Array
        The input array.

    Returns
    -------
    Array
        The input array converted to an integer array.

    Examples
    --------
    >>> import numpy.array_api as np
    >>> from cyclops.evaluate.metrics.experimental.utils.ops import to_int
    >>> x = np.asarray([1.1, 2.0, 3.0])
    >>> to_int(x)
    Array([1, 2, 3], dtype=int64)

    """
    xp = apc.array_namespace(array)
    return xp.astype(array, xp.int64, copy=False)


def _adjust_weight_apply_average(
    score: Array,
    average: Optional[Literal["macro", "weighted", "none"]],
    is_multilabel: bool,
    *,
    tp: Array,
    fp: Array,
    fn: Array,
    xp: ModuleType,
) -> Array:
    """Apply the specified averaging method to the accuracy scores."""
    if average is None or average == "none":
        return score
    if average == "weighted":
        weights = tp + fn
    else:  # average == "macro"
        weights = xp.ones_like(score)
        if not is_multilabel:
            weights[tp + fp + fn == 0] = 0.0

    weights = xp.astype(weights, xp.float32, copy=False)
    return xp.sum(  # type: ignore[no-any-return]
        safe_divide(
            weights * score,
            xp.sum(weights, axis=-1, dtype=score.dtype, keepdims=True),
        ),
        axis=-1,
        dtype=score.dtype,
    )


def _array_indexing(arr: Array, idx: Array) -> Array:
    try:
        return arr[idx]
    except IndexError:
        xp = apc.array_namespace(arr, idx)
        np_idx = np.from_dlpack(apc.to_device(idx, "cpu"))
        np_arr = np.from_dlpack(apc.to_device(arr, "cpu"))[np_idx]
        return xp.asarray(np_arr, dtype=arr.dtype, device=apc.device(arr))


def _auc_compute(
    x: Array,
    y: Array,
    direction: Optional[float] = None,
    axis: int = -1,
    reorder: bool = False,
) -> Array:
    """Compute the area under the curve using the trapezoidal rule.

    Adapted from: https://github.com/Lightning-AI/torchmetrics/blob/fd2e332b66df1b484728efedad9d430c7efae990/src/torchmetrics/utilities/compute.py#L99-L115

    Parameters
    ----------
    x : Array
        The x-coordinates of the curve.
    y : Array
        The y-coordinates of the curve.
    direction : float, optional, default=None
        The direction of the curve. If None, the direction will be inferred from the
        values in `x`.
    axis : int, optional, default=-1
        The axis along which to compute the area under the curve.
    reorder : bool, optional, default=False
        Whether to sort the arrays `x` and `y` by `x` before computing the area under
        the curve.
    """
    xp = apc.array_namespace(x, y)
    if reorder:
        x, x_idx = xp.sort(x, stable=True)
        y = _array_indexing(y, x_idx)

    if direction is None:
        dx = x[1:] - x[:-1]
        if xp.any(dx < 0):
            if xp.all(dx <= 0):
                direction = -1.0
            else:
                raise ValueError(
                    "The array `x` is neither increasing or decreasing. "
                    "Try setting the reorder argument to `True`.",
                )
        else:
            direction = 1.0

    return xp.astype(_trapz(y, x, axis=axis) * direction, xp.float32, copy=False)


def _cumsum(x: Array, axis: Optional[int] = None, dtype: Optional[Any] = None) -> Array:
    """Compute the cumulative sum of an array along a given axis.

    Parameters
    ----------
    x : Array
        The input array.
    axis : int, optional, default=None
        The axis along which to compute the cumulative sum. If None, the input array
        will be flattened before computing the cumulative sum.
    dtype : Any, optional, default=None
        The data type of the output array. If None, the data type of the input array
        will be used.

    Returns
    -------
    Array
        An array containing the cumulative sum of the input array along the given axis.
    """
    xp = apc.array_namespace(x)
    if hasattr(xp, "cumsum"):
        return xp.cumsum(x, axis, dtype=dtype)

    if axis is None:
        x = flatten(x)
        axis = 0

    if axis < 0 or axis >= x.ndim:
        raise ValueError("Invalid axis value")

    if dtype is None:
        dtype = x.dtype

    if axis < 0:
        axis += x.ndim

    if int(apc.size(x) or 0) == 0:
        return x

    result = xp.empty_like(x, dtype=dtype, device=apc.device(x))

    # create slice object with `axis` at the appropriate position
    curr_indices = [slice(None)] * x.ndim
    prev_indices = [slice(None)] * x.ndim

    curr_indices[axis] = 0  # type: ignore[call-overload]
    result[tuple(curr_indices)] = x[tuple(curr_indices)]
    for i in range(1, x.shape[axis]):
        prev_indices[axis] = i - 1  # type: ignore[call-overload]
        curr_indices[axis] = i  # type: ignore[call-overload]
        result[tuple(curr_indices)] = (
            result[tuple(prev_indices)] + x[tuple(curr_indices)]
        )

    return result


def _diff(
    a: Array,
    n: int = 1,
    axis: int = -1,
    prepend: Optional[Array] = None,
    append: Optional[Array] = None,
) -> Array:
    """Calculate the n-th discrete difference along the given axis.

    Adapted from: https://github.com/numpy/numpy/blob/v1.26.0/numpy/lib/function_base.py#L1324-L1454

    Parameters
    ----------
    a : Array
        Input array.
    n : int, optional, default=1
        The number of times values are differenced. If zero, the input is returned
        as-is.
    axis : int, optional, default=-1
        The axis along which the difference is taken, default is the last axis.
    prepend : Array, optional, default=None
        Values to prepend to `a` along `axis` prior to performing the difference.
    append : Array, optional, default=None
        Values to append to `a` along `axis` after performing the difference.

    Returns
    -------
    Array
        The n-th differences. The shape of the output is the same as `a` except along
        `axis` where the dimension is smaller by `n`. The type of the output is the
        same as the type of the difference between any two elements of `a`. This is
        the same type as `a` in most cases.
    """
    xp = apc.array_namespace(a)

    if prepend is not None and not apc.is_array_api_obj(prepend):
        raise TypeError(
            "Expected argument `prepend` to be an object that is compatible with the "
            f"Python array API standard. Got {type(prepend)} instead.",
        )
    if append is not None and not apc.is_array_api_obj(append):
        raise TypeError(
            "Expected argument `append` to be an object that is compatible with the "
            f"Python array API standard. Got {type(append)} instead.",
        )

    if n == 0:
        return a
    if n < 0:
        raise ValueError("order must be non-negative but got " + repr(n))

    ndim = a.ndim
    if ndim == 0:
        raise ValueError("diff requires input that is at least one dimensional")

    combined = []
    if prepend is not None:
        if prepend.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            prepend = xp.broadcast_to(prepend, tuple(shape))
        combined.append(prepend)

    combined.append(a)

    if append is not None:
        if append.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            append = xp.broadcast_to(append, tuple(shape))
        combined.append(append)

    if len(combined) > 1:
        a = xp.concat(combined, axis)

    slice1 = [slice(None)] * ndim
    slice2 = [slice(None)] * ndim
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)  # type: ignore[assignment]
    slice2 = tuple(slice2)  # type: ignore[assignment]

    op = xp.not_equal if a.dtype == xp.bool else xp.subtract
    for _ in range(n):
        a = op(a[slice1], a[slice2])

    return a


def _interp(x: Array, xcoords: Array, ycoords: Array) -> Array:
    """Perform linear interpolation for 1D arrays.

    Parameters
    ----------
    x : Array
        The 1D array of points on which to interpolate.
    xcoords : Array
        The 1D array of x-coordinates containing known data points.
    ycoords : Array
        The 1D array of y-coordinates containing known data points.

    Returns
    -------
    Array
        The interpolated values.
    """
    xp = apc.array_namespace(x, xcoords, ycoords)
    if hasattr(xp, "interp"):
        return xp.interp(x, xcoords, ycoords)

    if is_torch_array(x):
        weight = (x - xcoords[0]) / (xcoords[-1] - xcoords[0])
        return xp.lerp(ycoords[0], ycoords[-1], weight)

    if xcoords.ndim != 1 or ycoords.ndim != 1:
        raise ValueError(
            "Expected `xcoords` and `ycoords` to be 1D arrays. "
            f"Got xcoords.ndim={xcoords.ndim} and ycoords.ndim={ycoords.ndim}.",
        )
    if xcoords.shape[0] != ycoords.shape[0]:
        raise ValueError(
            "Expected `xcoords` and `ycoords` to have the same shape along axis 0. "
            f"Got xcoords.shape={xcoords.shape} and ycoords.shape={ycoords.shape}.",
        )

    m = safe_divide(ycoords[1:] - ycoords[:-1], xcoords[1:] - xcoords[:-1])
    b = ycoords[:-1] - (m * xcoords[:-1])

    # create slices to work for any ndim of x and xcoords
    indices = (
        xp.sum(
            xp.astype(x[..., None] >= xcoords[None, ...], xp.int32, copy=False), axis=1
        )
        - 1
    )
    _min_val = xp.asarray(0, dtype=xp.int32, device=apc.device(x))
    _max_val = xp.asarray(
        m.shape[0] if m.ndim > 0 else 1 - 1,
        dtype=xp.int32,
        device=apc.device(x),
    )
    # clamp indices to _min_val and _max_val
    indices = xp.where(indices < _min_val, _min_val, indices)
    indices = xp.where(indices > _max_val, _max_val, indices)

    return _array_indexing(m, indices) * x + _array_indexing(b, indices)


def _select_topk(  # noqa: PLR0912
    scores: Array,
    top_k: int = 1,
    axis: int = -1,
) -> Array:
    """Compute a one-hot array indicating the top-k scores along an axis.

    Parameters
    ----------
    scores : Array
        An array of scores of shape `[..., C, ...]` where `C` is in the axis `axis`.
    top_k : int, optional, default=1
        The number of top scores to select.
    axis : int, optional, default=-1
        The axis along which to select the top-k scores.

    Returns
    -------
    Array
        A one-hot array indicating the top-k scores along an axis.

    Raises
    ------
    ValueError
        If `top_k` is not positive, `axis` is greater than or equal to the number of
        dimensions in `scores`, or `top_k` is greater than the size of `scores` along
        `axis`.

    Warnings
    --------
    This method may be slow or memory-intensive for some array namespaces. See
    `Notes` for more details.

    Notes
    -----
    This method can be slow or memory-intensive for some array namespaces due to
    several factors:
    1. The use of `argsort` to fully sort the array as opposed to a partial sort.
    However, an upcoming version of the array API will include a `topk` method
    that will be more efficient. See https://github.com/data-apis/array-api/issues/629
    2. The lack of support for advanced indexing in the array API standard.
    3. The lack of support for methods that set elements along an axis, like
    `np.put_along_axis` or `torch.scatter`.

    Examples
    --------
    >>> import numpy.array_api as np
    >>> from cyclops.evaluate.metrics.experimental.utils.ops import _select_topk
    >>> x = np.asarray([[1.1, 2.0, 3.0], [2.0, 1.0, 0.5]])
    >>> _select_topk(x, top_k=2, axis=1)
    Array([[0, 1, 1],
           [1, 1, 0]], dtype=int32)
    """
    xp = apc.array_namespace(scores)

    if axis >= scores.ndim:
        raise ValueError(f"`axis={axis}` must be less than `scores.ndim={scores.ndim}`")
    if top_k <= 0:
        raise ValueError(f"`top_k` must be a positive integer, got {top_k}")
    if scores.ndim == 0 and top_k != 1:
        raise ValueError("`top_k` must be 1 for 0-dim scores, got {top_k}")
    if top_k > scores.shape[axis]:
        raise ValueError(
            f"`top_k={top_k}` must be less than or equal to "
            f"`scores.shape[axis]={scores.shape[axis]}`",
        )

    if top_k == 1:  # more efficient than argsort for top_k=1
        topk_indices = xp.argmax(scores, axis=axis, keepdims=True)
    else:
        topk_indices = xp.argsort(scores, axis=axis, descending=True, stable=False)
        slice_indices = [slice(None)] * scores.ndim
        slice_indices[axis] = slice(None, top_k)
        topk_indices = topk_indices[tuple(slice_indices)]

    zeros = xp.zeros_like(scores, dtype=xp.int32)

    if is_torch_array(scores):
        return zeros.scatter(axis, topk_indices, 1)
    if is_numpy_array(scores):
        xp.put_along_axis(zeros, topk_indices, 1, axis)
        return zeros

    result = np.zeros(scores.shape, dtype=np.int32)
    topk_indices = np.from_dlpack(apc.to_device(topk_indices, "cpu"))
    np.put_along_axis(result, topk_indices, 1, axis)

    return xp.asarray(result, device=apc.device(scores))


def _searchsorted(
    a: Array,
    v: Array,
    side: str = "left",
    sorter: Optional[Array] = None,
) -> Array:
    """Find indices where elements of `v` should be inserted to maintain order.

    Parameters
    ----------
    a : Array
        Input array. Must be sorted in ascending order if `sorter` is `None`.
    v : Array
        Values to insert into `a`.
    side : {'left', 'right'}, optional, default='left'
        If 'left', the index of the first suitable location found is given.
        If 'right', return the last such index. If there is no suitable index,
        return either 0 or `N` (where N is the length of `a`).
    sorter : Array, optional, default=None
        An optional array of integer indices that sort array `a` into ascending order.
        This is typically the result of `argsort`.

    Returns
    -------
    Array
        Array of insertion points with the same shape as `v`.

    Warnings
    --------
    This method uses `numpy.from_dlpack` to convert the input arrays to NumPy arrays
    and then uses `numpy.searchsorted` to perform the search. This may result in
    unexpected behavior for some array namespaces.

    """
    xp = apc.array_namespace(a, v)
    if hasattr(xp, "searchsorted"):
        return xp.searchsorted(a, v, side=side, sorter=sorter)

    np_a = np.from_dlpack(apc.to_device(a, "cpu"))
    np_v = np.from_dlpack(apc.to_device(v, "cpu"))
    np_sorter = (
        np.from_dlpack(apc.to_device(sorter, "cpu")) if sorter is not None else None
    )
    np_result = np.searchsorted(np_a, np_v, side=side, sorter=np_sorter)  # type: ignore[call-overload]
    return xp.asarray(np_result, dtype=xp.int32, device=apc.device(a))


def _to_one_hot(
    array: Array,
    num_classes: Optional[int] = None,
) -> Array:
    """Convert an array of integer labels to a one-hot encoded array.

    Parameters
    ----------
    array : Array
        An array of integer labels.
    num_classes : int, optional, default=None
        The number of classes. If not provided, the number of classes will be inferred
        from the array.

    Returns
    -------
    Array
        A one-hot encoded representation of `array`.

    Warnings
    --------
    This method can be slow or memory-intensive for some array namespaces due to
    the lack of support for advanced indexing in the array API standard.

    """
    xp = apc.array_namespace(array)

    input_shape = array.shape
    if array.dtype not in _get_int_dtypes(namespace=xp):
        array = to_int(array)
    array = flatten(array)

    arr_device = apc.device(array)
    n = array.shape[0]
    if num_classes is None:
        unique_values = xp.unique_values(array)
        num_classes = int(apc.size(unique_values))

    categorical = xp.zeros((n, num_classes), dtype=xp.int64, device=arr_device)
    try:  # advanced indexing
        categorical[xp.arange(n, device=arr_device), array] = 1
    except IndexError:
        indices = xp.stack([xp.arange(n, device=arr_device), array], axis=-1)
        for idx in range(indices.shape[0]):
            categorical[tuple(indices[idx, ...])] = 1

    output_shape = input_shape + (num_classes,)
    return xp.reshape(categorical, output_shape)


def _trapz(
    y: Array,
    x: Optional[Array] = None,
    dx: float = 1.0,
    axis: int = -1,
) -> Array:
    """Integrate along the given axis using the composite trapezoidal rule.

    Adapted from: https://github.com/cupy/cupy/blob/v12.3.0/cupy/_math/sumprod.py#L580-L626

    Parameters
    ----------
    y : Array
        Input array to integrate.
    x : Array, optional, default=None
        Sample points over which to integrate. If `x` is None, the sample points are
        assumed to be evenly spaced `dx` apart.
    dx : float, optional, default=1.0
        Spacing between sample points when `x` is None.
    axis : int, optional, default=-1
        Axis along which to integrate.

    Returns
    -------
    Array
        Definite integral as approximated by trapezoidal rule.
    """
    xp = apc.array_namespace(y)

    if not apc.is_array_api_obj(y):
        raise TypeError(
            "The type for `y` should be compatible with the Python array API standard.",
        )

    if x is None:
        d = dx
    else:
        if not apc.is_array_api_obj(x):
            raise TypeError(
                "The type for `x` should be compatible with the Python array API standard.",
            )
        if x.ndim == 1:
            d = _diff(x)  # type: ignore[assignment]
            # reshape to correct shape
            shape = [1] * y.ndim
            shape[axis] = d.shape[0]  # type: ignore[attr-defined]
            d = xp.reshape(d, shape)
        else:
            d = _diff(x, axis=axis)  # type: ignore[assignment]

    ndim = y.ndim
    slice1 = [slice(None)] * ndim
    slice2 = [slice(None)] * ndim
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    product = d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0
    return xp.sum(product, dtype=xp.float32, axis=axis)
