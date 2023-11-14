"""Utility functions for performing operations on array-API-compatible objects."""
import warnings
from collections import OrderedDict, defaultdict
from typing import (
    Any,
    Callable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import array_api_compat as apc
from array_api_compat.common._helpers import _is_numpy_array, _is_torch_array
from numpy.core.multiarray import normalize_axis_index  # type: ignore

from cyclops.evaluate.metrics.experimental.utils.typing import Array
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
    Array([1, 2, 3, 0, 0], dtype=int32)

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
        return x

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
    x = x if is_floating_point(x) else xp.astype(x, xp.float32)
    return xp.mean(x, axis=0)


def dim_zero_min(x: Array) -> Array:
    """Min along the zero dimension."""
    xp = apc.array_namespace(x)
    return xp.min(x, axis=0)


def dim_zero_sum(x: Array) -> Array:
    """Summation along the zero dimension."""
    xp = apc.array_namespace(x)
    return xp.sum(x, axis=0)


def flatten(array: Array) -> Array:
    """Flatten an array.

    Parameters
    ----------
    array : Array
        The input array.

    Returns
    -------
    Array
        The flattened array.

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
        copy=True,
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
        mask = xp.zeros_like(target, dtype=xp.bool, device=apc.device(target))
        for index in ignore_index:
            mask = xp.logical_or(mask, target == index)

    return clone(target[~mask]), clone(preds[~mask])


def safe_divide(numerator: Array, denominator: Array) -> Array:
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
        numerator if is_floating_point(numerator) else xp.astype(numerator, xp.float32)
    )
    denominator = (
        denominator
        if is_floating_point(denominator)
        else xp.astype(denominator, xp.float32)
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

    array = array if is_floating_point(array) else xp.astype(array, xp.float32)

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
    Array([1, 2, 3], dtype=int32)

    """
    xp = apc.array_namespace(array)
    return xp.astype(array, xp.int64, copy=False)


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
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if axis >= scores.ndim:
        raise ValueError("axis must be less than scores.ndim")
    if scores.ndim == 0 and top_k != 1:
        raise ValueError("top_k must be 1 for 0-dim scores")
    if top_k > scores.shape[axis]:
        raise ValueError("top_k must be less than or equal to scores.shape[axis]")

    if top_k == 1:  # more efficient than argsort for top_k=1
        topk_indices = xp.argmax(scores, axis=axis, keepdims=True)
    else:
        topk_indices = xp.argsort(scores, axis=axis, descending=True, stable=False)
        slice_indices = [slice(None)] * scores.ndim
        slice_indices[axis] = slice(None, top_k)
        topk_indices = topk_indices[tuple(slice_indices)]

    zeros = xp.zeros_like(scores, dtype=xp.int32, device=apc.device(scores))

    if _is_torch_array(scores):
        return zeros.scatter(axis, topk_indices, 1)
    if _is_numpy_array(scores):
        return xp.put_along_axis(zeros, topk_indices, 1, axis)

    axis = normalize_axis_index(axis, scores.ndim)

    # --- begin code copied from numpy ---
    # from https://github.com/numpy/numpy/blob/v1.26.0/numpy/lib/shape_base.py#L27
    shape_ones = (1,) * topk_indices.ndim
    dest_dims = list(range(axis)) + [None] + list(range(axis + 1, topk_indices.ndim))

    # build a fancy index, consisting of orthogonal aranges, with the
    # requested index inserted at the right location
    fancy_index = []
    for dim, n in zip(dest_dims, scores.shape):
        if dim is None:
            fancy_index.append(topk_indices)
        else:
            ind_shape = shape_ones[:dim] + (-1,) + shape_ones[dim + 1 :]
            fancy_index.append(xp.reshape(xp.arange(n), shape=(ind_shape)))
    # --- end of code copied from numpy ---

    indices = xp.broadcast_arrays(*fancy_index)
    indices = xp.stack(indices, axis=-1)
    indices = xp.reshape(indices, shape=(-1, indices.shape[-1]))

    try:  # advanced indexing
        zeros[tuple(indices.T)] = 1
    except IndexError:
        warnings.warn(
            "The `select_topk` method is slow and memory-intensive for the array "
            f"namespace '{xp.__name__}' and will be deprecated in a future release."
            "Consider writing a custom implementation for your array namespace "
            "using operations that are more efficient for your array namespace.",
            category=UserWarning,
            stacklevel=1,
        )
        for idx in range(indices.shape[0]):
            zeros[tuple(indices[idx, ...])] = 1

    return zeros


def _to_one_hot(array: Array, num_classes: Optional[int] = None) -> Array:
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
    if array.dtype not in _get_int_dtypes(namespace=xp):
        array = to_int(array)
    input_shape = array.shape
    array = flatten(array)

    if num_classes is None:
        unique_values = xp.unique_values(array)
        num_classes = int(apc.size(unique_values))

    device = apc.device(array)

    try:  # advanced indexing
        return xp.eye(num_classes, dtype=xp.int64, device=device)[array]
    except IndexError:
        n = array.shape[0]
        categorical = xp.zeros((n, num_classes), dtype=xp.int64, device=device)

        indices = xp.stack((xp.arange(n, device=device), array), axis=-1)
        for idx in range(indices.shape[0]):
            categorical[tuple(indices[idx, ...])] = 1
        output_shape = input_shape + (num_classes,)

        return xp.reshape(categorical, output_shape)
