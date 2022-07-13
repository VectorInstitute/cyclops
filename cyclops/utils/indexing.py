"""Utility functions for indexing NumPy arrays."""

from typing import List, Optional, Tuple, Union

import numpy as np


def index_axis(ind: int, axis: int, shape: Tuple) -> Tuple:
    """Index one value over one axis and fetch everything from all other axes.

    E.g., for ind = 3, axis = 1, and shape = (10, 20, 30), this function is
    equivalent to indexing array[:, 3, :].

    Parameters
    ----------
    ind: int
        The index in the specified axis.
    axis: int
        The axis.
    shape: tuple
        The shape of the data being indexed.

    Returns
    -------
    tuple
        A tuple which can be used to index the array.

    """
    index: List[Union[slice, int]] = [slice(None)] * len(shape)
    index[axis] = ind
    return tuple(index)


def take_indices(
    data: np.ndarray,
    indexes: List[Optional[List[int]]],
) -> np.ndarray:
    """Index array by specifying the indices to take on each axis.

    Parameters
    ----------
    data: numpy.ndarray
        Data to index.
    indexes
        E.g., ([None, [1, 2, 3], None, [20]]), where each element can be
        None, a list, or a numpy.ndarray. If None, take all on the axis.

    Returns
    -------
    numpy.ndarray
        Indexed data.

    """
    for i, index in enumerate(indexes):
        if index is None:
            continue

        # Reshape idx to have same number of dimensions as the data
        np_index = np.array(index)
        shape = [1] * len(data.shape)
        shape[i] = len(np_index)
        data = np.take_along_axis(data, np_index.reshape(shape), axis=i)

    return data
