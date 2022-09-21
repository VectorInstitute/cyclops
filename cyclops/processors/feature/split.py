"""Dataset split processing."""

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from cyclops.utils.common import to_list


def intersect_datasets(
    datas: List[pd.DataFrame],
    on_col: str,
    sort: bool = True,
) -> Tuple:
    """Perform an intersection across datasets over a column.

    This can be used to align dataset samples e.g., aligning encounters for a tabular
    and temporal dataset.

    Parameters
    ----------
    datas: list of pandas.DataFrame
        List of datasets.
    on_col: str
        The column on which to perform the intersection.
    sort: bool, default = True
        Whether to sort the values in each dataset by the on column.

    Returns
    -------
    tuple
        A tuple of the processed datasets.

    """
    # Concatenate the unique values in each dataset and count how many of each
    unique, counts = np.unique(
        np.concatenate([data[on_col].unique() for data in datas]), return_counts=True
    )

    # If a count is equal to the length of datasets, it must exist in every dataset
    intersect = unique[counts == len(datas)]

    # Intersect on these unique values
    for i, data in enumerate(datas):
        data = data[data[on_col].isin(intersect)]
        if sort:
            data = data.sort_values(on_col)
        datas[i] = data

    return tuple(datas)


def fractions_to_split(
    fractions: Union[float, List[float]],
    data_len: int,
) -> np.ndarray:
    """Turn a number of fractions into a dividing list of lengths at which to split.

    Parameters
    ----------
    fractions: float or list of float
        Fraction(s) of samples between 0 and 1 to use for each split.
    data_len: int
        The total number of samples in the data being split.

    """
    if isinstance(fractions, float):
        frac_list = to_list(fractions)
    elif isinstance(fractions, list):
        frac_list = fractions
    else:
        raise ValueError("fractions must be a float or a list of floats.")

    # Element checking
    is_float = [isinstance(elem, float) for elem in frac_list]
    if not all(is_float):
        raise ValueError("fractions must be floats.")

    invalid = [frac <= 0 or frac >= 1 for frac in frac_list]
    if any(invalid):
        raise ValueError("fractions must be between 0 and 1.")

    if sum(frac_list) != 1:
        if sum(frac_list) < 1:
            frac_list.append(1 - sum(frac_list))
        else:
            raise ValueError("fractions must sum to 1.")

    # Turn into dividing list of lengths to split and return
    for i in range(1, len(frac_list)):
        frac_list[i] += frac_list[i - 1]

    assert frac_list[-1] == 1

    return np.round(np.array(frac_list[:-1]) * data_len).astype(int)


def split_idx(
    fractions: Union[float, List[float]],
    data_len: int,
    randomize: bool = True,
    seed: int = None,
) -> tuple:
    """Split encounters into train/test.

    Parameters
    ----------
    data: numpy.ndarray
        Data.
    fractions: list, optional
        Fraction(s) of samples between 0 and 1 to use for each split.
    randomize: bool, default = True
        Whether to randomize the data in the splits.
    seed: int, optional
        Seed for random number generator.

    Returns
    -------
    tuple of numpy.ndarray
        Splits with indices of each split.

    """
    split = fractions_to_split(fractions, data_len)
    idx = np.arange(data_len)

    if seed is not None:
        np.random.seed(seed)
    if randomize:
        np.random.shuffle(idx)

    return tuple(np.split(idx, split))


def split_datasets_by_idx(
    datasets: Union[np.ndarray, List[np.ndarray]],
    idx_splits: Tuple,
    axes: Optional[Union[int, List[int]]] = None,
):
    """Split datasets by index over given axes.

    Parameters
    ----------
    datasets: numpy.ndarray or list of numpy.ndarray
        Datasets to split in the same manner.
    idx_splits: tuple
        A tuple of the indices belonging to each individual split.
    axes: int or list of int, optional
        The axes along which to split each of the datasets.
        If not specified, defaults to the axis = 0 for all datasets.

    Returns
    -------
    tuple
        A tuple of the dataset splits, where each contains a tuple of splits.
        e.g., split1, split2 = split_features([features1, features2], 0.5)
        train1, test1 = split1
        train2, test2 = split2

    """
    if isinstance(datasets, np.ndarray):
        datasets = [datasets]

    if axes is None:
        axes_list = [0] * len(datasets)
    else:
        axes_list = to_list(axes)

    splits = []  # type: ignore
    # For each dataset
    for i, data in enumerate(datasets):
        splits.append([])
        # For each split
        for idx in idx_splits:
            # Reshape idx to have same number of dimensions as the data
            shape = [1] * len(data.shape)
            shape[axes_list[i]] = len(idx)
            idx = idx.reshape(shape)

            # Sample new dataset split
            splits[-1].append(np.take_along_axis(data, idx, axis=axes_list[i]))
        splits[-1] = tuple(splits[-1])

    if len(splits) == 1:
        return splits[0]

    return tuple(splits)


def split_datasets(
    datasets: Union[np.ndarray, List[np.ndarray]],
    fractions: Union[float, List[float]],
    axes: Optional[Union[int, List[int]]] = None,
    randomize: bool = True,
    seed: int = None,
) -> Tuple:
    """Split a dataset into a number of datasets.

    Parameters
    ----------
    datasets: np.ndarray or list of np.ndarray
        Datasets, or a dataset, to split.
    axes: int or list of int
        Axes, or axis, along which to split the data.
    fractions: float or list of float
        Fraction(s) of samples between 0 and 1 to use for each split.
    randomize: bool, default = True
        Whether to randomize the samples in the splits. Otherwise it splits
        the samples in the current order.
    seed: int, optional
        A seed for the randomization.

    Returns
    -------
    tuple
        A tuple of splits if a single dataset is given. Otherwise, a tuple of
        datasets of splits. All splits are also numpy.ndarray.

    """
    if isinstance(datasets, np.ndarray):
        datasets = [datasets]

    if axes is None:
        axes_list = [0] * len(datasets)
    else:
        axes_list = to_list(axes)

    sizes = [np.size(data, axes_list[i]) for i, data in enumerate(datasets)]

    # Make sure sizes along the specified axes are all the same
    if not sizes.count(sizes[0]) == len(sizes):
        raise ValueError("datasets must have the same sizes along the given axes.")

    idx_splits = split_idx(fractions, sizes[0], randomize=randomize, seed=seed)

    return split_datasets_by_idx(datasets, idx_splits)
