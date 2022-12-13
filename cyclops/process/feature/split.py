"""Dataset split processing."""

from typing import Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from cyclops.utils.common import to_list


def fractions_to_split(
    fractions: Union[float, List[float]],
    n_samples: int,
) -> np.ndarray:
    """Create an array of index split points useful for dataset splitting.

    Created using the length of the data and the desired split fractions.

    Parameters
    ----------
    fractions: float or list of float
        Fraction(s) of samples between 0 and 1 to use for each split.
    n_samples: int
        The total number of samples in the data being split.

    Returns
    -------
    np.ndarray
        Split indices to use in creating the desired split sizes.

    """
    frac_list: List[float]
    if isinstance(fractions, float):
        if fractions >= 1 or fractions <= 0:
            raise ValueError("As a float, fractions must be in the range (0, 1).")
        frac_list = [fractions]
    elif isinstance(fractions, list):
        # Necessary so as to not mutate the original fractions list
        frac_list = list(fractions)
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
            # Meant to handle floats e.g., 0.8 -> [0.8], which is actually [0.8, 0.2]
            # Doing it this way allows for directly entering [0.8] in list form
            frac_list.append(1.0 - sum(frac_list))
        else:
            raise ValueError("fractions must sum to 1.")

    # Turn into dividing list of lengths to split and return
    for i in range(1, len(frac_list)):
        frac_list[i] += frac_list[i - 1]

    assert frac_list[-1] == 1

    return np.round(np.array(frac_list[:-1]) * n_samples).astype(int)


def split_idx(
    fractions: Union[float, List[float]],
    n_samples: int,
    randomize: bool = True,
    seed: Optional[int] = None,
) -> tuple:
    """Create disjoint subsets of indices.

    Parameters
    ----------
    fractions: float or list of float
        Fraction(s) of samples between 0 and 1 to use for each split.
    n_samples: int
        The length of the data.
    randomize: bool, default = True
        Whether to randomize the data in the splits.
    seed: int, optional
        Seed for random number generator.

    Returns
    -------
    tuple of numpy.ndarray
        Disjoint subsets of indices.

    """
    split = fractions_to_split(fractions, n_samples)
    idx = np.arange(n_samples)

    # Optionally randomize
    if randomize:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    return tuple(np.split(idx, split))


def split_idx_stratified(
    fractions: Union[float, List[float]],
    stratify_labels: np.ndarray,
    randomize: bool = True,
    seed: Optional[int] = None,
) -> tuple:
    """Create disjoint, label-stratified subsets of indices.

    There will be the equal label proportions in each subset.

    Parameters
    ----------
    fractions: float or list of float
        Fraction(s) of samples between 0 and 1 to use for each split.
    stratify_labels: numpy.ndarray
        1D array of labels used for stratification.
    randomize: bool, default = True
        Whether to randomize the data in the splits.
    seed: int, optional
        Seed for random number generator.

    Returns
    -------
    tuple of numpy.ndarray
        Disjoint, label-stratified subsets of indices.

    """
    assert stratify_labels.ndim == 1

    # Stratify by label values
    series = pd.Series(stratify_labels)
    groups = series.groupby(series)
    stratified_idx = groups.apply(
        lambda group: [
            group.index.values[idx]
            for idx in split_idx(fractions, len(group), randomize=False)
        ]
    )

    # Combine stratified into subsets
    n_subsets = len(stratified_idx.iloc[0])
    idxs = []
    for subset_i in range(n_subsets):
        idxs.append(np.concatenate([ind[subset_i] for ind in stratified_idx.values]))

    # Optionally randomize
    if randomize:
        rng = np.random.default_rng(seed)
        for idx in idxs:
            rng.shuffle(idx)

    return tuple(idxs)


def split_kfold(
    k_folds: int,
    n_samples: int,
    randomize: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Create K disjoint subsets of indices equal in length.

    These K equally sized folds are useful for K-fold cross validation.

    Parameters
    ----------
    k_folds: int
        K, i.e., the number of folds.
    n_samples: int
        The number of samples.
    randomize: bool, default = True
        Whether to randomize the data in the splits.
    seed: int, optional
        Seed for random number generator.

    Returns
    -------
    tuple of numpy.ndarray
        K disjoint subsets of indices equal in length.

    """
    fracs = [1 / k_folds for i in range(k_folds - 1)]
    idxs = split_idx(fracs, n_samples, randomize=randomize, seed=seed)
    return idxs


def idxs_to_splits(
    samples: np.ndarray,
    idxs: Tuple,
):
    """Create data subsets using subsets of indices.

    Parameters
    ----------
    samples: numpy.ndarray
        A NumPy array with the first dimension being over the samples.
    idxs: tuple of numpy.ndarray
        Subsets of indices.

    Returns
    -------
    tuple of numpy.ndarray
        Dataset splits.

    """
    return tuple(samples[idx] for idx in idxs)


def kfold_cross_val(
    k_folds: int,
    samples: np.ndarray,
    randomize: bool = True,
    seed: Optional[int] = None,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Perform K-fold cross validation.

    Parameters
    ----------
    k_folds: int
        Number of folds in the K-fold cross validation.
    samples: numpy.ndarray
        A NumPy array with the first dimension being over the samples.
    randomize: bool, default = True
        Whether to randomize the data in the splits.
    seed: int, optional
        Seed for random number generator.

    Yields
    ------
    tuple of numpy.ndarray
        Yields the training and validation splits.

    """
    idxs = split_kfold(k_folds, len(samples), randomize=randomize, seed=seed)
    folds = idxs_to_splits(samples, idxs)

    for fold in range(k_folds):
        val_fold = folds[fold]
        train_fold = np.concatenate([f for i, f in enumerate(folds) if i != fold])
        yield train_fold, val_fold


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
    seed: Optional[int] = None,
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
        Seed for random number generator.

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
