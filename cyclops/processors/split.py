"""datasetset splits."""

import numpy as np
from typing import Union, List, Tuple, Optional
from cyclops.utils.common import to_list, to_list_optional

def fractions_to_split(
    fractions: Union[float, List, Tuple],
    data_len: int,
) -> np.ndarray:
    if isinstance(fractions, float):
        fractions = to_list(fractions)
        
    if isinstance(fractions, list) or isinstance(fractions, tuple):
        is_float = [isinstance(elem, float) for elem in fractions]
        if not all(is_float):
            raise ValueError("fractions must be floats.")
        
        invalid = [frac <= 0 or frac >= 1 for frac in fractions]
        if any(invalid):
            raise ValueError("fractions must be between 0 and 1.")
        
        if sum(fractions) != 1:
            if sum(fractions) < 1:
                fractions.append(1 - sum(fractions))
            else:
                raise ValueError("fractions must sum to 1.")
    else:
        raise ValueError("fractions must be a float or a list of floats.")
    
    for i in range(1, len(fractions)):
        fractions[i] += fractions[i - 1]
    
    assert fractions[-1] == 1
    
    return np.round(np.array(fractions[:-1])*data_len).astype(int)


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
        Fraction of samples to use for train, test sets. If an int,
        split into 2. If a list, split into .........................................................
    randomize: bool, default = True
        Whether to randomize the data in the splits.
    seed: int, optional
        Seed for random number generator.

    Returns
    -------
    tuple
        A tuple of the split numpy.ndarray.

    """
    split = fractions_to_split(fractions, data_len)
    idx = np.arange(data_len)
    
    if seed is not None:
        np.random.seed(seed)
    if randomize:
        np.random.shuffle(idx)
    
    return tuple(np.split(idx, split))


def split_data(
    datasets: Union[np.ndarray, List[np.ndarray]],
    fractions: Union[float, List[float]],
    axes: Optional[Union[int, List[int]]] = None,
    randomize: bool = True,
    seed: int = None,
) -> Union[np.ndarray, Tuple]:
    """
    datasets: np.ndarray or list of np.ndarray
        Datasets, or a dataset, to split.
    axes: int or list of int
        Axes, or axis, along which to split the data.
    fractions: float or list of float
        Fraction(s) of samples between 0 and 1 to use for each split.
    """
    if isinstance(datasets, np.ndarray):
        datasets = [datasets]
    axes = to_list_optional(axes)
    
    if axes is None:
        axes = [0]*len(datasets)
    
    sizes = [np.size(data, axes[i]) for i, data in enumerate(datasets)]
    
    # Make sure sizes along the specified axes are all the same
    if not sizes.count(sizes[0]) == len(sizes):
        raise ValueError("datasets must have the same sizes along the given axes.")
    
    idx_splits = split_idx(fractions, sizes[0], randomize=randomize, seed=seed)
    
    splits = []
    # For each dataset
    for i, data in enumerate(datasets):
        splits.append([])
        # For each split
        for idx in idx_splits:
            shape = [1]*len(data.shape)
            shape[axes[i]] = len(idx)
            idx = idx.reshape(shape)
            print(data.shape)
            print(idx.shape)
            splits[-1].append(np.take_along_axis(data, idx, axis=axes[i]))
    
    if len(splits) == 1:
        return splits[0]
    
    return tuple(splits)