"""Dataset creation for training/inference using PyTorch."""

from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import pandas as pd
import torch
from torch.utils.data import random_split


def register_dataset(
    register_dict: dict, func: Optional[Callable] = None, name: Optional[str] = None
):
    """Register datasets, returned by functions.

    Parameters
    ----------
    register_dict: dict
        Dictionary to track dataset implementations.
    func: Callable, optional
        Function that returns data subset(s).
    name: str, optional
        Name of dataset.

    Returns
    -------
    Callable:
        Function call that to register dataset, that can be used as a decorator.

    """

    def wrap(func):
        register_dict[func.__name__ if name is None else name] = func
        return func

    # called with params
    if func is None:
        return wrap

    return wrap(func)


DATA_REGISTRY: dict = {}
register = partial(register_dataset, DATA_REGISTRY)


@dataclass
class BaseData:
    """Base dataset class."""

    inputs: torch.Tensor
    target: torch.Tensor

    def __getitem__(self, idx: int) -> tuple:
        """Get item for iterator.

        Parameters
        ----------
        idx: int
            Index of sample to fetch from dataset.

        Returns
        -------
        tuple
            Input and target.

        """
        return self.inputs[idx], self.target[idx]

    def __len__(self) -> int:
        """Return size of dataset, i.e. no. of samples.

        Returns
        -------
        int
            Size of dataset.

        """
        return len(self.target)

    def dim(self) -> int:
        """Get dataset dimensions (no. of features).

        Returns
        -------
        int
            Number of features.

        """
        return self.inputs.size(dim=1)


def get_dataset(name):
    """Get dataset from registered datasets.

    Parameters
    ----------
    name: str
        Name of dataset.

    Returns
    -------
    Callable
        Function that can be called to return dataset(s).

    """
    return DATA_REGISTRY[name]


def split_train_and_val(dataset, percent_val=0.2, seed=42):
    """Split PyTorch dataset into train and validation set."""
    len_dset = len(dataset)

    validation_length = int(len_dset * percent_val)
    train_length = len_dset - validation_length

    train_dset, val_dset = random_split(
        dataset,
        (train_length, validation_length),
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )

    return train_dset, val_dset


def pandas_to_dataset(
    dataframe: pd.DataFrame,
    feature_cols: list,
    target_cols: list,
    stats: dict = None,
    config=None,
):
    """Convert pandas dataframe to dataset.

    Parameters
    ----------
    dataframe: pandas.DataFrame
        Dataset as a pandas dataframe.
    feature_cols: list
        List of feature columns to consider.

    """
    if stats is not None:
        dataframe[config.numerical_features] = (
            dataframe[config.numerical_features] - stats["means"]
        ) / stats["std"]
    inputs = torch.tensor(dataframe[feature_cols].values, dtype=torch.float32)
    target = torch.tensor(dataframe[target_cols].values, dtype=torch.float32)
    target = torch.flatten(target)
    return BaseData(inputs, target)


@register
def fakedata(args):
    """Fake data."""
    inputs = torch.randn(args.data_len, args.data_dim, dtype=torch.float32)
    target = (inputs.sum(1) > 0).long()

    dataset = BaseData(inputs, target)
    return split_train_and_val(dataset)


@register
def gemini():
    """GEMINI data."""
