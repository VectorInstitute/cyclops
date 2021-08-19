"""File for data pipelines which are then put into torch dataset format
"""
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from torch.utils.data import random_split

from registry import register_with_dictionary

DATA_REGISTRY = {}
register = partial(register_with_dictionary, DATA_REGISTRY)


def get_dataset(name):
    return DATA_REGISTRY[name]


def split_train_and_val(dataset, percent_val=0.2, seed=42):
    """Takes a pytorch dataset and split it into train and validation set
    """
    len_dset = len(dataset)

    validation_length = int(len_dset * percent_val)
    train_length = len_dset - validation_length

    train_dset, val_dset = random_split(
        dataset, (train_length, validation_length),
        generator=torch.Generator(device="cpu").manual_seed(seed))

    return train_dset, val_dset


@register
def fakedata(args):
    inputs = torch.randn(args.data_len, args.data_dim, dtype=torch.float32)
    target = (inputs.sum(1) > 0).long()

    @dataclass
    class FakeData:
        inputs: np.ndarray
        target: np.ndarray

        def __getitem__(self, idx):
            return self.inputs[idx], self.target[idx]

        def __len__(self):
            return len(self.target)

    return FakeData(inputs, target)
