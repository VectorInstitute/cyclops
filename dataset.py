"""File for data pipelines which are then put into torch dataset format
"""
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from torch.utils.data import random_split

import datapipeline.config as conf
from datapipeline.process_data import pipeline, get_splits, prune_columns

from registry import register_with_dictionary

DATA_REGISTRY = {}
register = partial(register_with_dictionary, DATA_REGISTRY)

@dataclass
class BaseData:
    inputs: torch.Tensor
    target: torch.Tensor

    def __getitem__(self, idx):
        return self.inputs[idx], self.target[idx]

    def __len__(self):
        return len(self.target)

    def dim(self):
        return self.inputs.size(dim=1)

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

def pandas_to_dataset(df, feature_cols, target_cols):
    features = prune_columns(feature_cols, df)
    inputs = torch.tensor(df[features].values, dtype=torch.float32)
    target = torch.tensor(df[target_cols].values, dtype=torch.float32)
    target = torch.flatten(target)
    return BaseData(inputs, target)

@register
def fakedata(args):
    inputs = torch.randn(args.data_len, args.data_dim, dtype=torch.float32)
    target = (inputs.sum(1) > 0).long()
    
    dataset = BaseData(inputs, target)
    return split_train_and_val(dataset)

@register
def gemini(args):
    # get data pipeline configuration
    config = conf.read_config(args.dataset_config)
    data, _ = pipeline(config)
    train, val, _ = get_splits(config, data)

    train_dset = pandas_to_dataset(train, config.features, config.target)
    val_dset = pandas_to_dataset(val, config.features, config.target)

    # return train and split for now to be consistent with fake data
    # TODO: change later
    return train_dset, val_dset
    

