from functools import partial

import torch
import torch.nn as nn

from tasks.registry import register_with_dictionary

MODEL_REGISTRY = {}
register = partial(register_with_dictionary, MODEL_REGISTRY)


def get_model(name):
    return MODEL_REGISTRY[name]


@register
def mlp(args):
    # assume its a binary classification problem for now
    data_dim = args.data_dim

    return nn.Sequential(
        nn.Linear(data_dim, 100),
        nn.SiLU(),
        nn.Linear(100, 150),
        nn.SiLU(),
        nn.Linear(150, 1),
    )
