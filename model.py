from functools import partial

import torch
import torch.nn as nn

from registry import register_with_dictionary

MODEL_REGISTRY = {}
register = partial(register_with_dictionary, MODEL_REGISTRY)


def get_model(name):
    return MODEL_REGISTRY[name]
