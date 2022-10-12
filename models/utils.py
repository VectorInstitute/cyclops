"""Utility functions for building models."""

import typing

import torch
from torch import nn

ACT_FN_MAP = {
    "none": None,
    "hardtanh": nn.Hardtanh(),
    "sigmoid": nn.Sigmoid(),
    "relu6": nn.ReLU6(),
    "tanh": nn.Tanh(),
    "tanhshrink": nn.Tanhshrink(),
    "hardshrink": nn.Hardshrink(),
    "leakyreluleakyrelu": nn.LeakyReLU(),
    "softshrink": nn.Softshrink(),
    "relu": nn.ReLU(),
    "prelu": nn.PReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "selu": nn.SELU(),
    "silu": nn.SiLU(),
}


def get_activation_fn(act: str) -> typing.Union[torch.nn.Module, None]:
    """Get activation function.

    Parameters
    ----------
    act: str
        String specifying activation function.

    Returns
    -------
    torch.nn.modules.activation or None
        Activation function module.

    Raises
    ------
    ValueError
        If the input activation string doesn't match supported ones.

    """
    if act not in ACT_FN_MAP:
        raise ValueError("[!] Invalid activation function.")
    return ACT_FN_MAP[act]
