"""Utility functions for building models."""

import typing

import torch
import torch.nn as nn


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


def get_activation_fn(act_str: str) -> typing.Union[torch.nn.modules.activation, None]:
    """Get activation function.

    Parameters
    ----------
    act_str: str
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
    if act_str in ACT_FN_MAP:
        return ACT_FN_MAP[act_str]
    else:
        raise ValueError("[!] Invalid activation function.")
