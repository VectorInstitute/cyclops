"""MLP models."""

import math
from typing import Optional

import torch
import torch.nn as nn

from models.utils import get_activation_fn


class MLP(nn.Module):
    """A Multi-Layer Perceptron (MLP).

    Also known as a Fully-Connected Network (FCN). This
    implementation assumes that all hidden layers have
    the same hidden size and the same activation function.

    Attributes
    ----------
        num_layers: int
            Number of layers in the network.
        in_dim: int
            Size of the input sample.
        hidden_dims: list
            Sizes of the hidden layers.
        out_dim: int
            Size of the output.
        activation: str
            Activation function.
    """

    def __init__(
        self,
        num_layers: int,
        in_dim: int,
        hidden_dims: list,
        out_dim: int,
        activation: Optional[str] = "relu",
    ):
        """Instantiate model.

        Parameters
        ----------
        num_layers: int
            Number of layers in the network.
        in_dim: int
            Size of the input sample.
        hidden_dims: list
            Sizes of the hidden layers.
        out_dim: int
            Size of the output.
        activation: str, optional
            Activation function.
        """
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.activation = get_activation_fn(activation)

        layers = []
        for i in range(self.num_layers - 1):
            layers.extend(
                self._layer(
                    self.hidden_dims[i] if i > 0 else in_dim,
                    self.hidden_dims[i + 1],
                    activation,
                )
            )
        layers.extend(self._layer(hidden_dims[i + 1], out_dim))

        self.model = nn.Sequential(*layers)

        # Initialize weights.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)

    def _layer(self, in_dim, out_dim, activation=None):
        if activation:
            return [
                nn.Linear(in_dim, out_dim),
                self.activation,
            ]
        else:
            return [
                nn.Linear(in_dim, out_dim),
            ]

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input to the model.

        Returns
        -------
        torch.Tensor
            Model output.
        """
        return self.model(x)
