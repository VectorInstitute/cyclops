"""MLP models."""

import math

import torch
from torch import nn

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

    def __init__(  # pylint: disable=R0913
        self,
        num_layers: int,
        in_dim: int,
        hidden_dims: list,
        out_dim: int,
        activation: str = "relu",
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
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(module.bias, -bound, bound)

    def _layer(self, in_dim, out_dim, activation=None):
        if activation:
            layer = [
                nn.Linear(in_dim, out_dim),
                self.activation,
            ]
        else:
            layer = [
                nn.Linear(in_dim, out_dim),
            ]
        return layer

    def forward(self, input_: torch.Tensor):
        """Forward pass.

        Parameters
        ----------
        input_: torch.Tensor
            Input to the model.

        Returns
        -------
        torch.Tensor
            Model output.
        """
        return self.model(input_)
