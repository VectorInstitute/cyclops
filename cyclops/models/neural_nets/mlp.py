"""MLP models."""
import math
from typing import List, Union

import torch
from torch import nn

from cyclops.models.catalog import register_model
from cyclops.models.utils import get_module


@register_model("mlp_pt", model_type="static")
class MLPModel(nn.Module):
    """A Multi-Layer Perceptron (MLP).

    Also known as a Fully-Connected Network (FCN). This implementation assumes that all
    hidden layers have the same hidden size and the same activation function.

    """

    def __init__(  # pylint: disable=dangerous-default-value
        self,
        input_dim: int,
        hidden_dims: List = [64, 64],
        output_dim: int = 1,
        activation: Union[str, nn.Module] = "ReLU",
    ) -> None:
        """Instantiate model.

        Parameters
        ----------
        input_dim : int
            Number of input features to the model.
        hidden_dims : list, default=[64, 64]
            A list of dimensions for each hidden layer.
        output_dim : int, default=1
            Dimension of the output.
        activation : str or torch.nn.modules.activation, default="ReLU"
            Activation function to use.

        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = get_module("activation", activation)

        layers = [self._layer(input_dim, hidden_dims[0], self.activation)]
        for i in range(len(hidden_dims) - 1):
            layers.extend(
                self._layer(
                    self.hidden_dims[i] if i > 0 else input_dim,
                    self.hidden_dims[i + 1],
                    activation,
                )
            )
        layers.extend(self._layer(self.hidden_dims[-1], output_dim, activation=None))

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
