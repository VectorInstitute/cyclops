"""RNN models."""

from typing import Optional

import torch
from torch import nn

# pylint: disable=invalid-name, too-many-arguments


class RNNModel(nn.Module):
    """RNN Class."""

    def __init__(
        self,
        device: torch.device,
        input_dim: int,
        hidden_dim: int,
        layer_dim: int,
        output_dim: int,
        dropout_prob: float,
        last_timestep_only: Optional[bool] = False,
    ) -> None:
        """Initialize model.

        Parameters
        ----------
        device : torch.device
            The device to allocate
        input_dim : int
            Number of input features
        hidden_dim : int
            Number of features in the hidden state
        layer_dim : int
            Number of hidden layers
        output_dim : int
            Dimension of output
        dropout_prob : float
            Dropout probability for dropout layer
        last_timestep_only : Optional[bool], optional
            Keep the last timestep, by default False

        """
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.last_timestep_only = last_timestep_only

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input to the model

        Returns
        -------
        torch.Tensor
            Model output

        """
        h0 = (
            torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
            .requires_grad_()
            .to(self.device)
        )
        out, h0 = self.rnn(x, h0.detach())
        if self.last_timestep_only:
            out = out[:, -1, :]
        out = self.fc(out)
        return out
