"""RNN models."""
from typing import Optional

import torch
from torch import nn

from cyclops.models.catalog import register_model


@register_model("rnn", model_type="temporal")
class RNNModel(nn.Module):
    """RNN Class."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        layer_dim: int = 2,
        output_dim: int = 1,
        dropout_prob: float = 0.2,
        last_timestep_only: Optional[bool] = False,
    ) -> None:
        """Initialize model.

        Parameters
        ----------
        input_dim : int
            Number of input features
        hidden_dim : int, default=64
            Number of features in the hidden state.
        layer_dim : int, default=2
            Number of hidden layers.
        output_dim : int, default=1
            Dimension of output layer.
        dropout_prob : float, default=0.2
            Dropout probability for dropout layer.
        last_timestep_only : bool, default=False
            Whether to use only the last timestep of the output.

        """
        super().__init__()
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
            .to(x.device)
        )
        out, h0 = self.rnn(x, h0.detach())
        if self.last_timestep_only:
            out = out[:, -1, :]
        out = self.fc(out)
        return out
