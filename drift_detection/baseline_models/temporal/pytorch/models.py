"""Model library for baseline temporal models."""
import torch
from torch import nn


class RNNModel(nn.Module):
    """RNN Class.

    Attributes
    ----------
    input_dim: int
        Batch size
    hidden_dim: int
        Number of features in the hidden state
    layer_dim: int
        Number of hidden layers
    output_dim: int
        Dimension of output
    dropout_prob: int
        Dropout probability for dropout layer
    last_timestep_only: bool
        Whether to use only the last timestep of the output

    """

    def __init__(
        self,
        device,
        input_dim,
        hidden_dim,
        layer_dim,
        output_dim,
        dropout_prob,
        last_timestep_only=False,
    ):
        super().__init__()
        self.device = device
        self.last_timestep_only = last_timestep_only

        self.rnn = nn.RNN(
            input_dim,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=dropout_prob,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor

        Returns
        -------
        out: torch.Tensor
            Output tensor

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


class GRUModel(nn.Module):
    """GRU Class.

    Attributes
    ----------
    input_dim: int
        Batch size
    hidden_dim: int
        Number of features in the hidden state
    layer_dim: int
        Number of hidden layers
    output_dim: int
        Dimension of output
    dropout_prob: int
        Dropout probability for dropout layer

    """

    def __init__(
        self,
        device,
        input_dim,
        hidden_dim,
        layer_dim,
        output_dim,
        dropout_prob,
        last_timestep_only=False,
    ):
        super().__init__()
        self.device = device
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.last_timestep_only = last_timestep_only

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor

        Returns
        -------
        out: torch.Tensor
            Output tensor

        """
        h0 = (
            torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
            .requires_grad_()
            .to(self.device)
        )
        out, _ = self.gru(x, h0.detach())
        if self.last_timestep_only:
            out = out[:, -1, :]
        out = self.fc(out)
        return out


class LSTMModel(nn.Module):
    """LSTM Class.

    Attributes
    ----------
    input_dim: int
        Batch size
    hidden_dim: int
        Number of features in the hidden state
    layer_dim: int
        Number of hidden layers
    output_dim: int
        Dimension of output
    dropout_prob: int
        Dropout probability for dropout layer

    """

    def __init__(
        self,
        device,
        input_dim,
        hidden_dim,
        layer_dim,
        output_dim,
        dropout_prob,
        last_timestep_only=False,
    ):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.last_timestep_only = last_timestep_only

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor

        Returns
        -------
        out: torch.Tensor
            Output tensor

        """
        h0 = (
            torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
            .requires_grad_()
            .to(self.device)
        )
        c0 = (
            torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
            .requires_grad_()
            .to(self.device)
        )
        out, (_, _) = self.lstm(x, (h0.detach(), c0.detach()))
        if self.last_timestep_only:
            out = out[:, -1, :]
        out = self.fc(out)
        return out
