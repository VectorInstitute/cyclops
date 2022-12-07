"""FFNN model for static baseline model."""
from torch import nn


class FFNNetModel(nn.Module):
    """FFNN model for static baseline model."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, output_dim)
        self.silu = nn.SiLU()

    def forward(self, x):
        """Forward pass."""
        x = self.silu(self.fc1(x))
        x = self.silu(self.fc2(x))
        x = self.fc3(x)
        return x
