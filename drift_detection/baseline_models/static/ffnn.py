import torch.nn as nn


class FFNNetModel(nn.Module):
    def __init__(self, device, input_dim, output_dim):
        super(FFNNetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, output_dim)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.silu(self.fc1(x))
        x = self.silu(self.fc2(x))
        x = self.fc3(x)
        return x
