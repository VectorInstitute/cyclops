import torch
import torch.nn as nn

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

    """
    def __init__(self, device, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(RNNModel, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        out, h0 = self.rnn(x, h0.detach())
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
    def __init__(self, device, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(GRUModel, self).__init__()
        self.device = device
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        self.fc = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        out, _ = self.gru(x, h0.detach())
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
    def __init__(self, device, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        self.fc = nn.Linear(hidden_dim, output_dim)       
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = out[:, -1, :]
        out = self.sigmoid(self.fc(out))
        return out
    
class LSTMCellModel(nn.Module):
    def __init__(self, device, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTMCellModel, self).__init__()
        self.device = device
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTMCell(input_size=input_dim, hidden_size=hidden_dim)
            
        self.dropout = nn.Dropout(p=dropout_prob)
        self.relu = nn.ReLU()
        self.hidden2tag = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, rolling=False, hn=None, cn=None):
        rolling=0

        outputs = []
        for i in range(x.size(1)):
            if (i==0)&(hn is None):
                hn, cn = self.lstm(x[:, i])
            else:
                hn, cn = self.lstm(x[:, i], (hn, cn))
                # print(torch.isnan(hn).sum().sum().sum())
            out = self.relu(hn)
            out = self.dropout(out)
            out = self.hidden2tag(out)
            outputs.append(out)

        if rolling:
          return torch.cat(outputs, dim=1)
        else:
          return outputs[-1]
