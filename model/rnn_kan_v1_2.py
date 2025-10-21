import torch
from torch import nn
from layer.kan_layer import KANLinear

class RNN_KAN_Cell(nn.Module):
    """
        x: (batch, in_features)
        h0: (batch, hidden_features)
    """
    def __init__(self, in_features, hidden_features, activation=nn.Tanh):
        super(RNN_KAN_Cell, self).__init__()
        assert in_features[-1] == hidden_features, f"in_features[-1]={in_features[-1]} phải bằng hidden_features={hidden_features}"
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.activation = activation()
        self.i2h = KANLinear(in_features[0], in_features[1])
        self.h2h = nn.Linear(hidden_features, hidden_features) # W_hh @ h + b_h

    def forward(self, x, h):
        return self.activation(self.i2h(x) + self.h2h(h))

class RNN_KAN(nn.Module):
    """
        x: (batch, seq_len, in_features)
        h0: (batch, hidden_features)
        in_features: list - last idx must equal to hidden_features
    """
    def __init__(self, in_features, hidden_features, output_features, n_ahead, activation=nn.Tanh):
        super(RNN_KAN, self).__init__()
        self.hidden_features = hidden_features
        self.in_features = in_features
        self.RNN_KAN_Cell = RNN_KAN_Cell(in_features, hidden_features, activation)
        self.fc_out = nn.Linear(hidden_features, output_features)

        self.n_ahead = n_ahead
        self.output_features = output_features
    def forward(self, x, h0=None):
        batch, Tx, _ = x.size()
        Ty = self.n_ahead
        h = torch.zeros(batch, Tx + Ty, self.hidden_features, device=x.device)
        y_pred = torch.zeros(batch, Ty, self.output_features, device=x.device)
        if h0 is None:
            h0 = torch.zeros(batch, self.hidden_features, device=x.device)

        h_t = h0
        for t in range(Tx):
            h_t = self.RNN_KAN_Cell(x[:, t, :], h_t)
            h[:, t, :] = h_t
        for t in range(Ty):
            h_t = self.RNN_KAN_Cell(torch.zeros(batch, self.in_features[0], device=x.device), h_t)
            h[:, Tx + t, :] = h_t
            y_t = self.fc_out(h_t)
            y_pred[:, t, :] = y_t
        return y_pred.squeeze(dim=2), h

