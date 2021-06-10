import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(1000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + torch.Tensor(self.pe[:, :x.size(1)], 
                            requires_grad=False)
        return self.dropout(x)

class PositionalEncoding2D(nn.Module):
    "Implement the 2D PE function."
    def __init__(self, d_model, dropout, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        orig_pe = PositionalEncoding(d_model, 0, max_len=max_len).state_dict()['pe']
        horiz_pe = orig_pe.repeat(max_len, 1) 
        vert_pe = pe.unsqueeze(1).repeat(max_len)
        pe = horiz_pe + vert_pe
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + torch.Tensor(self.pe[:, :, :x.size(1)], 
                            requires_grad=False)
        return self.dropout(x)