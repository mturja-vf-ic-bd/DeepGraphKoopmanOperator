import torch
import torch.nn as nn
import math


class PositionalEncoding3D(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding3D, self).__init__()
        self.register_buffer('positional_encoding', self._get_positional_encoding(max_seq_len, d_model))

    def _get_positional_encoding(self, max_seq_len, d_model):
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        odd = False
        if d_model % 2:
            d_model += 1
            odd = True
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add dimensions for batch and feature dimensions
        if odd:
            return pe[:, :, :-1]
        return pe

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.positional_encoding[:, :x.size(1), :]
        return x
