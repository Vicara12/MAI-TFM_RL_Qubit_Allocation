import math

import torch


class PositionalEncoder(torch.nn.Module):
    """ "
    Positional encoder for transformer models.
    This module is used to add positional encodings to the input of the model:
    x = x + pe[:, :x.shape[1]]
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoder, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x, add=True):
        """
        Args:
            x: Input tensor of shape [batch, seq_len, ...] or [batch, num_slices, num_qubits, hidden_dim]
            add: Whether to add PE to input or return PE only
        """
        # This check is for QA
        if x.dim() == 4:  # [batch, num_slices, num_qubits, hidden_dim]
            seq_len = x.shape[1]  # num_slices
            pe = self.pe[:, :seq_len].unsqueeze(2)  # [1, seq_len, 1, hidden_dim]
            return x + pe if add else pe
        else:  # Original behavior for other cases
            return x + self.pe[:, : x.shape[1]] if add else self.pe[:, : x.shape[1]]