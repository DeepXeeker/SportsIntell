from __future__ import annotations

import torch
import torch.nn as nn

from .common import PositionalEncoding


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 32,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.positional_encoding = PositionalEncoding()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.positional_encoding(x)
        return self.encoder(x)
