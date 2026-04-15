from __future__ import annotations

import torch
import torch.nn as nn

from .attention import TemporalTransformer
from .common import MLPHead
from .tfen import TFEN


class SportsIntell(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        embed_dim: int = 32,
        tfen_blocks: int = 4,
        transformer_layers: int = 6,
        attention_heads: int = 8,
        dropout: float = 0.1,
        use_tfen: bool = True,
    ) -> None:
        super().__init__()
        self.history_embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(inplace=True),
        )
        self.use_tfen = use_tfen
        self.tfen = TFEN(channels=embed_dim, blocks=tfen_blocks, dropout=dropout) if use_tfen else nn.Identity()
        self.transformer = TemporalTransformer(
            embed_dim=embed_dim,
            num_layers=transformer_layers,
            num_heads=attention_heads,
            dropout=dropout,
        )
        self.motion_head = MLPHead(embed_dim, embed_dim, 4)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        x = self.history_embed(history)
        x = self.tfen(x)
        x = self.transformer(x)
        return self.motion_head(x[:, -1])

    def predict_boxes(self, history: torch.Tensor, prev_boxes: torch.Tensor) -> torch.Tensor:
        offsets = self.forward(history)
        return prev_boxes + offsets
