from __future__ import annotations

import torch
import torch.nn as nn

from .attention import TemporalTransformer
from .common import MLPHead


class NoMotionPredictor(nn.Module):
    def forward(self, history: torch.Tensor) -> torch.Tensor:
        batch = history.size(0)
        return torch.zeros(batch, 4, device=history.device, dtype=history.dtype)


class VanillaTransformerPredictor(nn.Module):
    def __init__(self, input_dim: int = 8, embed_dim: int = 32, layers: int = 6, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.ReLU(inplace=True))
        self.transformer = TemporalTransformer(embed_dim=embed_dim, num_layers=layers, num_heads=heads, dropout=dropout)
        self.head = MLPHead(embed_dim, embed_dim, 4)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        x = self.embed(history)
        x = self.transformer(x)
        return self.head(x[:, -1])
