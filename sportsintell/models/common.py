from __future__ import annotations

import torch
import torch.nn as nn

from sportsintell.utils.geometry import sinusoidal_position_encoding


class PositionalEncoding(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = sinusoidal_position_encoding(x.size(1), x.size(2), x.device)
        return x + pe.unsqueeze(0)


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
