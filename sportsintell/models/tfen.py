from __future__ import annotations

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)


class TFENBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = x.transpose(1, 2)
        y = self.conv1(y).transpose(1, 2)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.conv2(y.transpose(1, 2)).transpose(1, 2)
        y = self.activation(y)
        y = self.dropout(y)
        return self.norm(residual + y)


class TFEN(nn.Module):
    def __init__(
        self,
        channels: int = 32,
        blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        dilations: list[int] | None = None,
    ) -> None:
        super().__init__()
        if dilations is None:
            dilations = [2 ** i for i in range(blocks)]
        assert len(dilations) == blocks, "Number of dilations must equal blocks."
        self.blocks = nn.ModuleList(
            TFENBlock(channels=channels, kernel_size=kernel_size, dilation=d, dropout=dropout)
            for d in dilations
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x
