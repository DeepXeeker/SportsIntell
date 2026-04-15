from __future__ import annotations

import torch
import torch.nn as nn


class OffsetL1Loss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.abs(pred - target).mean()
