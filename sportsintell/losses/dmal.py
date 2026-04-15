from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


def _corners(boxes: torch.Tensor) -> dict[str, torch.Tensor]:
    x, y, w, h = boxes.unbind(dim=-1)
    x1, y1 = x - w / 2.0, y - h / 2.0
    x2, y2 = x + w / 2.0, y + h / 2.0
    return {
        "c": torch.stack([x, y], dim=-1),
        "lt": torch.stack([x1, y1], dim=-1),
        "rt": torch.stack([x2, y1], dim=-1),
        "lb": torch.stack([x1, y2], dim=-1),
        "rb": torch.stack([x2, y2], dim=-1),
    }


def _angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    delta = b - a
    return torch.atan2(delta[..., 1], delta[..., 0])


def _wrapped_abs_angle_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    delta = a - b
    return torch.abs(torch.atan2(torch.sin(delta), torch.cos(delta)))


class DirectionalMotionAlignmentLoss(nn.Module):
    def forward(self, prev_boxes: torch.Tensor, pred_boxes: torch.Tensor, true_boxes: torch.Tensor) -> torch.Tensor:
        prev_pts = _corners(prev_boxes)
        pred_pts = _corners(pred_boxes)
        true_pts = _corners(true_boxes)
        losses = []
        for key in ("c", "lt", "rt", "lb", "rb"):
            pred_theta = _angle(prev_pts[key], pred_pts[key])
            true_theta = _angle(prev_pts[key], true_pts[key])
            losses.append(_wrapped_abs_angle_diff(pred_theta, true_theta))
        return torch.stack(losses, dim=0).mean()
