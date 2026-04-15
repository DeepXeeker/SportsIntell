from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


@dataclass
class BBox:
    x: float
    y: float
    w: float
    h: float

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (
            self.x - self.w / 2.0,
            self.y - self.h / 2.0,
            self.x + self.w / 2.0,
            self.y + self.h / 2.0,
        )

    def corners(self) -> dict[str, tuple[float, float]]:
        x1, y1, x2, y2 = self.as_xyxy()
        return {
            "c": (self.x, self.y),
            "lt": (x1, y1),
            "rt": (x2, y1),
            "lb": (x1, y2),
            "rb": (x2, y2),
        }


def bbox_from_array(values: Iterable[float]) -> BBox:
    x, y, w, h = values
    return BBox(float(x), float(y), float(w), float(h))


def iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def pairwise_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    out = np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            out[i, j] = iou_xywh(a, b)
    return out


def sinusoidal_position_encoding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-np.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
