from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sportsintell.losses.dmal import DirectionalMotionAlignmentLoss
from sportsintell.losses.prediction import OffsetL1Loss


@torch.no_grad()
def evaluate_regression(model, loader: DataLoader, device: str = "cpu") -> dict[str, float]:
    model.eval()
    pred_loss_fn = OffsetL1Loss()
    dmal_loss_fn = DirectionalMotionAlignmentLoss()
    pred_total = 0.0
    dmal_total = 0.0
    count = 0
    for batch in tqdm(loader, desc="eval", leave=False):
        history = batch["history"].to(device).float()
        prev_box = batch["prev_box"].to(device).float()
        next_box = batch["next_box"].to(device).float()
        target_offset = batch["target_offset"].to(device).float()

        pred_offset = model(history)
        pred_box = prev_box + pred_offset
        pred_total += float(pred_loss_fn(pred_offset, target_offset).item())
        dmal_total += float(dmal_loss_fn(prev_box, pred_box, next_box).item())
        count += 1

    return {
        "offset_l1": pred_total / max(count, 1),
        "dmal": dmal_total / max(count, 1),
    }
