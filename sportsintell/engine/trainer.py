from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sportsintell.losses.dmal import DirectionalMotionAlignmentLoss
from sportsintell.losses.prediction import OffsetL1Loss


@dataclass
class TrainMetrics:
    total_loss: float
    pred_loss: float
    dmal_loss: float


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        beta: float = 0.4,
        use_dmal: bool = True,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.beta = beta
        self.use_dmal = use_dmal
        self.device = device
        self.pred_loss_fn = OffsetL1Loss()
        self.dmal_loss_fn = DirectionalMotionAlignmentLoss()

    def train_epoch(self, loader: DataLoader) -> TrainMetrics:
        self.model.train()
        total = pred_total = dmal_total = 0.0
        count = 0
        for batch in tqdm(loader, desc="train", leave=False):
            history = batch["history"].to(self.device).float()
            prev_box = batch["prev_box"].to(self.device).float()
            next_box = batch["next_box"].to(self.device).float()
            target_offset = batch["target_offset"].to(self.device).float()

            pred_offset = self.model(history)
            pred_box = prev_box + pred_offset
            pred_loss = self.pred_loss_fn(pred_offset, target_offset)
            dmal_loss = self.dmal_loss_fn(prev_box, pred_box, next_box) if self.use_dmal else torch.tensor(0.0, device=self.device)
            loss = pred_loss + self.beta * dmal_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            total += float(loss.item())
            pred_total += float(pred_loss.item())
            dmal_total += float(dmal_loss.item())
            count += 1

        return TrainMetrics(
            total_loss=total / max(count, 1),
            pred_loss=pred_total / max(count, 1),
            dmal_loss=dmal_total / max(count, 1),
        )

    @torch.no_grad()
    def validate_epoch(self, loader: DataLoader) -> TrainMetrics:
        self.model.eval()
        total = pred_total = dmal_total = 0.0
        count = 0
        for batch in tqdm(loader, desc="val", leave=False):
            history = batch["history"].to(self.device).float()
            prev_box = batch["prev_box"].to(self.device).float()
            next_box = batch["next_box"].to(self.device).float()
            target_offset = batch["target_offset"].to(self.device).float()

            pred_offset = self.model(history)
            pred_box = prev_box + pred_offset
            pred_loss = self.pred_loss_fn(pred_offset, target_offset)
            dmal_loss = self.dmal_loss_fn(prev_box, pred_box, next_box) if self.use_dmal else torch.tensor(0.0, device=self.device)
            loss = pred_loss + self.beta * dmal_loss

            total += float(loss.item())
            pred_total += float(pred_loss.item())
            dmal_total += float(dmal_loss.item())
            count += 1

        return TrainMetrics(
            total_loss=total / max(count, 1),
            pred_loss=pred_total / max(count, 1),
            dmal_loss=dmal_total / max(count, 1),
        )
