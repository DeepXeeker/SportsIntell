from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from sportsintell.data.datasets import SoccerNetTrackingDataset, SportsMOTDataset
from sportsintell.data.mot import collate_batch
from sportsintell.models.baselines import NoMotionPredictor, VanillaTransformerPredictor
from sportsintell.models.sportsintell import SportsIntell


def build_dataset(config: dict[str, Any], split: str):
    dataset_name = config["data"]["name"].lower()
    root = config["data"]["root"]
    history_len = config["model"]["history_len"]
    normalize = config["data"].get("normalize", True)
    if dataset_name == "sportsmot":
        return SportsMOTDataset(root=root, split=split, history_len=history_len, normalize=normalize)
    if dataset_name in {"soccernet", "soccernet-tracking", "soccernet_tracking"}:
        return SoccerNetTrackingDataset(root=root, split=split, history_len=history_len, normalize=normalize)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_dataloaders(config: dict[str, Any]):
    train_ds = build_dataset(config, split=config["data"].get("train_split", "train"))
    val_ds = build_dataset(config, split=config["data"].get("val_split", "val"))
    batch_size = config["train"]["batch_size"]
    workers = config["train"].get("num_workers", 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate_batch)
    return train_loader, val_loader


def build_model(config: dict[str, Any]):
    model_name = config["model"]["name"].lower()
    common = dict(
        input_dim=config["model"].get("input_dim", 8),
        embed_dim=config["model"].get("embed_dim", 32),
        dropout=config["model"].get("dropout", 0.1),
    )
    if model_name == "sportsintell":
        return SportsIntell(
            **common,
            tfen_blocks=config["model"].get("tfen_blocks", 4),
            transformer_layers=config["model"].get("transformer_layers", 6),
            attention_heads=config["model"].get("attention_heads", 8),
            use_tfen=config["model"].get("use_tfen", True),
        )
    if model_name == "vanilla_transformer":
        return VanillaTransformerPredictor(
            input_dim=common["input_dim"],
            embed_dim=common["embed_dim"],
            layers=config["model"].get("transformer_layers", 6),
            heads=config["model"].get("attention_heads", 8),
            dropout=common["dropout"],
        )
    if model_name == "no_motion":
        return NoMotionPredictor()
    raise ValueError(f"Unsupported model: {model_name}")


def build_optimizer(config: dict[str, Any], model):
    return torch.optim.Adam(
        model.parameters(),
        lr=float(config["train"].get("lr", 1.5e-3)),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
