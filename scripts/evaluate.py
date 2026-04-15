from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sportsintell.engine.build import build_dataloaders, build_model
from sportsintell.engine.evaluator import evaluate_regression
from sportsintell.utils.checkpoint import load_checkpoint
from sportsintell.utils.config import load_config
from sportsintell.utils.logger import dump_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    _, val_loader = build_dataloaders(config)

    model = build_model(config)
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])

    device = "cuda" if torch.cuda.is_available() and config.get("device", "auto") != "cpu" else "cpu"
    model.to(device)

    metrics = evaluate_regression(model, val_loader, device=device)
    out_path = Path(args.checkpoint).with_name("eval_metrics.json")
    dump_json(metrics, out_path)
    print(metrics)


if __name__ == "__main__":
    main()
