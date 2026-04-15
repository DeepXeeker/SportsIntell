from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sportsintell.engine.build import build_dataloaders, build_model, build_optimizer
from sportsintell.engine.trainer import Trainer
from sportsintell.utils.checkpoint import save_checkpoint
from sportsintell.utils.config import load_config
from sportsintell.utils.logger import dump_json, setup_logger
from sportsintell.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))

    output_dir = Path(config["output"]["dir"]) / config["output"]["run_name"]
    logger = setup_logger(output_dir)

    device = "cuda" if torch.cuda.is_available() and config.get("device", "auto") != "cpu" else "cpu"
    logger.info("Using device: %s", device)

    train_loader, val_loader = build_dataloaders(config)
    model = build_model(config).to(device)
    optimizer = build_optimizer(config, model)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s", trainable_params)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        beta=float(config["loss"].get("beta", 0.4)),
        use_dmal=bool(config["loss"].get("use_dmal", True)),
        device=device,
    )

    best_val = float("inf")
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "train_pred": [], "val_pred": [], "train_dmal": [], "val_dmal": []}

    for epoch in range(int(config["train"].get("epochs", 50))):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate_epoch(val_loader)
        history["train_loss"].append(train_metrics.total_loss)
        history["val_loss"].append(val_metrics.total_loss)
        history["train_pred"].append(train_metrics.pred_loss)
        history["val_pred"].append(val_metrics.pred_loss)
        history["train_dmal"].append(train_metrics.dmal_loss)
        history["val_dmal"].append(val_metrics.dmal_loss)

        logger.info(
            "Epoch %d | train loss %.6f | val loss %.6f | train pred %.6f | val pred %.6f | train dmal %.6f | val dmal %.6f",
            epoch + 1,
            train_metrics.total_loss,
            val_metrics.total_loss,
            train_metrics.pred_loss,
            val_metrics.pred_loss,
            train_metrics.dmal_loss,
            val_metrics.dmal_loss,
        )

        save_checkpoint(
            output_dir / "last.pt",
            epoch=epoch + 1,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            config=config,
            history=history,
        )

        if val_metrics.total_loss < best_val:
            best_val = val_metrics.total_loss
            save_checkpoint(
                output_dir / "best.pt",
                epoch=epoch + 1,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                config=config,
                history=history,
            )

    dump_json({"best_val_loss": best_val, "history": history, "parameters": trainable_params}, output_dir / "metrics.json")


if __name__ == "__main__":
    main()
