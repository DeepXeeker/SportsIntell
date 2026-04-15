from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from sportsintell.engine.build import build_model
from sportsintell.tracker.online_tracker import OnlineSportsTracker
from sportsintell.utils.checkpoint import load_checkpoint
from sportsintell.utils.config import load_config
from sportsintell.data.mot import read_mot_file


def tlwh_rows_to_cxcywh(rows: list[list[float]], frame_id: int) -> np.ndarray:
    boxes = []
    for row in rows:
        if int(row[0]) != frame_id:
            continue
        x, y, w, h = row[2:6]
        boxes.append(np.array([x + w / 2.0, y + h / 2.0, w, h], dtype=np.float32))
    return np.stack(boxes, axis=0) if boxes else np.zeros((0, 4), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--sequence", required=True, help="Path to sequence directory containing seqinfo.ini or images")
    parser.add_argument("--detections", required=True, help="MOT-format detection file")
    parser.add_argument("--save", required=True)
    parser.add_argument("--kalman", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() and config.get("device", "auto") != "cpu" else "cpu"

    motion_model = None
    if not args.kalman:
        motion_model = build_model(config).to(device)
        if args.checkpoint:
            ckpt = load_checkpoint(args.checkpoint, map_location=device)
            motion_model.load_state_dict(ckpt["model_state"])
        motion_model.eval()

    tracker = OnlineSportsTracker(
        motion_model=motion_model,
        history_len=int(config["model"]["history_len"]),
        iou_threshold=float(config["tracker"].get("iou_threshold", 0.3)),
        max_age=int(config["tracker"].get("max_age", 30)),
        device=device,
        use_kalman=args.kalman,
    )

    rows = read_mot_file(args.detections)
    frame_ids = sorted({int(row[0]) for row in rows})
    output_lines: list[str] = []
    for frame_id in frame_ids:
        detections = tlwh_rows_to_cxcywh(rows, frame_id)
        tracks = tracker.update(detections)
        for track in tracks:
            box = track.last_box
            x = box[0] - box[2] / 2.0
            y = box[1] - box[3] / 2.0
            output_lines.append(f"{frame_id},{track.track_id},{x:.3f},{y:.3f},{box[2]:.3f},{box[3]:.3f},1,-1,-1,-1")

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save).write_text("\n".join(output_lines), encoding="utf-8")
    print(f"Saved {len(output_lines)} tracking rows to {args.save}")


if __name__ == "__main__":
    main()
