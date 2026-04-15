from __future__ import annotations

from pathlib import Path
from typing import Iterable


def write_mot_predictions(path: str | Path, rows: Iterable[tuple[int, int, float, float, float, float, float]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for frame, track_id, x, y, w, h, conf in rows:
        lines.append(f"{frame},{track_id},{x:.4f},{y:.4f},{w:.4f},{h:.4f},{conf:.4f},-1,-1,-1")
    path.write_text("\n".join(lines), encoding="utf-8")
