from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SequenceMeta:
    name: str
    width: int | None = None
    height: int | None = None
    frame_rate: int | None = None
    length: int | None = None


def read_seqinfo(path: str | Path) -> SequenceMeta:
    path = Path(path)
    values: dict[str, str] = {}
    if not path.exists():
        return SequenceMeta(name=path.parent.name)
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip()
    return SequenceMeta(
        name=values.get("name", path.parent.name),
        width=int(values["imWidth"]) if "imWidth" in values else None,
        height=int(values["imHeight"]) if "imHeight" in values else None,
        frame_rate=int(values["frameRate"]) if "frameRate" in values else None,
        length=int(values["seqLength"]) if "seqLength" in values else None,
    )


def read_mot_file(path: str | Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            rows.append([float(x) for x in row[:10]])
    return rows


def tlwh_to_cxcywh(tlwh: np.ndarray) -> np.ndarray:
    x, y, w, h = tlwh
    return np.array([x + w / 2.0, y + h / 2.0, w, h], dtype=np.float32)


def make_state(curr: np.ndarray, prev: np.ndarray | None) -> np.ndarray:
    if prev is None:
        delta = np.zeros(4, dtype=np.float32)
    else:
        delta = curr - prev
    return np.concatenate([curr, delta]).astype(np.float32)


class MOTTrajectoryDataset(Dataset):
    """
    Builds sliding-window trajectory samples from MOT-format GT files.

    Each sample contains:
    - history: [p, 8]
    - target_offset: [4]
    - prev_box: [4]
    - next_box: [4]
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        history_len: int = 12,
        normalize: bool = True,
    ) -> None:
        self.root = Path(root) / split
        self.history_len = history_len
        self.normalize = normalize
        self.samples: list[dict[str, np.ndarray | str | int]] = []
        self._build_index()

    def _build_index(self) -> None:
        for seq_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            gt_path = seq_dir / "gt" / "gt.txt"
            if not gt_path.exists():
                continue
            meta = read_seqinfo(seq_dir / "seqinfo.ini")
            rows = read_mot_file(gt_path)
            by_track: dict[int, list[tuple[int, np.ndarray]]] = {}
            for row in rows:
                frame_id = int(row[0])
                track_id = int(row[1])
                box = tlwh_to_cxcywh(np.array(row[2:6], dtype=np.float32))
                by_track.setdefault(track_id, []).append((frame_id, box))
            for track_id, items in by_track.items():
                items.sort(key=lambda x: x[0])
                boxes = [box for _, box in items]
                states = [make_state(boxes[i], boxes[i - 1] if i > 0 else None) for i in range(len(boxes))]
                for end in range(self.history_len, len(states)):
                    history = np.stack(states[end - self.history_len : end], axis=0)
                    prev_box = boxes[end - 1].copy()
                    next_box = boxes[end].copy()
                    target_offset = next_box - prev_box
                    if self.normalize and meta.width and meta.height:
                        scale8 = np.array(
                            [meta.width, meta.height, meta.width, meta.height] * 2,
                            dtype=np.float32,
                        )
                        scale4 = np.array([meta.width, meta.height, meta.width, meta.height], dtype=np.float32)
                        history = history / scale8
                        prev_box = prev_box / scale4
                        next_box = next_box / scale4
                        target_offset = target_offset / scale4
                    self.samples.append(
                        {
                            "history": history.astype(np.float32),
                            "prev_box": prev_box.astype(np.float32),
                            "next_box": next_box.astype(np.float32),
                            "target_offset": target_offset.astype(np.float32),
                            "sequence": seq_dir.name,
                            "track_id": track_id,
                        }
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        item = self.samples[index]
        return {
            "history": torch.from_numpy(item["history"]),
            "prev_box": torch.from_numpy(item["prev_box"]),
            "next_box": torch.from_numpy(item["next_box"]),
            "target_offset": torch.from_numpy(item["target_offset"]),
            "sequence": str(item["sequence"]),
            "track_id": int(item["track_id"]),
        }


def collate_batch(batch: list[dict[str, torch.Tensor | str | int]]) -> dict[str, torch.Tensor | list[str] | list[int]]:
    return {
        "history": torch.stack([item["history"] for item in batch], dim=0),
        "prev_box": torch.stack([item["prev_box"] for item in batch], dim=0),
        "next_box": torch.stack([item["next_box"] for item in batch], dim=0),
        "target_offset": torch.stack([item["target_offset"] for item in batch], dim=0),
        "sequence": [str(item["sequence"]) for item in batch],
        "track_id": [int(item["track_id"]) for item in batch],
    }


def iter_sequence_rows(path: str | Path) -> Iterator[tuple[int, int, np.ndarray]]:
    for row in read_mot_file(path):
        frame_id = int(row[0])
        track_id = int(row[1])
        box = tlwh_to_cxcywh(np.array(row[2:6], dtype=np.float32))
        yield frame_id, track_id, box
