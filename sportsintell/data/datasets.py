from __future__ import annotations

from pathlib import Path

from .mot import MOTTrajectoryDataset


class SportsMOTDataset(MOTTrajectoryDataset):
    def __init__(self, root: str | Path, split: str, history_len: int = 12, normalize: bool = True) -> None:
        super().__init__(root=root, split=split, history_len=history_len, normalize=normalize)


class SoccerNetTrackingDataset(MOTTrajectoryDataset):
    def __init__(self, root: str | Path, split: str, history_len: int = 12, normalize: bool = True) -> None:
        super().__init__(root=root, split=split, history_len=history_len, normalize=normalize)
