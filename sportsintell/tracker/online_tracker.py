from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from sportsintell.tracker.association import hungarian_iou_match
from sportsintell.tracker.kalman import SimpleKalmanBoxFilter


@dataclass
class Track:
    track_id: int
    history: list[np.ndarray] = field(default_factory=list)
    age: int = 0
    missed: int = 0
    kalman: SimpleKalmanBoxFilter | None = None

    @property
    def last_box(self) -> np.ndarray:
        return self.history[-1]

    def push(self, box: np.ndarray) -> None:
        self.history.append(box.astype(np.float32))
        self.age += 1
        self.missed = 0


class OnlineSportsTracker:
    def __init__(
        self,
        motion_model: nn.Module | None = None,
        history_len: int = 12,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        device: str = "cpu",
        use_kalman: bool = False,
    ) -> None:
        self.motion_model = motion_model
        self.history_len = history_len
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.device = device
        self.use_kalman = use_kalman
        self.tracks: list[Track] = []
        self.next_id = 1

    @staticmethod
    def _state_from_history(history: list[np.ndarray], history_len: int) -> np.ndarray:
        states = []
        for i, box in enumerate(history[-history_len:]):
            prev = history[-history_len + i - 1] if i > 0 else None
            if prev is None:
                delta = np.zeros(4, dtype=np.float32)
            else:
                delta = box - prev
            states.append(np.concatenate([box, delta], axis=0))
        if len(states) < history_len:
            pad = [np.zeros(8, dtype=np.float32) for _ in range(history_len - len(states))]
            states = pad + states
        return np.stack(states, axis=0)

    def _predict_track_boxes(self) -> np.ndarray:
        predicted = []
        for track in self.tracks:
            if self.use_kalman and track.kalman is not None:
                predicted.append(track.kalman.predict())
                continue
            if self.motion_model is None:
                predicted.append(track.last_box.copy())
                continue
            history = self._state_from_history(track.history, self.history_len)
            history_t = torch.from_numpy(history).unsqueeze(0).to(self.device)
            prev_t = torch.from_numpy(track.last_box).unsqueeze(0).to(self.device)
            with torch.no_grad():
                if hasattr(self.motion_model, "predict_boxes"):
                    pred_box = self.motion_model.predict_boxes(history_t.float(), prev_t.float())
                else:
                    offsets = self.motion_model(history_t.float())
                    pred_box = prev_t.float() + offsets
            predicted.append(pred_box.squeeze(0).cpu().numpy().astype(np.float32))
        return np.stack(predicted, axis=0) if predicted else np.zeros((0, 4), dtype=np.float32)

    def initialize(self, detections: np.ndarray) -> None:
        for det in detections:
            kalman = SimpleKalmanBoxFilter(det) if self.use_kalman else None
            self.tracks.append(Track(track_id=self.next_id, history=[det.astype(np.float32)], kalman=kalman))
            self.next_id += 1

    def update(self, detections: np.ndarray) -> list[Track]:
        if len(self.tracks) == 0:
            self.initialize(detections)
            return self.tracks

        predicted = self._predict_track_boxes()
        match = hungarian_iou_match(predicted, detections, iou_threshold=self.iou_threshold)

        for ti, di in match.matches:
            self.tracks[ti].push(detections[di])
            if self.tracks[ti].kalman is not None:
                self.tracks[ti].kalman.update(detections[di])

        for ti in match.unmatched_tracks:
            self.tracks[ti].missed += 1
            if self.use_kalman and self.tracks[ti].kalman is not None:
                predicted_box = self.tracks[ti].kalman.predict()
                self.tracks[ti].history.append(predicted_box)
            else:
                self.tracks[ti].history.append(predicted[ti])

        for di in match.unmatched_detections:
            det = detections[di]
            kalman = SimpleKalmanBoxFilter(det) if self.use_kalman else None
            self.tracks.append(Track(track_id=self.next_id, history=[det.astype(np.float32)], kalman=kalman))
            self.next_id += 1

        self.tracks = [track for track in self.tracks if track.missed <= self.max_age]
        return self.tracks
