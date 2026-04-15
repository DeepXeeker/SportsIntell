from __future__ import annotations

import numpy as np


class SimpleKalmanBoxFilter:
    """
    Minimal constant-velocity Kalman filter for bbox center/size.
    State: [x, y, w, h, vx, vy, vw, vh]
    """

    def __init__(self, bbox: np.ndarray) -> None:
        self.x = np.zeros(8, dtype=np.float32)
        self.x[:4] = bbox
        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0
        self.H = np.zeros((4, 8), dtype=np.float32)
        self.H[:4, :4] = np.eye(4, dtype=np.float32)
        self.Q = np.eye(8, dtype=np.float32) * 1e-2
        self.R = np.eye(4, dtype=np.float32) * 1.0

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4].copy()

    def update(self, bbox: np.ndarray) -> None:
        y = bbox - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8, dtype=np.float32) - K @ self.H) @ self.P
