"""Single-track Kalman filter state, following the standard SORT parameterization."""

from __future__ import annotations

import numpy as np
from filterpy.kalman import KalmanFilter


def _xyxy_to_state(xyxy: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [cx, cy, scale, aspect_ratio]."""
    width = xyxy[2] - xyxy[0]
    height = xyxy[3] - xyxy[1]
    center_x = xyxy[0] + width / 2.0
    center_y = xyxy[1] + height / 2.0
    scale = width * height
    aspect_ratio = width / height if height > 0 else 0.0
    return np.array([center_x, center_y, scale, aspect_ratio])


def _state_to_xyxy(state: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, scale, aspect_ratio, ...] back to [x1, y1, x2, y2]."""
    center_x, center_y, scale, aspect_ratio = state[:4]
    scale = max(scale, 0.0)
    width = np.sqrt(scale * aspect_ratio) if aspect_ratio > 0 else 0.0
    height = scale / width if width > 0 else 0.0
    return np.array(
        [
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        ]
    )


class KalmanBoxTracker:
    """Tracks one object's box across frames with a constant-velocity Kalman filter."""

    _next_id = 1

    def __init__(self, xyxy: np.ndarray, confidence: float, frame_index: int) -> None:
        self._filter = KalmanFilter(dim_x=7, dim_z=4)
        self._filter.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self._filter.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )
        self._filter.R[2:, 2:] *= 10.0
        self._filter.P[4:, 4:] *= 1000.0
        self._filter.P *= 10.0
        self._filter.Q[-1, -1] *= 0.01
        self._filter.Q[4:, 4:] *= 0.01
        self._filter.x[:4] = _xyxy_to_state(xyxy).reshape((4, 1))

        self._track_id = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1
        self._confidence = confidence
        self._hits = 1
        self._time_since_update = 0
        self._age = 0
        self._created_at = frame_index

    @classmethod
    def reset_ids(cls) -> None:
        cls._next_id = 1

    def predict(self) -> np.ndarray:
        """Advance state one frame and return the predicted xyxy box."""
        if self._filter.x[6] + self._filter.x[2] <= 0:
            self._filter.x[6] = 0.0
        self._filter.predict()
        self._age += 1
        self._time_since_update += 1
        return _state_to_xyxy(self._filter.x.reshape(-1))

    def update(self, xyxy: np.ndarray, confidence: float, frame_index: int) -> None:
        """Correct the filter with a matched detection."""
        self._time_since_update = 0
        self._hits += 1
        self._confidence = confidence
        self._filter.update(_xyxy_to_state(xyxy).reshape((4, 1)))

    @property
    def xyxy(self) -> np.ndarray:
        return _state_to_xyxy(self._filter.x.reshape(-1))

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def track_id(self) -> int:
        return self._track_id

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def time_since_update(self) -> int:
        return self._time_since_update

    @property
    def age(self) -> int:
        return self._age
