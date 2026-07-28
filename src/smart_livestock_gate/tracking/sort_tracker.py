"""Frame-by-frame SORT-style multi-object tracker."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smart_livestock_gate.tracking.association import associate
from smart_livestock_gate.tracking.kalman_box_tracker import KalmanBoxTracker
from smart_livestock_gate.tracking.schema import TrackedBox


@dataclass(frozen=True)
class SortTrackerConfig:
    """Reproducible tracker parameters."""

    iou_threshold: float = 0.3
    max_age: int = 5
    min_hits: int = 3


class SortTracker:
    """Associates per-frame detections into persistent tracks."""

    def __init__(self, config: SortTrackerConfig = SortTrackerConfig()) -> None:
        self._config = config
        self._tracks: list[KalmanBoxTracker] = []

    def reset(self) -> None:
        """Clear all state and restart ID numbering."""
        self._tracks = []
        KalmanBoxTracker.reset_ids()

    def update(
        self,
        detections: list[dict],
        frame_index: int,
        timestamp: float | None = None,
    ) -> list[TrackedBox]:
        """Advance every track one frame and associate it with new detections.

        ``detections`` is a list of {"box": [x1, y1, x2, y2], "confidence": float}
        in a consistent coordinate space (pixel or normalized, caller's choice).
        Returns confirmed tracks only (see ``SortTrackerConfig.min_hits``).
        """
        predicted_boxes = np.array(
            [track.predict() for track in self._tracks], dtype=np.float32
        ) if self._tracks else np.empty((0, 4), dtype=np.float32)

        detection_boxes = np.array(
            [detection["box"] for detection in detections], dtype=np.float32
        ) if detections else np.empty((0, 4), dtype=np.float32)

        matches, unmatched_tracks, unmatched_detections = associate(
            predicted_boxes, detection_boxes, iou_threshold=self._config.iou_threshold
        )

        for track_index, detection_index in matches:
            detection = detections[detection_index]
            self._tracks[track_index].update(
                np.array(detection["box"], dtype=np.float32),
                detection["confidence"],
                frame_index,
            )

        for detection_index in unmatched_detections:
            detection = detections[detection_index]
            self._tracks.append(
                KalmanBoxTracker(
                    np.array(detection["box"], dtype=np.float32),
                    detection["confidence"],
                    frame_index,
                )
            )

        max_age = self._config.max_age
        self._tracks = [
            track for track in self._tracks if track.time_since_update <= max_age
        ]

        results = []
        min_hits = self._config.min_hits
        for track in self._tracks:
            is_confirmed = track.hits >= min_hits or track.age <= min_hits
            if track.time_since_update == 0 and is_confirmed:
                results.append(
                    TrackedBox(
                        frame_index=frame_index,
                        timestamp=timestamp,
                        track_id=track.track_id,
                        box_xyxy=track.xyxy.tolist(),
                        confidence=track.confidence,
                    )
                )
        return results
