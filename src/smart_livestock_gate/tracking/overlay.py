"""Draw tracker output onto video frames for qualitative review.

Each visible track is drawn as a colored box with a persistent ID, the label,
detector confidence, and a short center-point trail. A header shows the current
frame number and the running processing FPS. Colors are derived from the track
ID so the same animal keeps the same color for the whole clip and an ID switch
is visible as a sudden color change.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable

import cv2
import numpy as np

from smart_livestock_gate.tracking.schema import TrackedBox

# Distinct, high-contrast BGR colors cycled by track ID.
_PALETTE = (
    (66, 133, 244),
    (52, 168, 83),
    (251, 188, 5),
    (234, 67, 53),
    (168, 85, 247),
    (0, 188, 212),
    (255, 112, 67),
    (156, 204, 101),
    (236, 64, 122),
    (38, 198, 218),
)


def color_for(track_id: int) -> tuple[int, int, int]:
    """Return a stable BGR color for a track ID."""
    return _PALETTE[track_id % len(_PALETTE)]


class TrailStore:
    """Keeps a short history of each track's box center for drawing trails."""

    def __init__(self, max_length: int = 30) -> None:
        self._points: dict[int, deque] = defaultdict(lambda: deque(maxlen=max_length))

    def update(self, track_id: int, center: tuple[int, int]) -> deque:
        self._points[track_id].append(center)
        return self._points[track_id]


def draw_overlay(
    frame: np.ndarray,
    tracks: Iterable[TrackedBox],
    trails: TrailStore,
    *,
    frame_index: int,
    fps: float,
) -> np.ndarray:
    """Draw boxes, IDs, confidence, trails, and a header onto a copy of ``frame``."""
    canvas = frame.copy()
    tracks = list(tracks)
    for track in tracks:
        x1, y1, x2, y2 = (int(round(value)) for value in track.box_xyxy)
        color = color_for(track.track_id)
        # A lost (coasting) track is drawn dashed-thin so it reads as uncertain.
        thickness = 2 if track.track_state == "confirmed" else 1
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        label = f"#{track.track_id} {track.label} {track.confidence:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            canvas, (x1, y1 - text_h - 6), (x1 + text_w + 4, y1), color, cv2.FILLED
        )
        cv2.putText(
            canvas,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        history = trails.update(track.track_id, center)
        for previous, current in zip(history, list(history)[1:], strict=False):
            cv2.line(canvas, previous, current, color, 2, cv2.LINE_AA)

    header = f"frame {frame_index}  |  {fps:.1f} FPS  |  {_count(tracks)} tracked"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (24, 24, 24), cv2.FILLED)
    cv2.putText(
        canvas,
        header,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _count(tracks: Iterable[TrackedBox]) -> int:
    return sum(1 for track in tracks if track.track_state == "confirmed")
