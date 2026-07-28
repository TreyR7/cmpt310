"""Output schema shared by the tracker, its video glue, and evaluation code."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackedBox:
    """One tracker output for one track in one frame."""

    frame_index: int
    timestamp: float | None
    track_id: int
    box_xyxy: list[float]
    confidence: float
