"""Frame iteration and detector-to-tracker glue for one CattleEyeView sequence."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2

from smart_livestock_gate.detection.detector import predict_frame
from smart_livestock_gate.tracking.schema import TrackedBox
from smart_livestock_gate.tracking.sort_tracker import SortTracker


def iter_frames_from_directory(frame_dir: Path) -> Iterator[tuple[int, Any]]:
    """Yield (frame_index, frame) for sorted *.jpg files in a sequence directory.

    Frame index is 0-based, matching TrackingRecord.frame_index.
    """
    for frame_index, frame_path in enumerate(sorted(frame_dir.glob("*.jpg"))):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise ValueError(f"Could not read frame: {frame_path}")
        yield frame_index, frame


def iter_frames_from_video(video_path: Path) -> Iterator[tuple[int, Any]]:
    """Yield (frame_index, frame) by decoding a video file directly."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        frame_index = 0
        while True:
            read_ok, frame = capture.read()
            if not read_ok:
                break
            yield frame_index, frame
            frame_index += 1
    finally:
        capture.release()


def _denormalize(box_xyxyn: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box_xyxyn
    return [x1 * width, y1 * height, x2 * width, y2 * height]


def track_sequence(
    model_path: Path,
    frame_source: Iterator[tuple[int, Any]],
    tracker: SortTracker,
    *,
    confidence: float = 0.25,
    image_size: int = 512,
    device: str = "auto",
    predict_fn=predict_frame,
) -> list[TrackedBox]:
    """Run detection and tracking over a sequence of frames.

    Converts the detector's normalized xyxy boxes to pixel coordinates using
    each frame's own dimensions before handing them to the tracker, since the
    tracker's Kalman motion model operates in pixel space. ``predict_fn`` is
    injectable so tests can supply a stub instead of a real model.
    """
    all_tracks: list[TrackedBox] = []
    for frame_index, frame in frame_source:
        prediction = predict_fn(
            model_path,
            frame,
            confidence=confidence,
            image_size=image_size,
            device=device,
        )
        width = prediction["image"]["width"]
        height = prediction["image"]["height"]
        detections = [
            {
                "box": _denormalize(detection["box"], width, height),
                "confidence": detection["confidence"],
            }
            for detection in prediction["detections"]
        ]
        all_tracks.extend(tracker.update(detections, frame_index=frame_index))
    return all_tracks
