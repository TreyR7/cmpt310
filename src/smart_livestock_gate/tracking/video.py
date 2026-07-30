"""Frame iteration and detector-to-tracker glue for one CattleEyeView sequence."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from smart_livestock_gate.detection.detector import predict_frame
from smart_livestock_gate.tracking.overlay import TrailStore, draw_overlay
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


def video_fps(video_path: Path, default: float = 25.0) -> float:
    """Return the source video's frame rate, or ``default`` if unavailable."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()
    return float(fps) if fps and fps > 0 else default


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


@dataclass
class TrackingRun:
    """Everything one video-file tracking pass produced."""

    tracks: list[TrackedBox] = field(default_factory=list)
    frames_processed: int = 0
    source_fps: float = 0.0
    total_seconds: float = 0.0
    detection_seconds: float = 0.0

    @property
    def processing_fps(self) -> float:
        """Frames processed per wall-clock second (detection + tracking + draw)."""
        if not self.total_seconds:
            return 0.0
        return round(self.frames_processed / self.total_seconds, 2)

    @property
    def detection_fps(self) -> float:
        if not self.detection_seconds:
            return 0.0
        return round(self.frames_processed / self.detection_seconds, 2)


def track_video_file(
    model_path: Path,
    frame_source: Iterator[tuple[int, Any]],
    tracker: SortTracker,
    *,
    video_name: str,
    source_fps: float = 25.0,
    annotated_path: Path | None = None,
    confidence: float = 0.25,
    image_size: int = 512,
    device: str = "auto",
    max_frames: int | None = None,
    predict_fn=predict_frame,
) -> TrackingRun:
    """Detect, track, and (optionally) render an annotated MP4 for one video.

    Times each frame so the caller can report a realistic processing FPS, and
    keeps ``lost`` tracks in the drawn overlay so a short occlusion does not blink
    the box off. The exported ``tracks`` include both confirmed and lost records,
    each tagged with its lifecycle state.
    """
    run = TrackingRun(source_fps=source_fps)
    trails = TrailStore()
    writer: cv2.VideoWriter | None = None
    running_fps = source_fps
    started = time.perf_counter()

    try:
        for frame_index, frame in frame_source:
            if max_frames is not None and frame_index >= max_frames:
                break
            frame_started = time.perf_counter()

            detect_started = time.perf_counter()
            prediction = predict_fn(
                model_path,
                frame,
                confidence=confidence,
                image_size=image_size,
                device=device,
            )
            run.detection_seconds += time.perf_counter() - detect_started

            width = prediction["image"]["width"]
            height = prediction["image"]["height"]
            detections = [
                {
                    "box": _denormalize(detection["box"], width, height),
                    "confidence": detection["confidence"],
                }
                for detection in prediction["detections"]
            ]
            timestamp = frame_index / source_fps if source_fps else None
            tracks = tracker.update(
                detections,
                frame_index=frame_index,
                timestamp=timestamp,
                video=video_name,
                include_lost=True,
            )
            run.tracks.extend(tracks)

            frame_elapsed = time.perf_counter() - frame_started
            if frame_elapsed > 0:
                # Smooth the on-screen FPS so it does not jump every frame.
                running_fps = 0.9 * running_fps + 0.1 * (1.0 / frame_elapsed)

            if annotated_path is not None:
                if writer is None:
                    annotated_path = Path(annotated_path)
                    annotated_path.parent.mkdir(parents=True, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(annotated_path), fourcc, source_fps, (width, height)
                    )
                writer.write(
                    draw_overlay(
                        frame, tracks, trails, frame_index=frame_index, fps=running_fps
                    )
                )

            run.frames_processed += 1
    finally:
        if writer is not None:
            writer.release()

    run.total_seconds = time.perf_counter() - started
    return run
