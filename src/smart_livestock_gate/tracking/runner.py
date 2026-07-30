"""End-to-end glue: run the tracker over one video and optionally score it."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from smart_livestock_gate.tracking.evaluate import evaluate_manifest
from smart_livestock_gate.tracking.schema import write_csv, write_jsonl
from smart_livestock_gate.tracking.sort_tracker import SortTracker, SortTrackerConfig
from smart_livestock_gate.tracking.video import (
    iter_frames_from_directory,
    iter_frames_from_video,
    track_video_file,
    video_fps,
)


@dataclass(frozen=True)
class TrackVideoConfig:
    """Reproducible settings for one tracking run over a video or frame sequence."""

    input_path: Path
    detections_path: Path
    model_path: Path
    report_path: Path
    annotated_path: Path | None = None
    manifest_path: Path | None = None
    sequence_name: str | None = None
    confidence: float = 0.25
    image_size: int = 512
    device: str = "auto"
    iou_threshold: float = 0.3
    max_age: int = 5
    min_hits: int = 3
    max_frames: int | None = None
    source_fps: float = 25.0


def _resolve_frames(config: TrackVideoConfig):
    """Return (frame_iterator, source_fps, sequence_name) for the input path."""
    input_path = config.input_path
    if input_path.is_dir():
        # A CattleEyeView frame directory such as images/01.mp4/.
        sequence = config.sequence_name or input_path.name
        return (
            iter_frames_from_directory(input_path),
            config.source_fps,
            sequence,
        )
    if input_path.is_file():
        sequence = config.sequence_name or input_path.name
        return (
            iter_frames_from_video(input_path),
            video_fps(input_path, default=config.source_fps),
            sequence,
        )
    raise FileNotFoundError(f"Input video or frame directory not found: {input_path}")


def run_track_video(config: TrackVideoConfig) -> dict:
    """Track one video, export records, render an overlay, and score if possible."""
    frames, source_fps, sequence = _resolve_frames(config)

    tracker = SortTracker(
        SortTrackerConfig(
            iou_threshold=config.iou_threshold,
            max_age=config.max_age,
            min_hits=config.min_hits,
        )
    )
    run = track_video_file(
        config.model_path,
        frames,
        tracker,
        video_name=sequence,
        source_fps=source_fps,
        annotated_path=config.annotated_path,
        confidence=config.confidence,
        image_size=config.image_size,
        device=config.device,
        max_frames=config.max_frames,
    )

    # Export tracks; extension decides the serialization format.
    if config.detections_path.suffix.lower() == ".csv":
        write_csv(run.tracks, config.detections_path)
    else:
        write_jsonl(run.tracks, config.detections_path)

    confirmed = [track for track in run.tracks if track.track_state == "confirmed"]
    report: dict = {
        "sequence": sequence,
        "input": str(config.input_path),
        "detections": str(config.detections_path),
        "annotated_video": (
            str(config.annotated_path) if config.annotated_path else None
        ),
        "config": {
            "confidence_threshold": config.confidence,
            "iou_threshold": config.iou_threshold,
            "min_hits": config.min_hits,
            "max_age": config.max_age,
            "image_size": config.image_size,
        },
        "runtime": {
            "frames_processed": run.frames_processed,
            "source_fps": round(source_fps, 2),
            "processing_fps": run.processing_fps,
            "detection_fps": run.detection_fps,
        },
        "tracks": {
            "records_exported": len(run.tracks),
            "confirmed_records": len(confirmed),
            "unique_track_ids": len({track.track_id for track in run.tracks}),
        },
    }

    if config.manifest_path is not None and config.manifest_path.is_file():
        # Score confirmed identities only; lost/coasting boxes are for display.
        evaluation = evaluate_manifest(config.manifest_path, {sequence: confirmed})
        report["evaluation"] = evaluation["sequences"].get(sequence)

    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# --- Backwards-compatible sequence runner (used by the older ``track`` command) ---


@dataclass(frozen=True)
class TrackingRunConfig:
    """Reproducible settings for tracking a CattleEyeView frame sequence by name."""

    sequence: str
    dataset_root: Path
    model_path: Path
    report_path: Path
    detections_path: Path | None = None
    annotated_path: Path | None = None
    manifest_path: Path | None = None
    confidence: float = 0.25
    image_size: int = 512
    device: str = "auto"
    iou_threshold: float = 0.3
    max_age: int = 5
    min_hits: int = 3
    max_frames: int | None = None


def run_tracking(config: TrackingRunConfig) -> dict:
    """Track one sequence's extracted frames by name and score it."""
    frame_dir = config.dataset_root / "images" / config.sequence
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")

    detections_path = config.detections_path or (
        config.report_path.parent / f"{config.sequence}_tracks.jsonl"
    )
    return run_track_video(
        TrackVideoConfig(
            input_path=frame_dir,
            detections_path=detections_path,
            model_path=config.model_path,
            report_path=config.report_path,
            annotated_path=config.annotated_path,
            manifest_path=config.manifest_path,
            sequence_name=config.sequence,
            confidence=config.confidence,
            image_size=config.image_size,
            device=config.device,
            iou_threshold=config.iou_threshold,
            max_age=config.max_age,
            min_hits=config.min_hits,
            max_frames=config.max_frames,
        )
    )


__all__ = [
    "TrackVideoConfig",
    "TrackingRunConfig",
    "run_track_video",
    "run_tracking",
]
