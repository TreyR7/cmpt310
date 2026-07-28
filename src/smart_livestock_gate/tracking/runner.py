"""End-to-end glue: run the tracker over one sequence and optionally score it."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from smart_livestock_gate.tracking.evaluate import evaluate_manifest
from smart_livestock_gate.tracking.schema import TrackedBox
from smart_livestock_gate.tracking.sort_tracker import SortTracker, SortTrackerConfig
from smart_livestock_gate.tracking.video import (
    iter_frames_from_directory,
    track_sequence,
)


@dataclass(frozen=True)
class TrackingRunConfig:
    """Reproducible settings for one tracking run over a CattleEyeView sequence."""

    sequence: str
    dataset_root: Path
    model_path: Path
    report_path: Path
    manifest_path: Path | None = None
    confidence: float = 0.25
    image_size: int = 512
    device: str = "auto"
    iou_threshold: float = 0.3
    max_age: int = 5
    min_hits: int = 3


def run_tracking(config: TrackingRunConfig) -> dict:
    """Track one sequence's frames and, if a manifest is given, score it."""
    frame_dir = config.dataset_root / "images" / config.sequence
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")

    tracker = SortTracker(
        SortTrackerConfig(
            iou_threshold=config.iou_threshold,
            max_age=config.max_age,
            min_hits=config.min_hits,
        )
    )
    tracks: list[TrackedBox] = track_sequence(
        config.model_path,
        iter_frames_from_directory(frame_dir),
        tracker,
        confidence=config.confidence,
        image_size=config.image_size,
        device=config.device,
    )

    report: dict = {
        "sequence": config.sequence,
        "frames_tracked": len({track.frame_index for track in tracks}),
        "tracks_reported": len(tracks),
        "unique_track_ids": len({track.track_id for track in tracks}),
    }

    if config.manifest_path is not None and config.manifest_path.is_file():
        evaluation = evaluate_manifest(
            config.manifest_path, {config.sequence: tracks}
        )
        report["evaluation"] = evaluation["sequences"].get(config.sequence)

    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(
        json.dumps({**report, "tracks": [asdict(track) for track in tracks]}, indent=2),
        encoding="utf-8",
    )
    return report
