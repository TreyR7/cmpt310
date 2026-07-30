"""Read-only tracking results for the web frontend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2


def render_sequence_frame(
    dataset_root: Path,
    sequence: str,
    frame_index: int,
    width: int = 640,
) -> bytes | None:
    """Return one extracted sequence frame as downscaled JPEG bytes.

    ``sequence`` is validated against the directories that actually exist under
    the dataset's images root, and the file name is rebuilt from the integer
    frame index, so no caller-supplied string ever reaches the filesystem path.
    Frames are 1920x1080 on disk; the client only needs a canvas-sized copy.
    """
    images_root = dataset_root / "images"
    if not images_root.is_dir():
        return None

    allowed = {path.name for path in images_root.iterdir() if path.is_dir()}
    if sequence not in allowed or frame_index < 0:
        return None

    # CattleEyeView names frames 1-based with five digits: index 30 -> 00031.jpg
    frame_path = images_root / sequence / f"{frame_index + 1:05d}.jpg"
    if not frame_path.is_file():
        return None

    image = cv2.imread(str(frame_path))
    if image is None:
        return None

    width = max(160, min(int(width), 1920))
    height = round(image.shape[0] * width / image.shape[1])
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    encoded, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 78])
    return buffer.tobytes() if encoded else None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def build_tracking_overview(
    metrics_path: Path,
    demo_path: Path,
    demo_video_path: Path,
) -> dict[str, Any]:
    """Summarize Step 2 tracking results for the dashboard.

    Returns the tuned tracker configuration, the aggregate and per-sequence
    identity metrics, and flags for whether the demo payload / annotated video
    are available to stream.
    """
    metrics = _read_json(metrics_path)
    ready = metrics is not None
    return {
        "ready": ready,
        "config": metrics.get("config") if ready else None,
        "evaluation_sequences": (
            metrics.get("evaluation_sequences", []) if ready else []
        ),
        "note": metrics.get("note") if ready else None,
        "aggregate": metrics.get("aggregate") if ready else None,
        "per_sequence": metrics.get("per_sequence") if ready else None,
        "demo_available": demo_path.is_file(),
        "video_available": demo_video_path.is_file(),
    }
