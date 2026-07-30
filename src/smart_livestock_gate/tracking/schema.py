"""Output schema shared by the tracker, its video glue, and evaluation code.

The record shape mirrors the Step 2 specification so downstream counting code
(Step 3) can read persisted tracks without rerunning the video:

    {
      "video": "01.mp4",
      "frame_index": 153,
      "timestamp_seconds": 5.1,
      "track_id": 7,
      "label": "cattle",
      "confidence": 0.91,
      "bbox_xyxy": [412.3, 180.6, 587.1, 394.2],
      "track_state": "confirmed"
    }
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# Lifecycle states a track can occupy. Only ``confirmed`` tracks are handed to
# the future counting stage; ``lost`` tracks are kept for visualization and
# short-occlusion recovery.
TRACK_STATES = ("tentative", "confirmed", "lost", "removed")

# Column order used for CSV export. Matches ``TrackedBox.to_record`` keys.
RECORD_FIELDS = (
    "video",
    "frame_index",
    "timestamp_seconds",
    "track_id",
    "label",
    "confidence",
    "bbox_xyxy",
    "track_state",
)


@dataclass(frozen=True)
class TrackedBox:
    """One tracker output for one track in one frame.

    ``video``, ``label`` and ``track_state`` carry defaults so unit tests and
    the in-memory tracker can build minimal records, while the video runner
    fills every field for the exported manifest.
    """

    frame_index: int
    timestamp: float | None
    track_id: int
    box_xyxy: list[float]
    confidence: float
    video: str = ""
    label: str = "cattle"
    track_state: str = "confirmed"

    def to_record(self) -> dict:
        """Return a JSON-ready dict using the Step 2 specification field names."""
        return {
            "video": self.video,
            "frame_index": self.frame_index,
            "timestamp_seconds": (
                None if self.timestamp is None else round(self.timestamp, 3)
            ),
            "track_id": self.track_id,
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "bbox_xyxy": [round(float(value), 2) for value in self.box_xyxy],
            "track_state": self.track_state,
        }


def write_jsonl(tracks: Iterable[TrackedBox], output_path: Path) -> Path:
    """Write one JSON object per line, one line per tracked box."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for track in tracks:
            file.write(json.dumps(track.to_record()) + "\n")
    return output_path


def write_csv(tracks: Iterable[TrackedBox], output_path: Path) -> Path:
    """Write tracked boxes as CSV; the bbox is stored as a JSON list string."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        for track in tracks:
            record = track.to_record()
            record["bbox_xyxy"] = json.dumps(record["bbox_xyxy"])
            writer.writerow(record)
    return output_path
