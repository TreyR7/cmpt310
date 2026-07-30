"""Tests for Step 2 additions: export schema, lifecycle, metrics, video runner."""

import csv
import json

import numpy as np

from smart_livestock_gate.tracking.evaluate import evaluate_sequence
from smart_livestock_gate.tracking.schema import (
    TrackedBox,
    write_csv,
    write_jsonl,
)
from smart_livestock_gate.tracking.sort_tracker import SortTracker, SortTrackerConfig
from smart_livestock_gate.tracking.video import track_video_file

# --- Export schema -------------------------------------------------------

def test_tracked_box_to_record_uses_spec_field_names():
    box = TrackedBox(
        frame_index=153,
        timestamp=5.1,
        track_id=7,
        box_xyxy=[412.3, 180.6, 587.1, 394.2],
        confidence=0.911,
        video="01.mp4",
        track_state="confirmed",
    )
    record = box.to_record()
    assert set(record) == {
        "video",
        "frame_index",
        "timestamp_seconds",
        "track_id",
        "label",
        "confidence",
        "bbox_xyxy",
        "track_state",
    }
    assert record["video"] == "01.mp4"
    assert record["timestamp_seconds"] == 5.1
    assert record["label"] == "cattle"
    assert record["track_state"] == "confirmed"


def test_write_jsonl_round_trips(tmp_path):
    tracks = [
        TrackedBox(0, 0.0, 1, [0, 0, 10, 10], 0.9, video="01.mp4"),
        TrackedBox(1, 0.04, 1, [1, 1, 11, 11], 0.8, video="01.mp4"),
    ]
    path = write_jsonl(tracks, tmp_path / "out.jsonl")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["track_id"] == 1
    assert first["bbox_xyxy"] == [0.0, 0.0, 10.0, 10.0]


def test_write_csv_has_header_and_rows(tmp_path):
    tracks = [TrackedBox(0, 0.0, 1, [0, 0, 10, 10], 0.9, video="01.mp4")]
    path = write_csv(tracks, tmp_path / "out.csv")
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    assert rows[0]["track_id"] == "1"
    assert json.loads(rows[0]["bbox_xyxy"]) == [0.0, 0.0, 10.0, 10.0]


# --- Lifecycle states ----------------------------------------------------

def _box_at(frame, start=0.0, step=5.0, size=10.0):
    x1 = start + step * frame
    return {"box": [x1, 0.0, x1 + size, size], "confidence": 0.9}


def test_confirmed_state_after_min_hits():
    tracker = SortTracker(SortTrackerConfig(min_hits=3, max_age=5))
    states = []
    for frame in range(5):
        results = tracker.update([_box_at(frame)], frame_index=frame)
        if results:
            states.append(results[0].track_state)
    assert states[-1] == "confirmed"


def test_lost_state_emitted_only_when_requested():
    tracker = SortTracker(SortTrackerConfig(min_hits=1, max_age=5))
    tracker.update([_box_at(0)], frame_index=0)
    tracker.update([_box_at(1)], frame_index=1)

    # A frame with no detection: default hides the coasting track...
    hidden = tracker.update([], frame_index=2)
    assert hidden == []
    # ...but include_lost surfaces it with a "lost" state for the overlay.
    shown = tracker.update([], frame_index=3, include_lost=True)
    assert shown and shown[0].track_state == "lost"


# --- Metrics -------------------------------------------------------------

def _gt(frame_index, uid, box_xywh, sequence="01.mp4"):
    return {
        "sequence": sequence,
        "frame_index": frame_index,
        "track_uid": uid,
        "bbox_xywh": box_xywh,
    }


def test_recall_and_mostly_tracked():
    gt = [_gt(f, "01.mp4:1", [0, 0, 10, 10]) for f in range(10)]
    # Predicted for 9 of 10 frames -> recall 0.9, mostly-tracked (>=0.8).
    preds = [
        TrackedBox(f, None, 5, [0, 0, 10, 10], 0.9)
        for f in range(10)
        if f != 4
    ]
    evaluation = evaluate_sequence(gt, preds)
    assert evaluation.ground_truth_detections == 10
    assert evaluation.matched_detections == 9
    assert evaluation.matched_track_recall == 0.9
    assert evaluation.mostly_tracked == 1
    assert evaluation.mostly_tracked_ratio == 1.0


# --- Video-file runner ---------------------------------------------------

def _fake_predict(box):
    def predict_fn(model_path, frame, *, confidence, image_size, device):
        height, width = frame.shape[:2]
        return {
            "image": {"width": width, "height": height},
            "detections": [{"label": "cow", "confidence": 0.9, "box": box}],
            "count": 1,
            "inference_ms": 0.0,
            "confidence_threshold": confidence,
        }

    return predict_fn


def _frames(n, width=100, height=100):
    for index in range(n):
        yield index, np.zeros((height, width, 3), dtype=np.uint8)


def test_track_video_file_records_state_video_and_fps():
    tracker = SortTracker(SortTrackerConfig(min_hits=1))
    run = track_video_file(
        model_path="unused",
        frame_source=_frames(4),
        tracker=tracker,
        video_name="01.mp4",
        source_fps=25.0,
        predict_fn=_fake_predict([0.0, 0.0, 0.5, 0.5]),
    )
    assert run.frames_processed == 4
    assert run.processing_fps > 0
    assert all(track.video == "01.mp4" for track in run.tracks)
    assert all(track.timestamp is not None for track in run.tracks)
    assert len({track.track_id for track in run.tracks}) == 1


def test_track_video_file_respects_max_frames():
    tracker = SortTracker(SortTrackerConfig(min_hits=1))
    run = track_video_file(
        model_path="unused",
        frame_source=_frames(10),
        tracker=tracker,
        video_name="01.mp4",
        predict_fn=_fake_predict([0.0, 0.0, 0.5, 0.5]),
        max_frames=3,
    )
    assert run.frames_processed == 3
