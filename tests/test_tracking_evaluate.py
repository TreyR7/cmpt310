import json

from smart_livestock_gate.tracking.evaluate import (
    evaluate_manifest,
    evaluate_sequence,
    match_frame,
)
from smart_livestock_gate.tracking.schema import TrackedBox


def _gt_record(frame_index, track_uid, box_xywh, sequence="01.mp4"):
    return {
        "sequence": sequence,
        "frame_index": frame_index,
        "track_uid": track_uid,
        "bbox_xywh": box_xywh,
    }


def _tracked_box(frame_index, track_id, box_xyxy):
    return TrackedBox(
        frame_index=frame_index,
        timestamp=None,
        track_id=track_id,
        box_xyxy=box_xyxy,
        confidence=0.9,
    )


def test_match_frame_pairs_overlapping_boxes():
    ground_truth = [("01.mp4:1", [0.0, 0.0, 10.0, 10.0])]
    predictions = [(5, [1.0, 1.0, 11.0, 11.0])]

    matches = match_frame(ground_truth, predictions)

    assert len(matches) == 1
    track_uid, track_id, iou = matches[0]
    assert track_uid == "01.mp4:1"
    assert track_id == 5
    assert iou > 0.5


def test_evaluate_sequence_counts_id_switch():
    ground_truth_records = [
        _gt_record(0, "01.mp4:1", [0.0, 0.0, 10.0, 10.0]),
        _gt_record(1, "01.mp4:1", [0.0, 0.0, 10.0, 10.0]),
        _gt_record(2, "01.mp4:1", [0.0, 0.0, 10.0, 10.0]),
    ]
    predicted_tracks = [
        _tracked_box(0, 5, [0.0, 0.0, 10.0, 10.0]),
        _tracked_box(1, 5, [0.0, 0.0, 10.0, 10.0]),
        _tracked_box(2, 9, [0.0, 0.0, 10.0, 10.0]),
    ]

    evaluation = evaluate_sequence(ground_truth_records, predicted_tracks)

    assert evaluation.id_switches == 1
    assert evaluation.fragmentations == 0
    assert evaluation.matched_detections == 3


def test_evaluate_sequence_counts_fragmentation():
    ground_truth_records = [
        _gt_record(0, "01.mp4:1", [0.0, 0.0, 10.0, 10.0]),
        _gt_record(1, "01.mp4:1", [0.0, 0.0, 10.0, 10.0]),
        _gt_record(2, "01.mp4:1", [0.0, 0.0, 10.0, 10.0]),
    ]
    predicted_tracks = [
        _tracked_box(0, 5, [0.0, 0.0, 10.0, 10.0]),
        # frame 1: no prediction, ground truth goes unmatched (occlusion)
        _tracked_box(2, 5, [0.0, 0.0, 10.0, 10.0]),
    ]

    evaluation = evaluate_sequence(ground_truth_records, predicted_tracks)

    assert evaluation.fragmentations == 1
    assert evaluation.id_switches == 0
    assert evaluation.missed_detections == 1


def test_evaluate_manifest_reads_jsonl_and_aggregates(tmp_path):
    manifest_path = tmp_path / "tracking_manifest.jsonl"
    records = [
        _gt_record(0, "01.mp4:1", [0.0, 0.0, 10.0, 10.0]),
        _gt_record(1, "01.mp4:1", [0.0, 0.0, 10.0, 10.0]),
    ]
    with manifest_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    predictions_by_sequence = {
        "01.mp4": [
            _tracked_box(0, 5, [0.0, 0.0, 10.0, 10.0]),
            _tracked_box(1, 5, [0.0, 0.0, 10.0, 10.0]),
        ]
    }

    report = evaluate_manifest(manifest_path, predictions_by_sequence)

    assert report["aggregate"]["matched_detections"] == 2
    assert report["aggregate"]["id_switches"] == 0
    assert report["sequences"]["01.mp4"]["ground_truth_tracks"] == 1
