"""Score a tracker's output against the CattleEyeView ground-truth manifest."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from smart_livestock_gate.tracking.association import associate, iou_matrix
from smart_livestock_gate.tracking.schema import TrackedBox


@dataclass(frozen=True)
class SequenceEvaluation:
    """Tracking-quality metrics for one video sequence."""

    sequence: str
    frames_evaluated: int
    ground_truth_tracks: int
    predicted_tracks: int
    matched_detections: int
    missed_detections: int
    false_positives: int
    id_switches: int
    fragmentations: int
    mean_iou: float


def _bbox_xywh_to_xyxy(bbox_xywh: list[float]) -> list[float]:
    x, y, width, height = bbox_xywh
    return [x, y, x + width, y + height]


def match_frame(
    ground_truth: list[tuple[str, list[float]]],
    predictions: list[tuple[int, list[float]]],
    iou_threshold: float = 0.5,
) -> list[tuple[str, int, float]]:
    """Match one frame's ground-truth track_uids to predicted track_ids by IoU.

    Returns a list of (track_uid, track_id, iou) for matched pairs only.
    """
    if not ground_truth or not predictions:
        return []

    gt_boxes = np.array([box for _, box in ground_truth], dtype=np.float32)
    pred_boxes = np.array([box for _, box in predictions], dtype=np.float32)

    matches, _, _ = associate(gt_boxes, pred_boxes, iou_threshold=iou_threshold)
    ious = iou_matrix(gt_boxes, pred_boxes)
    return [
        (
            ground_truth[gt_index][0],
            predictions[pred_index][0],
            float(ious[gt_index, pred_index]),
        )
        for gt_index, pred_index in matches
    ]


def evaluate_sequence(
    ground_truth_records: list[dict],
    predicted_tracks: list[TrackedBox],
    iou_threshold: float = 0.5,
) -> SequenceEvaluation:
    """Compare one sequence's ground truth against its predicted tracks."""
    gt_by_frame: dict[int, list[tuple[str, list[float]]]] = defaultdict(list)
    for record in ground_truth_records:
        gt_by_frame[record["frame_index"]].append(
            (record["track_uid"], _bbox_xywh_to_xyxy(record["bbox_xywh"]))
        )

    pred_by_frame: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    for track in predicted_tracks:
        pred_by_frame[track.frame_index].append((track.track_id, track.box_xyxy))

    all_frames = sorted(set(gt_by_frame) | set(pred_by_frame))

    last_matched_id: dict[str, int] = {}
    was_matched: dict[str, bool] = {}
    id_switches = 0
    fragmentations = 0
    matched_detections = 0
    missed_detections = 0
    false_positives = 0
    iou_values: list[float] = []

    for frame_index in all_frames:
        gt_entries = gt_by_frame.get(frame_index, [])
        pred_entries = pred_by_frame.get(frame_index, [])
        matches = match_frame(gt_entries, pred_entries, iou_threshold=iou_threshold)
        matched_gt_uids = {track_uid for track_uid, _, _ in matches}

        matched_detections += len(matches)
        missed_detections += len(gt_entries) - len(matches)
        false_positives += len(pred_entries) - len(matches)

        for track_uid, track_id, iou in matches:
            iou_values.append(iou)
            previous_id = last_matched_id.get(track_uid)
            if previous_id is not None and previous_id != track_id:
                id_switches += 1
            if was_matched.get(track_uid) is False:
                fragmentations += 1
            last_matched_id[track_uid] = track_id
            was_matched[track_uid] = True

        for track_uid, _ in gt_entries:
            if track_uid not in matched_gt_uids and track_uid in was_matched:
                was_matched[track_uid] = False

    ground_truth_uids = {
        uid for entries in gt_by_frame.values() for uid, _ in entries
    }
    predicted_ids = {
        track_id for entries in pred_by_frame.values() for track_id, _ in entries
    }

    return SequenceEvaluation(
        sequence=ground_truth_records[0]["sequence"] if ground_truth_records else "",
        frames_evaluated=len(all_frames),
        ground_truth_tracks=len(ground_truth_uids),
        predicted_tracks=len(predicted_ids),
        matched_detections=matched_detections,
        missed_detections=missed_detections,
        false_positives=false_positives,
        id_switches=id_switches,
        fragmentations=fragmentations,
        mean_iou=round(float(np.mean(iou_values)), 6) if iou_values else 0.0,
    )


def evaluate_manifest(
    manifest_path: Path,
    predictions_by_sequence: dict[str, list[TrackedBox]],
    iou_threshold: float = 0.5,
) -> dict:
    """Read the exported tracking manifest and score it against predictions."""
    records_by_sequence: dict[str, list[dict]] = defaultdict(list)
    with Path(manifest_path).open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            records_by_sequence[record["sequence"]].append(record)

    sequences = {}
    for sequence, predictions in predictions_by_sequence.items():
        ground_truth_records = records_by_sequence.get(sequence, [])
        evaluation = evaluate_sequence(
            ground_truth_records, predictions, iou_threshold=iou_threshold
        )
        sequences[sequence] = evaluation.__dict__

    total_id_switches = sum(item["id_switches"] for item in sequences.values())
    total_fragmentations = sum(item["fragmentations"] for item in sequences.values())
    total_matched = sum(item["matched_detections"] for item in sequences.values())
    total_missed = sum(item["missed_detections"] for item in sequences.values())
    total_false_positives = sum(item["false_positives"] for item in sequences.values())

    return {
        "sequences": sequences,
        "aggregate": {
            "id_switches": total_id_switches,
            "fragmentations": total_fragmentations,
            "matched_detections": total_matched,
            "missed_detections": total_missed,
            "false_positives": total_false_positives,
        },
    }
