"""Evaluate the SORT tracker on the held-out CattleEyeView test sequences.

Runs detection + tracking over each sequence's extracted frames (whose numbering
matches the ground-truth manifest), scores identities, and writes one combined
metrics report plus per-sequence JSONL track exports under artifacts/.

Usage:
    python scripts/evaluate_tracking.py 01.mp4 05.mp4 07.mp4 10.mp4
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from smart_livestock_gate.config import (
    CATTLE_DETECTOR_MODEL_PATH,
    CATTLE_EYE_VIEW_PATH,
    CATTLE_TRACKING_MANIFEST_PATH,
)
from smart_livestock_gate.tracking.runner import TrackingRunConfig, run_tracking

CONFIG = {
    "confidence": 0.25,
    "iou_threshold": 0.3,
    "min_hits": 3,
    "max_age": 5,
    "image_size": 512,
}
OUTPUT_DIR = Path("artifacts/reports/cattle_tracking")


def main(sequences: list[str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_sequence = {}
    started = time.perf_counter()
    for sequence in sequences:
        print(f"[evaluate] tracking {sequence} ...", flush=True)
        report = run_tracking(
            TrackingRunConfig(
                sequence=sequence,
                dataset_root=CATTLE_EYE_VIEW_PATH,
                model_path=CATTLE_DETECTOR_MODEL_PATH,
                report_path=OUTPUT_DIR / f"{sequence}_report.json",
                detections_path=OUTPUT_DIR / f"{sequence}_tracks.jsonl",
                manifest_path=CATTLE_TRACKING_MANIFEST_PATH,
                confidence=CONFIG["confidence"],
                iou_threshold=CONFIG["iou_threshold"],
                min_hits=CONFIG["min_hits"],
                max_age=CONFIG["max_age"],
                image_size=CONFIG["image_size"],
            )
        )
        per_sequence[sequence] = report
        ev = report.get("evaluation") or {}
        print(
            f"[evaluate] {sequence}: recall={ev.get('matched_track_recall')} "
            f"idsw={ev.get('id_switches')} frag={ev.get('fragmentations')} "
            f"MT={ev.get('mostly_tracked')}/{ev.get('ground_truth_tracks')} "
            f"fps={report['runtime']['processing_fps']}",
            flush=True,
        )

    def agg(field: str) -> int:
        return sum(r["evaluation"][field] for r in per_sequence.values())

    gt_det = agg("ground_truth_detections")
    matched = agg("matched_detections")
    mt = agg("mostly_tracked")
    gt_tracks = agg("ground_truth_tracks")
    combined = {
        "config": CONFIG,
        "evaluation_sequences": sequences,
        "note": (
            "Held-out test-split sequences, disjoint from 09.mp4 used during "
            "development tuning. Frame numbering matches the ground-truth manifest."
        ),
        "aggregate": {
            "ground_truth_tracks": gt_tracks,
            "ground_truth_detections": gt_det,
            "matched_detections": matched,
            "missed_detections": agg("missed_detections"),
            "false_positives": agg("false_positives"),
            "id_switches": agg("id_switches"),
            "fragmentations": agg("fragmentations"),
            "matched_track_recall": round(matched / gt_det, 4) if gt_det else 0.0,
            "mostly_tracked": mt,
            "mostly_tracked_ratio": round(mt / gt_tracks, 4) if gt_tracks else 0.0,
            "mean_processing_fps": round(
                sum(r["runtime"]["processing_fps"] for r in per_sequence.values())
                / len(per_sequence),
                2,
            ),
        },
        "per_sequence": {
            s: {**r["evaluation"], "runtime": r["runtime"]}
            for s, r in per_sequence.items()
        },
    }
    out = OUTPUT_DIR / "step2_metrics.json"
    out.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    elapsed = round(time.perf_counter() - started, 1)
    print(f"[evaluate] wrote {out} in {elapsed}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["01.mp4", "05.mp4", "07.mp4", "10.mp4"])
