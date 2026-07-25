"""Read-only project readiness information for API clients."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smart_livestock_gate.data.cattle_eye_view import (
    YOLO_TASKS,
    CattleEyeViewPaths,
)

SUMMARY_FIELDS = (
    "sequences",
    "frames",
    "videos",
    "tracking_annotations",
    "unique_tracks",
    "ground_truth_crossings",
    "sequence_splits",
    "task_label_files",
)


def _read_validation_report(report_path: Path) -> dict[str, Any] | None:
    if not report_path.is_file():
        return None
    try:
        document = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return document if isinstance(document, dict) else None


def build_project_status(
    dataset_root: Path,
    report_path: Path,
    detector_model_path: Path,
    detector_report_path: Path | None = None,
) -> dict[str, Any]:
    """Build a path-free status payload suitable for the web frontend."""
    paths = CattleEyeViewPaths.from_root(dataset_root)
    required_directories = (paths.images, paths.videos, paths.annotations)
    installed = all(path.is_dir() for path in required_directories)
    report = _read_validation_report(report_path) if installed else None
    summary = report.get("summary", {}) if report else {}
    safe_summary = {
        field: summary[field]
        for field in SUMMARY_FIELDS
        if field in summary
    }

    if not installed:
        validation = "dataset_missing"
    elif report is None:
        validation = "not_validated"
    elif report.get("valid") is True:
        validation = "valid"
    else:
        validation = "invalid"

    tasks = {}
    for task in YOLO_TASKS:
        labels_available = paths.task_labels(task).is_dir()
        prepared = (paths.root / "prepared" / task / "dataset.yaml").is_file()
        tasks[task] = {
            "labels_available": labels_available,
            "prepared": prepared,
        }

    detector_ready = detector_model_path.is_file()
    dataset_ready = installed and validation == "valid"
    if not installed:
        next_step = "install_dataset"
    elif validation == "not_validated":
        next_step = "validate_dataset"
    elif validation == "invalid":
        next_step = "repair_dataset"
    elif not tasks["detect"]["prepared"]:
        next_step = "prepare_detection_data"
    elif not detector_ready:
        next_step = "train_detector"
    else:
        next_step = "build_tracking_pipeline"

    warnings = report.get("warnings", []) if report else []
    errors = report.get("errors", []) if report else []
    detector_report = (
        _read_validation_report(detector_report_path)
        if detector_report_path is not None and detector_ready
        else None
    )
    return {
        "dataset": {
            "installed": installed,
            "ready": dataset_ready,
            "validation": validation,
            "summary": safe_summary,
            "warnings": warnings,
            "errors": errors,
        },
        "training_tasks": tasks,
        "models": {
            "cattle_detector": {
                "ready": detector_ready,
                "architecture": (
                    detector_report.get("model", {}).get("architecture")
                    if detector_report
                    else None
                ),
                "test_metrics": (
                    detector_report.get("test_metrics", {})
                    if detector_report
                    else {}
                ),
            }
        },
        "pipeline": {
            "dataset_ready": dataset_ready,
            "detector_ready": detector_ready,
            "tracking_ready": False,
            "counting_ready": False,
            "video_inference_ready": False,
            "next_step": next_step,
        },
    }
