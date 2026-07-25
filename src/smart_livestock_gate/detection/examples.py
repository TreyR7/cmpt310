"""Safe, deterministic test examples for the local detection UI."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from smart_livestock_gate.data.cattle_eye_view import CattleEyeViewPaths


@dataclass(frozen=True)
class DetectionExample:
    id: str
    sequence: str
    frame: str
    image_path: Path
    ground_truth: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "sequence": self.sequence,
            "frame": self.frame,
            "ground_truth": self.ground_truth,
            "ground_truth_count": len(self.ground_truth),
            "image_url": f"/api/detection/examples/{self.id}/image",
        }


def _read_yolo_boxes(label_path: Path) -> list[dict[str, object]]:
    def clamp(value: float) -> float:
        return round(min(1.0, max(0.0, value)), 6)

    boxes = []
    if not label_path.is_file():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if len(tokens) != 5:
            continue
        class_id, center_x, center_y, width, height = map(float, tokens)
        if class_id != 0:
            continue
        boxes.append(
            {
                "label": "cow",
                "box": [
                    clamp(center_x - width / 2),
                    clamp(center_y - height / 2),
                    clamp(center_x + width / 2),
                    clamp(center_y + height / 2),
                ],
            }
        )
    return boxes


def list_detection_examples(
    dataset_root: Path,
    *,
    limit: int = 5,
) -> list[DetectionExample]:
    """Select one labeled test frame per sequence, then fill up to the limit."""
    paths = CattleEyeViewPaths.from_root(dataset_root)
    prepared = paths.root / "prepared" / "detect"
    image_root = prepared / "images" / "test"
    label_root = prepared / "labels" / "test"
    if not image_root.is_dir() or not label_root.is_dir():
        return []

    selected_labels = []
    for sequence_root in sorted(path for path in label_root.iterdir() if path.is_dir()):
        labeled = sorted(
            path
            for path in sequence_root.rglob("*.txt")
            if path.stat().st_size > 0
        )
        if labeled:
            selected_labels.append(labeled[len(labeled) // 2])

    if len(selected_labels) < limit:
        all_labels = sorted(
            path for path in label_root.rglob("*.txt") if path.stat().st_size > 0
        )
        for label in all_labels:
            if label not in selected_labels:
                selected_labels.append(label)
            if len(selected_labels) >= limit:
                break

    examples = []
    for label_path in selected_labels[:limit]:
        relative = label_path.relative_to(label_root)
        image_path = image_root / relative.with_suffix(".jpg")
        if not image_path.is_file():
            continue
        identifier = hashlib.sha256(relative.as_posix().encode()).hexdigest()[:12]
        examples.append(
            DetectionExample(
                id=identifier,
                sequence=relative.parent.as_posix(),
                frame=image_path.name,
                image_path=image_path,
                ground_truth=_read_yolo_boxes(label_path),
            )
        )
    return examples


def find_detection_example(
    dataset_root: Path,
    example_id: str,
) -> DetectionExample | None:
    """Resolve an opaque example ID only through the server-generated allowlist."""
    return next(
        (
            example
            for example in list_detection_examples(dataset_root)
            if example.id == example_id
        ),
        None,
    )
