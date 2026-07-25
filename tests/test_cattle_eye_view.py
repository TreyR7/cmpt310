import json
import os
from pathlib import Path

from openpyxl import Workbook

from smart_livestock_gate.data.cattle_eye_view import (
    export_tracking_manifest,
    iter_tracking_records,
    load_sequence_metadata,
    prepare_yolo_dataset,
    validate_dataset,
)


def _create_test_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "CattleEyeView"
    image = root / "images" / "01.mp4" / "00001.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    video = root / "videos" / "01.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")

    annotation_root = root / "annotation"
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append(
        [
            "video_filename",
            "Date",
            "Time Start",
            "Time End",
            "Frames",
            "Split",
            "Count",
        ]
    )
    worksheet.append(
        ["01.mp4", "2021-01-01", "07:00:00", "07:01:00", 1, "Train", 1]
    )
    annotation_root.mkdir(parents=True)
    workbook.save(annotation_root / "metadata and count.xlsx")

    coco_root = annotation_root / "pose_COCO"
    coco_root.mkdir()
    images = [
        {
            "id": 0,
            "file_name": "01.mp4/00001.jpg",
            "video_id": 1,
            "frame_id": 0,
            "width": 1920,
            "height": 1080,
        }
    ]
    annotation = {
        "id": 5,
        "file_name": "01.mp4/00001.jpg",
        "video_id": 1,
        "image_id": 0,
        "category_id": 1,
        "instance_id": 42,
        "bbox": [10, 20, 30, 40],
        "bbox_head": float("nan"),
        "keypoints": [0, 0, 0] * 24,
        "num_keypoints": 0,
        "occluded": False,
        "truncated": False,
        "split": "train",
    }
    common = {"videos": [{"video_id": 1, "name": "01.mp4"}], "images": images}
    (coco_root / "coco_track_train.json").write_text(
        json.dumps({**common, "annotations": [annotation]}),
        encoding="utf-8",
    )
    (coco_root / "coco_track_test.json").write_text(
        json.dumps({**common, "annotations": []}),
        encoding="utf-8",
    )

    labels = {
        "detect": "0 0.5 0.5 0.25 0.25\n",
        "detect_head": "0 0.5 0.5 0.1 0.1\n",
        "pose": " ".join(
            ["0", "0.5", "0.5", "0.25", "0.25"] + ["0", "0", "0"] * 24
        )
        + "\n",
        "segment": "0 0.1 0.1 0.2 0.1 0.2 0.2\n",
    }
    for task, content in labels.items():
        for split in ("train", "val", "test"):
            split_root = annotation_root / task / "labels" / split
            split_root.mkdir(parents=True)
        label = annotation_root / task / "labels" / "train" / "01.mp4" / "00001.txt"
        label.parent.mkdir()
        label.write_text(content, encoding="utf-8")
    return root


def test_loads_metadata_and_normalizes_tracking_record(tmp_path):
    root = _create_test_dataset(tmp_path)

    metadata = load_sequence_metadata(root)
    record = next(iter_tracking_records(root, splits=("train",)))

    assert metadata[0].video_filename == "01.mp4"
    assert metadata[0].count == 1
    assert record.track_uid == "01.mp4:42"
    assert record.image == "images/01.mp4/00001.jpg"
    assert record.head_bbox_xywh is None


def test_validates_consistent_fixture(tmp_path):
    root = _create_test_dataset(tmp_path)

    report = validate_dataset(root)

    assert report.valid
    assert report.summary["frames"] == 1
    assert report.summary["unique_tracks"] == 1
    assert report.summary["task_label_files"]["pose"]["train"] == 1


def test_exports_normalized_tracking_manifest(tmp_path):
    root = _create_test_dataset(tmp_path)
    output = tmp_path / "processed"

    paths = export_tracking_manifest(root, output)
    record = json.loads((output / "tracking_manifest.jsonl").read_text().strip())

    assert Path(paths["sequences"]).is_file()
    assert record["track_uid"] == "01.mp4:42"
    assert record["head_bbox_xywh"] is None


def test_prepares_hardlinked_ultralytics_dataset(tmp_path):
    root = _create_test_dataset(tmp_path)
    output = tmp_path / "prepared"

    result = prepare_yolo_dataset(root, "detect", output)

    prepared_image = output / "images" / "train" / "01.mp4" / "00001.jpg"
    source_image = root / "images" / "01.mp4" / "00001.jpg"
    assert result.split_images == {"train": 1, "val": 0, "test": 0}
    assert os.path.samefile(source_image, prepared_image)
    assert (output / "dataset.yaml").is_file()
    assert (output / "manifest.csv").is_file()
