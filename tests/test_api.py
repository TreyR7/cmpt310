import json

from smart_livestock_gate.api.app import create_app
from smart_livestock_gate.api.status import build_project_status
from smart_livestock_gate.api.tracking import render_sequence_frame


def test_health_endpoint():
    client = create_app().test_client()

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_project_status_handles_missing_dataset(tmp_path):
    status = build_project_status(
        tmp_path / "missing",
        tmp_path / "report.json",
        tmp_path / "detector.pt",
    )

    assert not status["dataset"]["installed"]
    assert status["dataset"]["validation"] == "dataset_missing"
    assert status["pipeline"]["next_step"] == "install_dataset"


def test_detection_examples_endpoint_handles_missing_local_data():
    client = create_app().test_client()

    response = client.get("/api/detection/examples")

    assert response.status_code == 200
    assert "examples" in response.get_json()


def test_tracking_overview_endpoint_returns_ready_flag():
    client = create_app().test_client()

    response = client.get("/api/tracking/overview")

    assert response.status_code == 200
    body = response.get_json()
    assert "ready" in body
    assert "demo_available" in body


def test_render_sequence_frame_rejects_unknown_sequence(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "images" / "01.mp4").mkdir(parents=True)

    assert render_sequence_frame(dataset, "../secrets", 0) is None
    assert render_sequence_frame(dataset, "99.mp4", 0) is None
    assert render_sequence_frame(dataset, "01.mp4", -1) is None
    # Known sequence but no such frame on disk.
    assert render_sequence_frame(dataset, "01.mp4", 5) is None


def test_render_sequence_frame_downscales_existing_frame(tmp_path):
    import cv2
    import numpy as np

    dataset = tmp_path / "dataset"
    frame_dir = dataset / "images" / "01.mp4"
    frame_dir.mkdir(parents=True)
    # frame_index 0 maps to the 1-based file name 00001.jpg
    cv2.imwrite(
        str(frame_dir / "00001.jpg"),
        np.full((1080, 1920, 3), 128, dtype=np.uint8),
    )

    jpeg = render_sequence_frame(dataset, "01.mp4", 0, width=320)

    assert jpeg is not None
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[1] == 320
    assert decoded.shape[0] == 180


def test_tracking_frame_endpoint_returns_404_for_unknown_sequence():
    client = create_app().test_client()

    response = client.get("/api/tracking/frames/nope.mp4/0")

    assert response.status_code == 404


def test_project_status_uses_validated_report_without_exposing_paths(tmp_path):
    dataset = tmp_path / "dataset"
    for directory in ("images", "videos", "annotation"):
        (dataset / directory).mkdir(parents=True)
    (dataset / "annotation" / "detect" / "labels").mkdir(parents=True)
    prepared = dataset / "prepared" / "detect"
    prepared.mkdir(parents=True)
    (prepared / "dataset.yaml").touch()

    report = tmp_path / "validation_report.json"
    report.write_text(
        json.dumps(
            {
                "root": "private/local/path",
                "valid": True,
                "summary": {"frames": 30_703, "sequences": 14},
                "warnings": ["normalized source IDs"],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )

    status = build_project_status(dataset, report, tmp_path / "detector.pt")
    payload = json.dumps(status)

    assert status["dataset"]["ready"]
    assert status["dataset"]["summary"]["frames"] == 30_703
    assert status["training_tasks"]["detect"]["prepared"]
    assert status["pipeline"]["next_step"] == "train_detector"
    assert "private/local/path" not in payload
