import json

from smart_livestock_gate.api.app import create_app
from smart_livestock_gate.api.status import build_project_status


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
