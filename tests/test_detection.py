from smart_livestock_gate.detection.detector import DetectorTrainingConfig
from smart_livestock_gate.detection.examples import list_detection_examples


def test_training_config_is_path_explicit(tmp_path):
    config = DetectorTrainingConfig(
        dataset_yaml=tmp_path / "dataset.yaml",
        output_model=tmp_path / "detector.pt",
        report_path=tmp_path / "metrics.json",
        runs_dir=tmp_path / "runs",
    )

    assert config.epochs == 20
    assert config.image_size == 512
    assert config.output_model.name == "detector.pt"


def test_lists_allowlisted_detection_example(tmp_path):
    root = tmp_path / "dataset"
    prepared = root / "prepared" / "detect"
    image = prepared / "images" / "test" / "01.mp4" / "00010.jpg"
    label = prepared / "labels" / "test" / "01.mp4" / "00010.txt"
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    image.write_bytes(b"example")
    label.write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")

    examples = list_detection_examples(root)

    assert len(examples) == 1
    assert examples[0].sequence == "01.mp4"
    assert examples[0].ground_truth[0]["box"] == [0.4, 0.3, 0.6, 0.7]
    assert "00010" not in examples[0].id
