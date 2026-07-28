"""Project paths shared by the CLI and API."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models"

DEFAULT_DATASET_PATH = RAW_DATA_DIR / "animal_counting_dataset"
DEFAULT_MODEL_PATH = MODEL_DIR / "animal_knn.joblib"
CATTLE_EYE_VIEW_PATH = RAW_DATA_DIR / "cattle_eye_view"
CATTLE_EYE_VIEW_REPORT_PATH = (
    PROCESSED_DATA_DIR / "cattle_eye_view" / "validation_report.json"
)
CATTLE_DETECTOR_MODEL_PATH = MODEL_DIR / "cattle_detector.pt"
CATTLE_DETECTOR_REPORT_PATH = (
    ARTIFACTS_DIR / "reports" / "cattle_detector" / "metrics.json"
)
CATTLE_DETECTION_DATASET_YAML = (
    CATTLE_EYE_VIEW_PATH / "prepared" / "detect" / "dataset.yaml"
)
CATTLE_TRACKING_MANIFEST_PATH = (
    PROCESSED_DATA_DIR / "cattle_eye_view" / "tracking_manifest.jsonl"
)
CATTLE_TRACKING_REPORT_PATH = (
    ARTIFACTS_DIR / "reports" / "cattle_tracking" / "metrics.json"
)
