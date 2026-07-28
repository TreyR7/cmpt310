import argparse
import json
from dataclasses import asdict
from pathlib import Path

from smart_livestock_gate.baseline.classifier import (
    evaluate_model,
    load_dataset,
    load_model,
    predict_paths,
    save_model,
)
from smart_livestock_gate.config import (
    ARTIFACTS_DIR,
    CATTLE_DETECTION_DATASET_YAML,
    CATTLE_DETECTOR_MODEL_PATH,
    CATTLE_DETECTOR_REPORT_PATH,
    CATTLE_EYE_VIEW_PATH,
    CATTLE_TRACKING_MANIFEST_PATH,
    CATTLE_TRACKING_REPORT_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    PROCESSED_DATA_DIR,
)
from smart_livestock_gate.counting.tally import tally_predictions
from smart_livestock_gate.data.cattle_eye_view import (
    YOLO_TASKS,
    export_tracking_manifest,
    prepare_yolo_dataset,
    validate_dataset,
)
from smart_livestock_gate.data.preprocessing import IMAGE_EXTENSIONS, image_paths
from smart_livestock_gate.detection.detector import (
    DetectorTrainingConfig,
    train_detector,
)
from smart_livestock_gate.tracking.runner import TrackingRunConfig, run_tracking


def train(args: argparse.Namespace) -> None:
    max_per_class = None if args.max_per_class == 0 else args.max_per_class
    features, labels = load_dataset(
        args.dataset, max_per_class=max_per_class, seed=args.seed
    )
    print(
        f"\nFeature matrix: {features.shape[0]} images x "
        f"{features.shape[1]} features"
    )
    model = evaluate_model(
        features,
        labels,
        neighbors=args.neighbors,
        test_size=args.test_size,
        seed=args.seed,
    )
    # The metrics above stay honest because they came from unseen holdout data.
    # Refit afterward so the saved model can learn from every available sample.
    model.fit(features, labels)
    save_model(model, args.model)
    print(f"\nRefit on all {len(labels)} images and saved model to {args.model}")


def predict(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    if input_path.is_dir():
        paths = image_paths(input_path)
    elif input_path.is_file() and input_path.suffix.lower() in IMAGE_EXTENSIONS:
        paths = [input_path]
    else:
        raise ValueError(f"Input must be a supported image or directory: {input_path}")

    results = predict_paths(load_model(args.model), paths)
    for path, label in results:
        print(f"{path.name}: {label}")

    print("\nTally")
    for label, count in tally_predictions(label for _, label in results).items():
        print(f"{label}: {count}")


def validate_cattle_eye_view(args: argparse.Namespace) -> None:
    """Validate the extracted CattleEyeView release."""
    report = validate_dataset(
        CATTLE_EYE_VIEW_PATH,
        validate_labels=not args.skip_labels,
        label_sample_limit=None if args.deep_labels else 25,
    )
    print(json.dumps(report.to_dict(), indent=2))
    if not report.valid:
        raise SystemExit(1)


def export_cattle_eye_view(args: argparse.Namespace) -> None:
    """Export normalized tracking and sequence manifests."""
    output = args.output or PROCESSED_DATA_DIR / "cattle_eye_view"
    results = export_tracking_manifest(CATTLE_EYE_VIEW_PATH, output)
    print(json.dumps(results, indent=2))


def prepare_cattle_eye_view(args: argparse.Namespace) -> None:
    """Build an Ultralytics-compatible task directory using links or copies."""
    prepared = prepare_yolo_dataset(
        CATTLE_EYE_VIEW_PATH,
        args.task,
        args.output,
        link_mode=args.link_mode,
    )
    print(json.dumps(asdict(prepared), indent=2))


def train_cattle_detector(args: argparse.Namespace) -> None:
    """Train and evaluate the first cattle detection model."""
    report = train_detector(
        DetectorTrainingConfig(
            dataset_yaml=args.dataset,
            output_model=args.output_model,
            report_path=args.report,
            runs_dir=args.runs_dir,
            pretrained_model=args.pretrained_model,
            epochs=args.epochs,
            image_size=args.image_size,
            batch_size=args.batch_size,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            seed=args.seed,
            run_name=args.run_name,
        )
    )
    print(json.dumps(report, indent=2))


def track_video(args: argparse.Namespace) -> None:
    """Run the cattle tracker over one sequence and optionally score it."""
    manifest = args.manifest if args.evaluate else None
    report = run_tracking(
        TrackingRunConfig(
            sequence=args.sequence,
            dataset_root=CATTLE_EYE_VIEW_PATH,
            model_path=args.model,
            report_path=args.report,
            manifest_path=manifest,
            confidence=args.confidence,
            image_size=args.image_size,
            device=args.device,
            iou_threshold=args.iou_threshold,
            max_age=args.max_age,
            min_hits=args.min_hits,
        )
    )
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify farm animal images using OpenCV features and KNN."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train and evaluate a model")
    train_parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    train_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    train_parser.add_argument("--neighbors", type=int, default=5)
    train_parser.add_argument(
        "--max-per-class",
        type=int,
        default=500,
        help="images sampled per class; use 0 for the full dataset",
    )
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.set_defaults(func=train)

    predict_parser = subparsers.add_parser(
        "predict", help="classify an image or directory and print a tally"
    )
    predict_parser.add_argument("input")
    predict_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    predict_parser.set_defaults(func=predict)

    dataset_parser = subparsers.add_parser(
        "dataset",
        help="validate and prepare CattleEyeView data",
    )
    dataset_subparsers = dataset_parser.add_subparsers(
        dest="dataset_command",
        required=True,
    )
    validate_parser = dataset_subparsers.add_parser(
        "validate",
        help="validate frames, videos, labels, tracking IDs, and counts",
    )
    validate_parser.add_argument(
        "--skip-labels",
        action="store_true",
        help="skip the slower native YOLO label-content checks",
    )
    validate_parser.add_argument(
        "--deep-labels",
        action="store_true",
        help="parse every polygon/keypoint value instead of sampling each split",
    )
    validate_parser.set_defaults(func=validate_cattle_eye_view)

    export_parser = dataset_subparsers.add_parser(
        "export",
        help="export normalized tracking JSONL and sequence metadata",
    )
    export_parser.add_argument(
        "--output",
        type=Path,
        help="output directory; defaults to data/processed/cattle_eye_view",
    )
    export_parser.set_defaults(func=export_cattle_eye_view)

    prepare_parser = dataset_subparsers.add_parser(
        "prepare-yolo",
        help="build a valid Ultralytics image/label layout",
    )
    prepare_parser.add_argument("task", choices=YOLO_TASKS)
    prepare_parser.add_argument(
        "--output",
        type=Path,
        help="output directory; defaults to <root>/prepared/<task>",
    )
    prepare_parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="hardlink avoids duplicating the large image data",
    )
    prepare_parser.set_defaults(func=prepare_cattle_eye_view)

    detector_parser = subparsers.add_parser(
        "train-detector",
        help="fine-tune and test a cattle detector on CattleEyeView",
    )
    detector_parser.add_argument(
        "--dataset",
        type=Path,
        default=CATTLE_DETECTION_DATASET_YAML,
    )
    detector_parser.add_argument("--pretrained-model", default="yolo11n.pt")
    detector_parser.add_argument("--epochs", type=int, default=20)
    detector_parser.add_argument("--image-size", type=int, default=512)
    detector_parser.add_argument("--batch-size", type=int, default=4)
    detector_parser.add_argument("--device", default="auto")
    detector_parser.add_argument("--workers", type=int, default=2)
    detector_parser.add_argument("--patience", type=int, default=8)
    detector_parser.add_argument("--seed", type=int, default=42)
    detector_parser.add_argument("--run-name", default="cattle_detector")
    detector_parser.add_argument(
        "--runs-dir",
        type=Path,
        default=ARTIFACTS_DIR / "training",
    )
    detector_parser.add_argument(
        "--output-model",
        type=Path,
        default=CATTLE_DETECTOR_MODEL_PATH,
    )
    detector_parser.add_argument(
        "--report",
        type=Path,
        default=CATTLE_DETECTOR_REPORT_PATH,
    )
    detector_parser.set_defaults(func=train_cattle_detector)

    track_parser = subparsers.add_parser(
        "track",
        help="run the cattle tracker over one CattleEyeView sequence",
    )
    track_parser.add_argument("sequence", help='e.g. "01.mp4"')
    track_parser.add_argument("--model", type=Path, default=CATTLE_DETECTOR_MODEL_PATH)
    track_parser.add_argument("--confidence", type=float, default=0.25)
    track_parser.add_argument("--image-size", type=int, default=512)
    track_parser.add_argument("--device", default="auto")
    track_parser.add_argument("--iou-threshold", type=float, default=0.3)
    track_parser.add_argument("--max-age", type=int, default=5)
    track_parser.add_argument("--min-hits", type=int, default=3)
    track_parser.add_argument(
        "--manifest",
        type=Path,
        default=CATTLE_TRACKING_MANIFEST_PATH,
        help="ground-truth manifest for scoring; see --no-evaluate",
    )
    track_parser.add_argument(
        "--no-evaluate",
        dest="evaluate",
        action="store_false",
        default=True,
        help="skip scoring against the ground-truth manifest",
    )
    track_parser.add_argument(
        "--report",
        type=Path,
        default=CATTLE_TRACKING_REPORT_PATH,
    )
    track_parser.set_defaults(func=track_video)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
