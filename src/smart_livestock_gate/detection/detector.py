"""Ultralytics cattle detector training, evaluation, and inference."""

from __future__ import annotations

import json
import platform
import shutil
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2


@dataclass(frozen=True)
class DetectorTrainingConfig:
    """Reproducible settings for one detector experiment."""

    dataset_yaml: Path
    output_model: Path
    report_path: Path
    runs_dir: Path
    pretrained_model: str = "yolo11n.pt"
    epochs: int = 20
    image_size: int = 512
    batch_size: int = 4
    device: str = "auto"
    workers: int = 2
    patience: int = 8
    seed: int = 42
    run_name: str = "cattle_detector"


def resolve_device(requested: str) -> str:
    """Resolve the CLI's portable auto device to an Ultralytics device value."""
    if requested != "auto":
        return requested
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def _metric(value: Any) -> float | None:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _box_metrics(metrics: Any) -> dict[str, float | None]:
    box = getattr(metrics, "box", None)
    return {
        "precision": _metric(getattr(box, "mp", None)),
        "recall": _metric(getattr(box, "mr", None)),
        "map50": _metric(getattr(box, "map50", None)),
        "map75": _metric(getattr(box, "map75", None)),
        "map50_95": _metric(getattr(box, "map", None)),
    }


def train_detector(config: DetectorTrainingConfig) -> dict[str, Any]:
    """Fine-tune a detector, evaluate on the test split, and save a report."""
    if not config.dataset_yaml.is_file():
        raise FileNotFoundError(
            f"Detection dataset YAML not found: {config.dataset_yaml}"
        )
    if config.epochs < 1 or config.image_size < 128 or config.batch_size < 1:
        raise ValueError("epochs and batch must be positive; image size must be >= 128")

    import torch
    import ultralytics
    from ultralytics import YOLO

    device = resolve_device(config.device)
    config.runs_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model = YOLO(config.pretrained_model)
    training = model.train(
        data=str(config.dataset_yaml),
        epochs=config.epochs,
        imgsz=config.image_size,
        batch=config.batch_size,
        device=device,
        workers=config.workers,
        patience=config.patience,
        seed=config.seed,
        deterministic=True,
        project=str(config.runs_dir),
        name=config.run_name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    run_directory = Path(training.save_dir)
    best_model = run_directory / "weights" / "best.pt"
    if not best_model.is_file():
        raise RuntimeError(f"Training did not create expected weights: {best_model}")

    config.output_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_model, config.output_model)
    evaluation_model = YOLO(str(config.output_model))
    evaluation = evaluation_model.val(
        data=str(config.dataset_yaml),
        split="test",
        imgsz=config.image_size,
        batch=config.batch_size,
        device=device,
        workers=config.workers,
        project=str(config.runs_dir),
        name=f"{config.run_name}_test",
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    elapsed = round(time.perf_counter() - started, 2)
    report = {
        "model": {
            "architecture": config.pretrained_model,
            "task": "single-class cattle detection",
            "output": config.output_model.name,
        },
        "training": {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config).items()
                if key
                not in {"dataset_yaml", "output_model", "report_path", "runs_dir"}
            },
            "device": device,
            "elapsed_seconds": elapsed,
        },
        "test_metrics": _box_metrics(evaluation),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    clear_model_cache()
    return report


@lru_cache(maxsize=2)
def _load_model(model_path: str):
    from ultralytics import YOLO

    return YOLO(model_path)


def clear_model_cache() -> None:
    """Clear cached model instances after replacing detector weights."""
    _load_model.cache_clear()


def predict_image(
    model_path: Path,
    image_path: Path,
    *,
    confidence: float = 0.25,
    image_size: int = 512,
    device: str = "auto",
) -> dict[str, Any]:
    """Run cattle detection and return JSON-ready normalized boxes."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Detector weights not found: {model_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read example image: {image_path}")
    height, width = image.shape[:2]
    started = time.perf_counter()
    results = _load_model(str(model_path.resolve())).predict(
        source=image,
        conf=confidence,
        imgsz=image_size,
        device=resolve_device(device),
        verbose=False,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    detections = []
    if results:
        result = results[0]
        boxes = result.boxes
        for xyxy, score, class_id in zip(
            boxes.xyxyn.cpu().tolist(),
            boxes.conf.cpu().tolist(),
            boxes.cls.cpu().tolist(),
            strict=True,
        ):
            detections.append(
                {
                    "label": str(result.names[int(class_id)]),
                    "confidence": round(float(score), 4),
                    "box": [round(float(value), 6) for value in xyxy],
                }
            )
    return {
        "image": {"width": width, "height": height},
        "detections": detections,
        "count": len(detections),
        "inference_ms": elapsed_ms,
        "confidence_threshold": confidence,
    }
