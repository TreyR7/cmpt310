"""Cattle detection training and inference services."""

from smart_livestock_gate.detection.detector import (
    DetectorTrainingConfig,
    predict_image,
    train_detector,
)

__all__ = ["DetectorTrainingConfig", "predict_image", "train_detector"]
