"""IoU-based cost matrix and Hungarian matching between tracks and detections."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(tracks_xyxy: np.ndarray, detections_xyxy: np.ndarray) -> np.ndarray:
    """Return pairwise IoU, shape (n_tracks, n_detections)."""
    if len(tracks_xyxy) == 0 or len(detections_xyxy) == 0:
        return np.zeros((len(tracks_xyxy), len(detections_xyxy)), dtype=np.float32)

    tracks = tracks_xyxy[:, None, :]
    detections = detections_xyxy[None, :, :]

    left = np.maximum(tracks[..., 0], detections[..., 0])
    top = np.maximum(tracks[..., 1], detections[..., 1])
    right = np.minimum(tracks[..., 2], detections[..., 2])
    bottom = np.minimum(tracks[..., 3], detections[..., 3])

    intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
    track_width = np.maximum(0.0, tracks[..., 2] - tracks[..., 0])
    track_height = np.maximum(0.0, tracks[..., 3] - tracks[..., 1])
    track_area = track_width * track_height

    detection_width = np.maximum(0.0, detections[..., 2] - detections[..., 0])
    detection_height = np.maximum(0.0, detections[..., 3] - detections[..., 1])
    detection_area = detection_width * detection_height
    union = track_area + detection_area - intersection
    return np.where(union > 0, intersection / union, 0.0).astype(np.float32)


def associate(
    tracks_xyxy: np.ndarray,
    detections_xyxy: np.ndarray,
    iou_threshold: float = 0.3,
    appearance_cost: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match tracks to detections by Hungarian assignment on an IoU-based cost.

    Returns (matches, unmatched_track_indices, unmatched_detection_indices).
    ``matches`` is an (n, 2) array of (track_index, detection_index) pairs.
    ``appearance_cost``, if given, is an (n_tracks, n_detections) matrix added
    to the IoU cost before assignment; reserved for a future re-identification
    term and unused when None.
    """
    n_tracks = len(tracks_xyxy)
    n_detections = len(detections_xyxy)

    if n_tracks == 0 or n_detections == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(n_tracks),
            np.arange(n_detections),
        )

    ious = iou_matrix(tracks_xyxy, detections_xyxy)
    cost = 1.0 - ious
    if appearance_cost is not None:
        cost = cost + appearance_cost

    track_indices, detection_indices = linear_sum_assignment(cost)

    matches = []
    unmatched_tracks = set(range(n_tracks))
    unmatched_detections = set(range(n_detections))
    assigned_pairs = zip(track_indices, detection_indices, strict=True)
    for track_index, detection_index in assigned_pairs:
        if ious[track_index, detection_index] < iou_threshold:
            continue
        matches.append((track_index, detection_index))
        unmatched_tracks.discard(track_index)
        unmatched_detections.discard(detection_index)

    if matches:
        matches_array = np.array(matches, dtype=int)
    else:
        matches_array = np.empty((0, 2), dtype=int)
    return (
        matches_array,
        np.array(sorted(unmatched_tracks), dtype=int),
        np.array(sorted(unmatched_detections), dtype=int),
    )
