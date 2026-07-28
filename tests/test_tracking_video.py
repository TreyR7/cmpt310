import numpy as np

from smart_livestock_gate.tracking.sort_tracker import SortTracker, SortTrackerConfig
from smart_livestock_gate.tracking.video import track_sequence


def _fake_predict(boxes_in_order):
    calls = iter(boxes_in_order)

    def predict_fn(model_path, frame, *, confidence, image_size, device):
        height, width = frame.shape[:2]
        box = next(calls, None)
        detections = [{"label": "cow", "confidence": 0.9, "box": box}] if box else []
        return {
            "image": {"width": width, "height": height},
            "detections": detections,
            "count": len(detections),
            "inference_ms": 0.0,
            "confidence_threshold": confidence,
        }

    return predict_fn


def _fake_frame_source(frame_indices, width=100, height=100):
    for index in frame_indices:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        yield index, frame


def test_track_sequence_denormalizes_boxes_before_tracking():
    # normalized box covering the left half of a 100x100 frame, held steady
    box = [0.0, 0.0, 0.5, 0.5]
    predict_fn = _fake_predict([box, box, box])
    tracker = SortTracker(SortTrackerConfig(min_hits=1))

    tracks = track_sequence(
        model_path="unused",
        frame_source=_fake_frame_source(range(3)),
        tracker=tracker,
        predict_fn=predict_fn,
    )

    assert len(tracks) == 3
    for track in tracks:
        assert track.box_xyxy == [0.0, 0.0, 50.0, 50.0]


def test_track_sequence_yields_no_tracks_without_detections():
    predict_fn = _fake_predict([])
    tracker = SortTracker(SortTrackerConfig(min_hits=1))

    tracks = track_sequence(
        model_path="unused",
        frame_source=_fake_frame_source(range(2)),
        tracker=tracker,
        predict_fn=predict_fn,
    )

    assert tracks == []
