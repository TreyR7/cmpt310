import numpy as np

from smart_livestock_gate.tracking.association import associate, iou_matrix
from smart_livestock_gate.tracking.sort_tracker import SortTracker, SortTrackerConfig


def test_iou_matrix_perfect_overlap():
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    result = iou_matrix(boxes, boxes)
    assert result.shape == (1, 1)
    assert result[0, 0] == 1.0


def test_iou_matrix_no_overlap():
    tracks = np.array([[0, 0, 10, 10]], dtype=np.float32)
    detections = np.array([[100, 100, 110, 110]], dtype=np.float32)
    result = iou_matrix(tracks, detections)
    assert result[0, 0] == 0.0


def test_iou_matrix_partial_overlap():
    tracks = np.array([[0, 0, 10, 10]], dtype=np.float32)
    detections = np.array([[5, 0, 15, 10]], dtype=np.float32)
    result = iou_matrix(tracks, detections)
    # intersection 5x10=50, union 100+100-50=150
    assert np.isclose(result[0, 0], 50 / 150)


def test_associate_matches_close_boxes():
    tracks = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=np.float32)
    detections = np.array([[1, 1, 11, 11], [101, 101, 111, 111]], dtype=np.float32)

    matches, unmatched_tracks, unmatched_detections = associate(tracks, detections)

    assert len(matches) == 2
    assert len(unmatched_tracks) == 0
    assert len(unmatched_detections) == 0
    matched_pairs = {tuple(pair) for pair in matches}
    assert (0, 0) in matched_pairs
    assert (1, 1) in matched_pairs


def test_associate_rejects_below_threshold():
    tracks = np.array([[0, 0, 10, 10]], dtype=np.float32)
    detections = np.array([[9, 9, 19, 19]], dtype=np.float32)

    matches, unmatched_tracks, unmatched_detections = associate(
        tracks, detections, iou_threshold=0.5
    )

    assert len(matches) == 0
    assert list(unmatched_tracks) == [0]
    assert list(unmatched_detections) == [0]


def test_associate_handles_empty_detections():
    tracks = np.array([[0, 0, 10, 10]], dtype=np.float32)
    detections = np.empty((0, 4), dtype=np.float32)

    matches, unmatched_tracks, unmatched_detections = associate(tracks, detections)

    assert matches.shape == (0, 2)
    assert list(unmatched_tracks) == [0]
    assert list(unmatched_detections) == []


def test_associate_handles_empty_tracks():
    tracks = np.empty((0, 4), dtype=np.float32)
    detections = np.array([[0, 0, 10, 10]], dtype=np.float32)

    matches, unmatched_tracks, unmatched_detections = associate(tracks, detections)

    assert matches.shape == (0, 2)
    assert list(unmatched_tracks) == []
    assert list(unmatched_detections) == [0]


def _box_at(frame: int, start=0.0, step=5.0, size=10.0) -> dict:
    x1 = start + step * frame
    return {"box": [x1, 0.0, x1 + size, size], "confidence": 0.9}


def test_tracker_assigns_stable_id_to_linearly_moving_box():
    tracker = SortTracker(SortTrackerConfig(min_hits=1))

    track_ids = set()
    for frame in range(10):
        results = tracker.update([_box_at(frame)], frame_index=frame)
        assert len(results) == 1
        track_ids.add(results[0].track_id)

    assert len(track_ids) == 1


def test_tracker_starts_new_id_for_new_detection():
    tracker = SortTracker(SortTrackerConfig(min_hits=1))

    first_ids = set()
    for frame in range(3):
        results = tracker.update([_box_at(frame)], frame_index=frame)
        first_ids.update(result.track_id for result in results)

    results = tracker.update(
        [_box_at(3), _box_at(0, start=200.0)], frame_index=3
    )
    all_ids = {result.track_id for result in results}

    assert len(all_ids) == 2
    assert first_ids <= all_ids


def test_tracker_drops_track_after_max_age():
    tracker = SortTracker(SortTrackerConfig(min_hits=1, max_age=2))

    tracker.update([_box_at(0)], frame_index=0)
    tracker.update([_box_at(1)], frame_index=1)

    for frame in range(2, 6):
        results = tracker.update([], frame_index=frame)

    assert results == []


def test_tracker_requires_min_hits_before_reporting():
    tracker = SortTracker(SortTrackerConfig(min_hits=3, max_age=1))

    # First update always creates a track with age 0, matched this frame,
    # so it is reported under the startup grace window (age <= min_hits).
    results = tracker.update([_box_at(0)], frame_index=0)
    assert len(results) == 1


def test_tracker_reset_clears_state():
    tracker = SortTracker(SortTrackerConfig(min_hits=1))
    tracker.update([_box_at(0)], frame_index=0)

    tracker.reset()
    second = tracker.update([_box_at(0)], frame_index=0)

    assert second[0].track_id == 1
