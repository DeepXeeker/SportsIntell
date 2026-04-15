import numpy as np

from sportsintell.tracker.association import hungarian_iou_match


def test_association_perfect_match():
    tracks = np.array([[10, 10, 4, 4], [30, 30, 5, 5]], dtype=np.float32)
    dets = np.array([[10, 10, 4, 4], [30, 30, 5, 5]], dtype=np.float32)
    result = hungarian_iou_match(tracks, dets, iou_threshold=0.5)
    assert len(result.matches) == 2
    assert not result.unmatched_tracks
    assert not result.unmatched_detections
