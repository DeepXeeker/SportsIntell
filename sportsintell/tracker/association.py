from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from sportsintell.utils.geometry import pairwise_iou


@dataclass
class MatchResult:
    matches: list[tuple[int, int]]
    unmatched_tracks: list[int]
    unmatched_detections: list[int]


def hungarian_iou_match(
    track_boxes: np.ndarray,
    det_boxes: np.ndarray,
    iou_threshold: float = 0.3,
) -> MatchResult:
    if len(track_boxes) == 0:
        return MatchResult(matches=[], unmatched_tracks=[], unmatched_detections=list(range(len(det_boxes))))
    if len(det_boxes) == 0:
        return MatchResult(matches=[], unmatched_tracks=list(range(len(track_boxes))), unmatched_detections=[])

    ious = pairwise_iou(track_boxes, det_boxes)
    cost = 1.0 - ious
    row_idx, col_idx = linear_sum_assignment(cost)

    matches: list[tuple[int, int]] = []
    matched_tracks = set()
    matched_dets = set()
    for r, c in zip(row_idx, col_idx):
        if ious[r, c] >= iou_threshold:
            matches.append((r, c))
            matched_tracks.add(r)
            matched_dets.add(c)

    unmatched_tracks = [i for i in range(len(track_boxes)) if i not in matched_tracks]
    unmatched_detections = [j for j in range(len(det_boxes)) if j not in matched_dets]
    return MatchResult(matches=matches, unmatched_tracks=unmatched_tracks, unmatched_detections=unmatched_detections)
