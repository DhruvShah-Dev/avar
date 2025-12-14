# src/avar/tracking/simple_tracker.py

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


def iou(b1: np.ndarray, b2: np.ndarray) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = area1 + area2 - inter + 1e-6
    return float(inter / union)


@dataclass
class TrackState:
    track_id: int
    bbox: np.ndarray
    score: float
    cls: int
    age: int = 0
    hits: int = 1


class MultiObjectTracker:
    """
    IOU-based greedy tracker: good, simple baseline.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: List[TrackState] = []
        self.next_id = 1

    def update(self, detections: np.ndarray) -> List[Dict[str, Any]]:
        if detections is None or len(detections) == 0:
            for t in self.tracks:
                t.age += 1
            self.tracks = [t for t in self.tracks if t.age <= self.max_age]
            return [
                {
                    "track_id": int(t.track_id),
                    "bbox": t.bbox.tolist(),
                    "score": float(t.score),
                    "class_id": int(t.cls),
                }
                for t in self.tracks
            ]

        dets = detections.copy()
        det_boxes = dets[:, :4]
        det_scores = dets[:, 4]
        det_classes = dets[:, 5].astype(int)
        assigned = [-1] * len(det_boxes)

        # Match existing tracks
        for t_idx, track in enumerate(self.tracks):
            best_iou = 0.0
            best_det = -1
            for d_idx, box in enumerate(det_boxes):
                if assigned[d_idx] != -1:
                    continue
                val = iou(track.bbox, box)
                if val > best_iou:
                    best_iou = val
                    best_det = d_idx

            if best_det >= 0 and best_iou >= self.iou_threshold:
                track.bbox = det_boxes[best_det]
                track.score = float(det_scores[best_det])
                track.cls = int(det_classes[best_det])
                track.age = 0
                track.hits += 1
                assigned[best_det] = t_idx
            else:
                track.age += 1

        # New tracks
        for d_idx, box in enumerate(det_boxes):
            if assigned[d_idx] != -1:
                continue
            self.tracks.append(
                TrackState(
                    track_id=self.next_id,
                    bbox=box,
                    score=float(det_scores[d_idx]),
                    cls=int(det_classes[d_idx]),
                )
            )
            self.next_id += 1

        # Remove stale tracks
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]

        # Output
        return [
            {
                "track_id": int(t.track_id),
                "bbox": t.bbox.tolist(),
                "score": float(t.score),
                "class_id": int(t.cls),
            }
            for t in self.tracks
        ]
