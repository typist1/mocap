"""Subject identity tracking across frames."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Optional, Tuple

from src.detection.person_detector import DetectionResult
from src.utils.logging import get_logger

logger = get_logger(__name__)

BBox = Tuple[float, float, float, float]  # x1,y1,x2,y2 normalized


class TrackState(Enum):
    ACTIVE = "active"
    LOST = "lost"
    RECOVERED = "recovered"


@dataclass
class SubjectTrack:
    track_id: int
    state: TrackState = TrackState.ACTIVE
    last_seen_frame: int = 0
    gap_frames: int = 0
    bbox_history: Deque[BBox] = field(default_factory=lambda: deque(maxlen=10))
    confidence_history: Deque[float] = field(default_factory=lambda: deque(maxlen=30))

    @property
    def is_active(self) -> bool:
        return self.state in (TrackState.ACTIVE, TrackState.RECOVERED)

    @property
    def consecutive_lost(self) -> int:
        return self.gap_frames

    @property
    def last_bbox(self) -> Optional[BBox]:
        return self.bbox_history[-1] if self.bbox_history else None


def iou(a: BBox, b: BBox) -> float:
    """Intersection over Union of two normalized bboxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 1e-9 else 0.0


class SubjectTracker:
    def __init__(self, max_gap_frames: int = 15, iou_threshold: float = 0.3) -> None:
        self.max_gap_frames = max_gap_frames
        self.iou_threshold = iou_threshold
        self._track = SubjectTrack(track_id=0)

    def update(self, detection: Optional[DetectionResult], frame_index: int) -> TrackState:
        if detection is None:
            self._track.gap_frames += 1
            if self._track.gap_frames > self.max_gap_frames:
                if self._track.state != TrackState.LOST:
                    logger.debug("Track lost at frame %d after %d gap frames", frame_index, self._track.gap_frames)
                self._track.state = TrackState.LOST
            else:
                self._track.state = TrackState.LOST
            return self._track.state

        # Check IoU against last known position
        if self._track.last_bbox is not None:
            overlap = iou(self._track.last_bbox, detection.bbox)
            if overlap < self.iou_threshold and self._track.gap_frames == 0:
                # Potential identity switch — accept cautiously
                logger.debug("Low IoU %.2f at frame %d, potential subject switch", overlap, frame_index)

        was_lost = self._track.state == TrackState.LOST
        self._track.bbox_history.append(detection.bbox)
        self._track.confidence_history.append(detection.confidence)
        self._track.last_seen_frame = frame_index
        self._track.gap_frames = 0
        self._track.state = TrackState.RECOVERED if was_lost else TrackState.ACTIVE
        return self._track.state

    @property
    def track(self) -> SubjectTrack:
        return self._track
