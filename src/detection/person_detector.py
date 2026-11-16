"""Human detection interface and MediaPipe implementation.

NOTE: MediaPipe Pose solutions are NOT thread-safe. Instantiate one
detector per thread.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src.config.settings import DetectionConfig
from src.pose.keypoints import BODY_JOINTS
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DetectionResult:
    # Normalized [0,1] bounding box (x1, y1, x2, y2)
    bbox: Tuple[float, float, float, float]
    confidence: float
    frame_index: int


class PersonDetector(ABC):
    @abstractmethod
    def detect(self, frame_rgb: np.ndarray, frame_index: int) -> Optional[DetectionResult]:
        ...

    def close(self) -> None:
        pass


class MediaPipeDetector(PersonDetector):
    """Derives detection bounding box from visible pose landmarks."""

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self._init_model()

    def _init_model(self) -> None:
        try:
            import mediapipe as mp
            self._mp_pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize MediaPipe Pose: {exc}") from exc

    def detect(self, frame_rgb: np.ndarray, frame_index: int) -> Optional[DetectionResult]:
        try:
            result = self._mp_pose.process(frame_rgb)
        except Exception as exc:
            logger.warning("MediaPipe detection failed at frame %d: %s", frame_index, exc)
            return None

        if not result.pose_landmarks:
            return None

        lm = result.pose_landmarks.landmark
        # Derive bbox from body joints only
        xs, ys, vis = [], [], []
        for idx in BODY_JOINTS:
            p = lm[idx]
            if p.visibility >= self.config.min_visibility:
                xs.append(p.x)
                ys.append(p.y)
                vis.append(p.visibility)

        if len(xs) < 2:
            return None

        x1, y1 = max(0.0, min(xs) - 0.05), max(0.0, min(ys) - 0.05)
        x2, y2 = min(1.0, max(xs) + 0.05), min(1.0, max(ys) + 0.05)

        # Degenerate box check
        if x2 - x1 < 1e-3 or y2 - y1 < 1e-3:
            return None

        confidence = float(np.mean(vis))
        return DetectionResult(bbox=(x1, y1, x2, y2), confidence=confidence, frame_index=frame_index)

    def close(self) -> None:
        try:
            self._mp_pose.close()
        except Exception:
            pass
