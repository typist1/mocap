"""2D pose estimation — MediaPipe backend with rotation fallback."""
from __future__ import annotations

import os
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config.settings import DetectionConfig, PoseConfig
from src.pose.keypoints import BODY_JOINTS
from src.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)

ROTATIONS = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]


@dataclass
class Pose2DResult:
    landmarks_image: np.ndarray   # [33, 3]: x, y (normalized 0-1), visibility
    landmarks_world: np.ndarray   # [33, 3]: x, y, z in meters (MediaPipe world space)
    body_confidence: float        # mean visibility of body joints
    frame_index: int
    rotation_applied: Optional[int]  # OpenCV rotation constant or None


class Pose2DEstimator(ABC):
    @abstractmethod
    def estimate(self, frame_rgb: np.ndarray, frame_index: int) -> Optional[Pose2DResult]:
        ...

    def close(self) -> None:
        pass


def _unrotate_landmarks(image_pts: np.ndarray, world_pts: np.ndarray, rot: Optional[int]):
    """Map landmarks from rotated image back to original image orientation."""
    img = image_pts.copy()
    wld = world_pts.copy()
    if rot is None:
        return img, wld
    if rot == cv2.ROTATE_180:
        img[:, 0] = 1.0 - img[:, 0]
        img[:, 1] = 1.0 - img[:, 1]
        wld[:, 0] = -wld[:, 0]
        wld[:, 1] = -wld[:, 1]
    elif rot == cv2.ROTATE_90_CLOCKWISE:
        img[:, 0], img[:, 1] = img[:, 1].copy(), 1.0 - img[:, 0].copy()
        wld[:, 0], wld[:, 1] = wld[:, 1].copy(), -wld[:, 0].copy()
    else:  # ROTATE_90_COUNTERCLOCKWISE
        img[:, 0], img[:, 1] = 1.0 - img[:, 1].copy(), img[:, 0].copy()
        wld[:, 0], wld[:, 1] = -wld[:, 1].copy(), wld[:, 0].copy()
    return img, wld


def _detect_one(pose_solution, frame_rgb: np.ndarray, rot: Optional[int]):
    """Run detection on possibly-rotated frame. Returns (score, image_pts, world_pts) or None."""
    if rot is not None:
        frame_rgb = cv2.rotate(frame_rgb, rot)
    result = pose_solution.process(frame_rgb)
    if not result.pose_landmarks:
        return None
    image_pts = np.array([[p.x, p.y, p.visibility] for p in result.pose_landmarks.landmark])
    world_pts = np.array([[p.x, p.y, p.z] for p in result.pose_world_landmarks.landmark])
    image_pts, world_pts = _unrotate_landmarks(image_pts, world_pts, rot)
    score = float(image_pts[list(BODY_JOINTS), 2].mean())
    return score, image_pts, world_pts


class MediaPipePose2D(Pose2DEstimator):
    def __init__(self, pose_config: PoseConfig, detection_config: DetectionConfig) -> None:
        self.pose_config = pose_config
        self.detection_config = detection_config
        self._prev_rot: Optional[int] = None
        self._solution = None
        self._init_solution()

    def _init_solution(self) -> None:
        try:
            import mediapipe as mp
            self._solution = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=self.pose_config.model_complexity,
                min_detection_confidence=self.detection_config.min_detection_confidence,
                min_tracking_confidence=self.detection_config.min_tracking_confidence,
                smooth_landmarks=False,  # we do our own smoothing
            )
            logger.info("MediaPipe Pose initialized (complexity=%d)", self.pose_config.model_complexity)
        except Exception as exc:
            raise RuntimeError(f"Cannot initialize MediaPipe Pose: {exc}") from exc

    def estimate(self, frame_rgb: np.ndarray, frame_index: int) -> Optional[Pose2DResult]:
        threshold = self.detection_config.rotation_fallback_threshold
        results: dict = {}

        # Always try previous rotation first
        results[self._prev_rot] = _detect_one(self._solution, frame_rgb, self._prev_rot)
        first = results[self._prev_rot]

        # Only try other rotations if confidence is low
        if first is None or first[0] < threshold:
            for rot in ROTATIONS:
                if rot not in results:
                    results[rot] = _detect_one(self._solution, frame_rgb, rot)

        best_rot = self._prev_rot
        best = first
        for rot, res in results.items():
            if res is None:
                continue
            # Prefer previous rotation unless new one is clearly better (prevents flicker)
            if best is None or res[0] > best[0] + 0.1:
                best_rot = rot
                best = res

        if best is None:
            return None

        self._prev_rot = best_rot
        score, image_pts, world_pts = best

        return Pose2DResult(
            landmarks_image=image_pts,
            landmarks_world=world_pts,
            body_confidence=score,
            frame_index=frame_index,
            rotation_applied=best_rot,
        )

    def close(self) -> None:
        if self._solution is not None:
            try:
                self._solution.close()
            except Exception:
                pass
            self._solution = None
