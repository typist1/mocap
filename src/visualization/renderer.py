"""OpenCV-based skeleton visualization."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from src.pose.keypoints import MP33_NAMES, SKELETON_CONNECTIONS
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SkeletonRenderer:
    def __init__(self, min_visibility: float = 0.3) -> None:
        self.min_visibility = min_visibility

    def draw_skeleton_2d(
        self,
        frame: np.ndarray,
        landmarks_image: np.ndarray,  # [33, 3]: x,y norm + vis
        color: tuple = (0, 255, 0),
        thickness: int = 3,
    ) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        for a, b in SKELETON_CONNECTIONS:
            if landmarks_image[a, 2] > self.min_visibility and landmarks_image[b, 2] > self.min_visibility:
                pa = (int(landmarks_image[a, 0] * w), int(landmarks_image[a, 1] * h))
                pb = (int(landmarks_image[b, 0] * w), int(landmarks_image[b, 1] * h))
                cv2.line(out, pa, pb, color, thickness)
        # Draw joint dots
        for i in range(33):
            if landmarks_image[i, 2] > self.min_visibility:
                px = int(landmarks_image[i, 0] * w)
                py = int(landmarks_image[i, 1] * h)
                cv2.circle(out, (px, py), 4, (255, 255, 0), -1)
        return out

    def draw_confidence(self, frame: np.ndarray, landmarks_image: np.ndarray) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        for i in range(33):
            vis = float(landmarks_image[i, 2])
            px = int(landmarks_image[i, 0] * w)
            py = int(landmarks_image[i, 1] * h)
            if vis > 0.7:
                color = (0, 255, 0)
            elif vis > 0.4:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
            cv2.circle(out, (px, py), 5, color, -1)
        return out

    def draw_info_text(self, frame: np.ndarray, info: Dict[str, str]) -> np.ndarray:
        out = frame.copy()
        y = 20
        for k, v in info.items():
            text = f"{k}: {v}"
            cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y += 20
        return out


class VideoWriter:
    def __init__(self, output_path: Path, fps: float, width: int, height: int) -> None:
        self.output_path = Path(output_path)
        self.fps = fps
        self.width = width
        self.height = height
        self._writer: Optional[cv2.VideoWriter] = None
        self._frames_written = 0

    def _ensure_open(self) -> None:
        if self._writer is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(self.output_path), fourcc, self.fps, (self.width, self.height)
            )
            if not self._writer.isOpened():
                raise RuntimeError(f"Cannot open VideoWriter for {self.output_path}")

    def write(self, frame: np.ndarray) -> None:
        self._ensure_open()
        self._writer.write(frame)
        self._frames_written += 1

    @property
    def frames_written(self) -> int:
        return self._frames_written

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *_) -> None:
        self.release()


class PreviewGenerator:
    def __init__(self, renderer: Optional[SkeletonRenderer] = None) -> None:
        self.renderer = renderer or SkeletonRenderer()

    def generate_preview(
        self,
        video_path: Path,
        landmarks_sequence: np.ndarray,   # [F, 33, 3]
        fps: float,
        output_path: Optional[Path] = None,
    ) -> Path:
        video_path = Path(video_path)
        if output_path is None:
            output_path = video_path.parent / (video_path.stem + "_preview.mp4")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        try:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            with VideoWriter(output_path, fps, w, h) as writer:
                frame_idx = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_idx < len(landmarks_sequence):
                        frame = self.renderer.draw_skeleton_2d(frame, landmarks_sequence[frame_idx])
                    writer.write(frame)
                    frame_idx += 1
        finally:
            cap.release()

        logger.info("Preview written to %s", output_path)
        return output_path
