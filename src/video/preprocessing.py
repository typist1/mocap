"""Frame preprocessing: resize, color conversion, normalization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class FrameData:
    index: int
    timestamp_sec: float
    frame_bgr: np.ndarray
    frame_rgb: np.ndarray
    scale_factor: float  # original_px * scale_factor = processed_px


class FramePreprocessor:
    def __init__(self, max_dimension: int = 1280, normalize: bool = False) -> None:
        if max_dimension <= 0:
            raise ValueError(f"max_dimension must be > 0, got {max_dimension}")
        self.max_dimension = max_dimension
        self.normalize = normalize

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """Resize so max(w,h) <= max_dimension. Returns (processed_bgr, scale_factor)."""
        if frame is None or frame.size == 0:
            raise ValueError("Frame is empty or None")
        h, w = frame.shape[:2]
        max_hw = max(h, w)
        if max_hw <= self.max_dimension:
            scale = 1.0
            out = frame
        else:
            scale = self.max_dimension / max_hw
            new_w = int(w * scale)
            new_h = int(h * scale)
            out = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return out, scale

    def to_rgb(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            raise ValueError("Frame is empty or None")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """uint8 → float32 [0, 1]."""
        return frame.astype(np.float32) / 255.0
