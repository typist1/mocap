"""Video metadata extraction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class VideoMetadata:
    path: Path
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    codec: str
    has_audio: bool
    aspect_ratio: float
    orientation: str  # "landscape" / "portrait" / "square"

    def __str__(self) -> str:
        return (
            f"{self.path.name}: {self.width}x{self.height} @ {self.fps:.2f}fps, "
            f"{self.total_frames} frames ({self.duration_seconds:.1f}s), "
            f"codec={self.codec}, {self.orientation}"
        )


def extract_metadata(path: Path) -> VideoMetadata:
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)).strip()
    finally:
        cap.release()

    if fps <= 0:
        raise ValueError(f"Video has invalid FPS={fps}: {path}")
    if total_frames <= 0:
        raise ValueError(f"Video reports 0 frames: {path}")
    if width <= 0 or height <= 0:
        raise ValueError(f"Video has invalid dimensions {width}x{height}: {path}")

    duration = total_frames / fps
    aspect_ratio = width / height

    if width > height:
        orientation = "landscape"
    elif height > width:
        orientation = "portrait"
    else:
        orientation = "square"

    # OpenCV doesn't expose audio track info; default to False
    has_audio = False

    return VideoMetadata(
        path=path,
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
        duration_seconds=duration,
        codec=codec,
        has_audio=has_audio,
        aspect_ratio=aspect_ratio,
        orientation=orientation,
    )
