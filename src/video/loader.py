"""Video loading with frame-rate sampling."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator, Optional

import cv2

from src.config.settings import VideoConfig
from src.utils.logging import get_logger
from src.video.metadata import VideoMetadata
from src.video.preprocessing import FrameData, FramePreprocessor

logger = get_logger(__name__)


class VideoLoader:
    def __init__(
        self,
        metadata: VideoMetadata,
        config: VideoConfig,
        preprocessor: FramePreprocessor,
    ) -> None:
        self.metadata = metadata
        self.config = config
        self.preprocessor = preprocessor
        self._cap: Optional[cv2.VideoCapture] = None

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(str(self.metadata.path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.metadata.path}")
        return cap

    @property
    def total_output_frames(self) -> int:
        return math.ceil(self.metadata.duration_seconds * self.config.target_fps)

    def frames(self, start: int = 0, end: int = -1) -> Iterator[FrameData]:
        """Yield FrameData at target_fps, correctly handling source fps != target fps."""
        src_fps = self.metadata.fps
        tgt_fps = self.config.target_fps
        cap = self._open()
        self._cap = cap

        if end < 0:
            end = self.metadata.total_frames

        try:
            frame_index = 0        # source frame counter
            output_index = 0       # output frame counter
            next_sample_time = start / src_fps  # time of next sample to emit

            # Seek to start frame if needed
            if start > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)

            prev_bgr: Optional[cv2.typing.MatLike] = None

            while frame_index < end:
                ok, bgr = cap.read()
                if not ok:
                    break

                current_time = (start + frame_index) / src_fps
                frame_index += 1

                # Skip frames before next sample time
                if current_time + 1e-9 < next_sample_time:
                    continue

                if bgr is None or bgr.size == 0:
                    logger.warning("Corrupted frame at source index %d, skipping", start + frame_index - 1)
                    continue

                try:
                    processed_bgr, scale = self.preprocessor.process(bgr)
                    processed_rgb = self.preprocessor.to_rgb(processed_bgr)
                except Exception as exc:
                    logger.warning("Frame preprocessing failed at index %d: %s", start + frame_index - 1, exc)
                    continue

                # Emit one or more frames at target fps (handles upsampling)
                while next_sample_time <= current_time + 1e-9:
                    fd = FrameData(
                        index=output_index,
                        timestamp_sec=next_sample_time,
                        frame_bgr=processed_bgr,
                        frame_rgb=processed_rgb,
                        scale_factor=scale,
                    )
                    yield fd
                    output_index += 1
                    next_sample_time += 1.0 / tgt_fps

                    if output_index % 100 == 0:
                        logger.info("Loaded %d frames (%.1fs)", output_index, next_sample_time)

                prev_bgr = bgr

        finally:
            cap.release()
            self._cap = None

    def __enter__(self) -> "VideoLoader":
        return self

    def __exit__(self, *_) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
