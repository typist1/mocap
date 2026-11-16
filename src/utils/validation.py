"""Input validation utilities."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def validate_video_path(path: Union[str, Path]) -> Path:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Video file not found: {p}")
    if not p.is_file():
        raise ValueError(f"Path is not a file: {p}")
    if p.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported video extension '{p.suffix}'. "
            f"Supported: {sorted(SUPPORTED_VIDEO_EXTENSIONS)}"
        )
    return p


def validate_output_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    test_file = p / ".write_test"
    try:
        test_file.touch()
        test_file.unlink()
    except OSError as e:
        raise ValueError(f"Output directory not writable: {p} ({e})")
    return p


def validate_positive(value: float, name: str) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def validate_range(value: float, name: str, lo: float, hi: float) -> float:
    if not (lo <= value <= hi):
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {value}")
    return value


def validate_numpy_shape(arr: np.ndarray, expected_shape: tuple, name: str) -> None:
    if arr.ndim != len(expected_shape):
        raise ValueError(
            f"{name}: expected {len(expected_shape)}D array, got {arr.ndim}D "
            f"with shape {arr.shape}"
        )
    for i, (actual, expected) in enumerate(zip(arr.shape, expected_shape)):
        if expected != -1 and actual != expected:
            raise ValueError(
                f"{name}: expected shape {expected_shape}, got {arr.shape} "
                f"(mismatch at dim {i})"
            )


def validate_config(settings) -> None:
    from src.config.settings import Settings
    if not isinstance(settings, Settings):
        raise TypeError(f"Expected Settings, got {type(settings)}")
    validate_positive(settings.video.target_fps, "video.target_fps")
    validate_positive(settings.video.max_dimension, "video.max_dimension")
    validate_range(
        settings.detection.min_detection_confidence,
        "detection.min_detection_confidence", 0.0, 1.0
    )
    validate_range(
        settings.detection.min_visibility,
        "detection.min_visibility", 0.0, 1.0
    )
    known_formats = {"bvh", "npz", "json"}
    for fmt in settings.animation.export_formats:
        if fmt not in known_formats:
            raise ValueError(f"Unknown export format: {fmt!r}. Known: {known_formats}")
    known_devices = {"auto", "cpu", "cuda", "mps"}
    if settings.device not in known_devices:
        raise ValueError(f"Unknown device: {settings.device!r}. Known: {known_devices}")
