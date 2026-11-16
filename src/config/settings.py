"""Central configuration. Load from YAML or use defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class VideoConfig:
    target_fps: float = 30.0
    max_dimension: int = 1280
    min_dimension: int = 320
    # Minimum consecutive detected frames before we accept a subject
    min_detection_frames: int = 3


@dataclass
class DetectionConfig:
    min_detection_confidence: float = 0.3
    min_tracking_confidence: float = 0.3
    min_visibility: float = 0.3
    # Try rotations when body confidence is below this
    rotation_fallback_threshold: float = 0.8
    # Maximum frames without detection before track is lost
    max_gap_frames: int = 15


@dataclass
class PoseConfig:
    # 2D model: "mediapipe" (default, no download needed)
    backend_2d: str = "mediapipe"
    model_complexity: int = 2  # MediaPipe: 0, 1, or 2
    # 3D lifting: "geometric" (pure geometry) or "temporal" (future: VideoPose3D)
    backend_3d: str = "geometric"
    # Path for model checkpoints (optional)
    checkpoint_dir: Optional[str] = None


@dataclass
class ReconstructionConfig:
    # Focal length estimation: "auto" (from shoulder width) or float (pixels)
    focal_length: str | float = "auto"
    # Enforce consistent bone lengths across sequence
    normalize_bone_lengths: bool = True
    # Number of frames used for bone length estimation (0 = all)
    bone_estimation_frames: int = 0
    # World coordinate system: "blender" (X right, Y forward, Z up) or "opengl"
    coordinate_system: str = "blender"


@dataclass
class FilteringConfig:
    # One Euro filter parameters (applied per joint, per axis)
    one_euro_min_cutoff: float = 1.5
    one_euro_beta: float = 0.1
    one_euro_d_cutoff: float = 1.0
    # Savitzky-Golay (offline smoothing after all frames collected)
    savgol_window: int = 11
    savgol_polyorder: int = 3
    savgol_enabled: bool = True
    # Butterworth (simple, used for root motion)
    butterworth_cutoff: float = 6.0
    butterworth_order: int = 2
    # Kalman: enabled for occluded landmark prediction
    kalman_enabled: bool = True
    kalman_process_noise: float = 1e-3
    kalman_measurement_noise: float = 1e-2


@dataclass
class AnimationConfig:
    # Export formats: list of "bvh", "npz", "json"
    export_formats: list[str] = field(default_factory=lambda: ["bvh", "npz", "json"])
    output_dir: str = "."
    # BVH scale factor (1.0 = metric meters)
    bvh_scale: float = 100.0  # cm


@dataclass
class EvaluationConfig:
    # Only compute MPJPE if ground-truth file provided
    ground_truth_path: Optional[str] = None
    # Save per-frame metrics to CSV
    save_metrics_csv: bool = True


@dataclass
class Settings:
    video: VideoConfig = field(default_factory=VideoConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    animation: AnimationConfig = field(default_factory=AnimationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    # Device: "auto" selects cuda > mps > cpu
    device: str = "auto"
    seed: int = 42
    log_level: str = "INFO"


def load_settings(path: Optional[str | Path] = None) -> Settings:
    """Load settings from optional YAML file, falling back to defaults."""
    if path is None:
        env_path = os.environ.get("MOCAP_CONFIG")
        if env_path:
            path = Path(env_path)
        else:
            default = Path("configs/default.yaml")
            path = default if default.exists() else None

    if path is None:
        return Settings()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    s = Settings()
    for section_name, section_cls in {
        "video": VideoConfig,
        "detection": DetectionConfig,
        "pose": PoseConfig,
        "reconstruction": ReconstructionConfig,
        "filtering": FilteringConfig,
        "animation": AnimationConfig,
        "evaluation": EvaluationConfig,
    }.items():
        if section_name in data:
            current = getattr(s, section_name)
            for k, v in data[section_name].items():
                if hasattr(current, k):
                    setattr(current, k, v)

    for top_key in ("device", "seed", "log_level"):
        if top_key in data:
            setattr(s, top_key, data[top_key])

    return s
