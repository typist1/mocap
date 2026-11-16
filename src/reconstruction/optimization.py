"""Post-processing optimization of the 3D pose sequence."""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.config.settings import FilteringConfig, ReconstructionConfig
from src.filtering.smoothing import (
    compute_jitter,
    fill_temporal_gaps,
    normalize_bone_lengths_sequence,
    savitzky_golay_smooth,
    smooth_root_motion,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PoseOptimizer:
    def __init__(
        self,
        filtering_config: FilteringConfig,
        reconstruction_config: ReconstructionConfig,
    ) -> None:
        self.filtering_cfg = filtering_config
        self.reconstruction_cfg = reconstruction_config

    def optimize(
        self,
        sequence_3d: np.ndarray,       # [F, 33, 3]
        visibility: np.ndarray,         # [F, 33]
        fps: float,
        body_proportions=None,          # Optional BodyProportions
        skeleton=None,                  # Optional Skeleton (required if normalize)
    ) -> np.ndarray:
        """Full offline optimization pipeline. Returns [F, 33, 3]."""
        seq = sequence_3d.copy()

        # Step 1: Fill temporal gaps
        seq = fill_temporal_gaps(seq, visibility)

        # Step 2: Bone length normalization
        if (
            self.reconstruction_cfg.normalize_bone_lengths
            and body_proportions is not None
            and skeleton is not None
            and body_proportions.bone_lengths
        ):
            seq = normalize_bone_lengths_sequence(seq, body_proportions.bone_lengths, skeleton)
            logger.debug("Bone normalization applied")

        # Step 3: Savitzky-Golay smoothing (whole sequence)
        if self.filtering_cfg.savgol_enabled:
            seq = savitzky_golay_smooth(
                seq,
                window=self.filtering_cfg.savgol_window,
                polyorder=self.filtering_cfg.savgol_polyorder,
            )

        # Step 4: Smooth root motion (hip midpoint trajectory)
        # Extract hip midpoint trajectory, smooth separately, then re-inject
        # (hip midpoint = average of left_hip [23] and right_hip [24])
        hip_mid = (seq[:, 23] + seq[:, 24]) / 2.0
        hip_mid_smooth = smooth_root_motion(hip_mid, fps, self.filtering_cfg.butterworth_cutoff)
        delta = hip_mid_smooth - hip_mid
        seq += delta[:, np.newaxis, :]  # translate all joints uniformly

        return seq

    def compute_metrics(
        self,
        raw: np.ndarray,
        optimized: np.ndarray,
    ) -> dict:
        jitter_raw = compute_jitter(raw)
        jitter_opt = compute_jitter(optimized)
        reduction = (jitter_raw - jitter_opt) / max(jitter_raw, 1e-9)

        def _bone_length_variance(seq: np.ndarray) -> float:
            # Use shoulder and hip widths as proxy
            shoulder_w = np.linalg.norm(seq[:, 11] - seq[:, 12], axis=1)
            hip_w = np.linalg.norm(seq[:, 23] - seq[:, 24], axis=1)
            combined = np.concatenate([shoulder_w, hip_w])
            return float(combined.std() / (combined.mean() + 1e-9))

        return {
            "jitter_reduction": float(reduction),
            "bone_length_variance_before": _bone_length_variance(raw),
            "bone_length_variance_after": _bone_length_variance(optimized),
        }
