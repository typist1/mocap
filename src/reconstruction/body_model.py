"""Body proportions estimation and bone length normalization."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from src.pose.keypoints import BODY_JOINTS, MP33
from src.reconstruction.kinematics import Skeleton
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BodyProportions:
    bone_lengths: Dict[str, float] = field(default_factory=dict)

    @property
    def leg_length(self) -> float:
        left = (
            self.bone_lengths.get("left_hip_left_knee", 0.0)
            + self.bone_lengths.get("left_knee_left_ankle", 0.0)
        )
        right = (
            self.bone_lengths.get("right_hip_right_knee", 0.0)
            + self.bone_lengths.get("right_knee_right_ankle", 0.0)
        )
        candidates = [v for v in [left, right] if v > 0]
        return float(np.mean(candidates)) if candidates else 0.0

    @property
    def arm_length(self) -> float:
        left = (
            self.bone_lengths.get("left_shoulder_left_elbow", 0.0)
            + self.bone_lengths.get("left_elbow_left_wrist", 0.0)
        )
        right = (
            self.bone_lengths.get("right_shoulder_right_elbow", 0.0)
            + self.bone_lengths.get("right_elbow_right_wrist", 0.0)
        )
        candidates = [v for v in [left, right] if v > 0]
        return float(np.mean(candidates)) if candidates else 0.0

    @property
    def torso_length(self) -> float:
        return self.bone_lengths.get("spine_shoulder_mid", 0.0)


def estimate_body_proportions(
    sequence_world: np.ndarray,     # [F, 33, 3]
    sequence_visibility: np.ndarray,  # [F, 33]
    skeleton: Skeleton,
    n_frames: int = 0,
) -> BodyProportions:
    """Estimate median bone lengths from frames where body is well-detected."""
    F = sequence_world.shape[0]
    body_conf = sequence_visibility[:, list(BODY_JOINTS)].mean(axis=1)
    good_frames = np.where(body_conf >= 0.5)[0]

    if len(good_frames) == 0:
        logger.warning("No high-confidence frames for body proportion estimation")
        good_frames = np.arange(F)

    if n_frames > 0 and len(good_frames) > n_frames:
        indices = np.linspace(0, len(good_frames) - 1, n_frames, dtype=int)
        good_frames = good_frames[indices]

    bone_data: Dict[str, list] = {}
    for frame_idx in good_frames:
        positions = sequence_world[frame_idx]
        vis = sequence_visibility[frame_idx]
        lengths = skeleton.compute_bone_lengths(positions)
        for name, joint in skeleton.joints.items():
            if joint.parent is None or joint.index < 0 or joint.parent.index < 0:
                continue
            if joint.parent.index >= 33 or joint.index >= 33:
                continue
            if vis[joint.parent.index] < 0.5 or vis[joint.index] < 0.5:
                continue
            bone_name = f"{joint.parent.name}_{name}"
            if bone_name in lengths:
                bone_data.setdefault(bone_name, []).append(lengths[bone_name])

    bone_lengths: Dict[str, float] = {}
    for bone_name, values in bone_data.items():
        if values:
            bone_lengths[bone_name] = float(np.median(values))

    logger.debug("Estimated proportions from %d frames, %d bones", len(good_frames), len(bone_lengths))
    return BodyProportions(bone_lengths=bone_lengths)


def apply_bone_normalization(
    positions: np.ndarray,  # [33, 3]
    proportions: BodyProportions,
    skeleton: Skeleton,
) -> np.ndarray:
    """Normalize bone lengths to target proportions, anchored at root."""
    if not proportions.bone_lengths:
        return positions
    return skeleton.normalize_bone_lengths(positions, proportions.bone_lengths)
