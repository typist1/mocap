"""Retarget MediaPipe 33 landmarks to AnimationSkeleton."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.animation.skeleton import (
    AnimationData,
    AnimationFrame,
    AnimationSkeleton,
    build_animation_skeleton,
    quat_slerp,
    rotation_from_directions,
)
from src.pose.keypoints import MP33
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Maps each BVH joint → (parent_mp33_idx, child_mp33_idx)
# Direction of bone = positions[child] - positions[parent]
MP33_TO_ANIM_JOINT: Dict[str, tuple] = {
    "Hips":          (MP33.LEFT_HIP,       MP33.RIGHT_HIP),
    "Spine":         (MP33.LEFT_HIP,       MP33.LEFT_SHOULDER),
    "Spine1":        (MP33.LEFT_HIP,       MP33.LEFT_SHOULDER),
    "Spine2":        (MP33.RIGHT_HIP,      MP33.RIGHT_SHOULDER),
    "Neck":          (MP33.LEFT_SHOULDER,  MP33.NOSE),
    "Head":          (MP33.LEFT_EAR,       MP33.RIGHT_EAR),
    "LeftShoulder":  (MP33.RIGHT_SHOULDER, MP33.LEFT_SHOULDER),
    "LeftArm":       (MP33.LEFT_SHOULDER,  MP33.LEFT_ELBOW),
    "LeftForeArm":   (MP33.LEFT_ELBOW,     MP33.LEFT_WRIST),
    "LeftHand":      (MP33.LEFT_WRIST,     MP33.LEFT_INDEX),
    "RightShoulder": (MP33.LEFT_SHOULDER,  MP33.RIGHT_SHOULDER),
    "RightArm":      (MP33.RIGHT_SHOULDER, MP33.RIGHT_ELBOW),
    "RightForeArm":  (MP33.RIGHT_ELBOW,    MP33.RIGHT_WRIST),
    "RightHand":     (MP33.RIGHT_WRIST,    MP33.RIGHT_INDEX),
    "LeftUpLeg":     (MP33.LEFT_HIP,       MP33.LEFT_KNEE),
    "LeftLeg":       (MP33.LEFT_KNEE,      MP33.LEFT_ANKLE),
    "LeftFoot":      (MP33.LEFT_ANKLE,     MP33.LEFT_FOOT_INDEX),
    "LeftToeBase":   (MP33.LEFT_HEEL,      MP33.LEFT_FOOT_INDEX),
    "RightUpLeg":    (MP33.RIGHT_HIP,      MP33.RIGHT_KNEE),
    "RightLeg":      (MP33.RIGHT_KNEE,     MP33.RIGHT_ANKLE),
    "RightFoot":     (MP33.RIGHT_ANKLE,    MP33.RIGHT_FOOT_INDEX),
    "RightToeBase":  (MP33.RIGHT_HEEL,     MP33.RIGHT_FOOT_INDEX),
}

# Rest direction for each joint (what direction the bone points in rest pose)
_REST_DIRECTIONS: Dict[str, np.ndarray] = {
    "Hips":          np.array([0.0, 1.0, 0.0]),
    "Spine":         np.array([0.0, 1.0, 0.0]),
    "Spine1":        np.array([0.0, 1.0, 0.0]),
    "Spine2":        np.array([0.0, 1.0, 0.0]),
    "Neck":          np.array([0.0, 1.0, 0.0]),
    "Head":          np.array([0.0, 1.0, 0.0]),
    "LeftShoulder":  np.array([-1.0, 0.0, 0.0]),
    "LeftArm":       np.array([-1.0, 0.0, 0.0]),
    "LeftForeArm":   np.array([-1.0, 0.0, 0.0]),
    "LeftHand":      np.array([-1.0, 0.0, 0.0]),
    "RightShoulder": np.array([1.0, 0.0, 0.0]),
    "RightArm":      np.array([1.0, 0.0, 0.0]),
    "RightForeArm":  np.array([1.0, 0.0, 0.0]),
    "RightHand":     np.array([1.0, 0.0, 0.0]),
    "LeftUpLeg":     np.array([0.0, -1.0, 0.0]),
    "LeftLeg":       np.array([0.0, -1.0, 0.0]),
    "LeftFoot":      np.array([0.0, -1.0, 0.0]),
    "LeftToeBase":   np.array([0.0, 0.0, 1.0]),
    "RightUpLeg":    np.array([0.0, -1.0, 0.0]),
    "RightLeg":      np.array([0.0, -1.0, 0.0]),
    "RightFoot":     np.array([0.0, -1.0, 0.0]),
    "RightToeBase":  np.array([0.0, 0.0, 1.0]),
}
_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


class MediaPipeToSkeletonRetargeter:
    def retarget_frame(
        self,
        positions_3d: np.ndarray,          # [33, 3]
        frame_index: int,
        timestamp: float,
        skeleton: AnimationSkeleton,
        prev_frame: Optional[AnimationFrame] = None,
    ) -> AnimationFrame:
        # Root position = hip midpoint
        hip_mid = (positions_3d[MP33.LEFT_HIP] + positions_3d[MP33.RIGHT_HIP]) / 2.0
        joint_rotations: Dict[str, np.ndarray] = {}

        for joint in skeleton.joints:
            if joint not in MP33_TO_ANIM_JOINT:
                joint_rotations[joint] = _IDENTITY_QUAT.copy()
                continue

            p_idx, c_idx = MP33_TO_ANIM_JOINT[joint]
            p_pos = positions_3d[p_idx]
            c_pos = positions_3d[c_idx]
            direction = c_pos - p_pos
            length = np.linalg.norm(direction)
            if length < 1e-6:
                joint_rotations[joint] = _IDENTITY_QUAT.copy()
                continue
            direction /= length

            rest_dir = _REST_DIRECTIONS.get(joint, np.array([0.0, 1.0, 0.0]))
            q = rotation_from_directions(rest_dir, direction)

            # Minimal SLERP blend with previous frame to reduce discontinuities
            if prev_frame is not None and joint in prev_frame.joint_rotations:
                q_prev = prev_frame.joint_rotations[joint]
                q = quat_slerp(q_prev, q, 0.9)  # 90% toward new, 10% prev

            joint_rotations[joint] = q

        return AnimationFrame(
            frame_index=frame_index,
            timestamp=timestamp,
            root_position=hip_mid,
            joint_rotations=joint_rotations,
        )

    def retarget_sequence(
        self,
        sequence_3d: np.ndarray,     # [F, 33, 3]
        root_motion: np.ndarray,     # [F, 3]
        fps: float,
        skeleton: Optional[AnimationSkeleton] = None,
        source_video: str = "",
    ) -> AnimationData:
        if skeleton is None:
            skeleton = build_animation_skeleton()

        F = sequence_3d.shape[0]
        frames: List[AnimationFrame] = []
        prev: Optional[AnimationFrame] = None

        for f in range(F):
            ts = f / fps
            # Apply root motion offset
            pts = sequence_3d[f].copy()
            pts += root_motion[f]
            frame = self.retarget_frame(pts, f, ts, skeleton, prev_frame=prev)
            frames.append(frame)
            prev = frame

        logger.debug("Retargeted %d frames", F)
        return AnimationData(
            frames=frames,
            skeleton=skeleton,
            fps=fps,
            source_video=source_video,
            sequence_3d=sequence_3d.copy(),
            root_motion=root_motion.copy(),
        )
