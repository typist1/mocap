"""Skeleton hierarchy and quaternion math for animation."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Quaternion utilities  (all in xyzw convention)
# ---------------------------------------------------------------------------

def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two unit quaternions [x,y,z,w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=np.float64)


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """Inverse of a unit quaternion = conjugate."""
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions."""
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = float(np.dot(q1, q2))
    # Ensure shortest path
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        # Numerically close — use linear interpolation
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s1 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0
    return (s1 * q1 + s2 * q2)


def rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to unit quaternion [x,y,z,w]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [x,y,z,w] → 3x3 rotation matrix."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y)],
        [  2*(x*y + w*z), 1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [  2*(x*z - w*y),   2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def rotation_from_directions(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Return unit quaternion that rotates v_from to v_to."""
    v_from = v_from / (np.linalg.norm(v_from) + 1e-9)
    v_to = v_to / (np.linalg.norm(v_to) + 1e-9)
    dot = float(np.dot(v_from, v_to))
    if dot >= 1.0 - 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if dot <= -1.0 + 1e-9:
        # 180° rotation — find perpendicular axis
        axis = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(v_from, axis)) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = np.cross(v_from, axis)
        axis /= np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0])
    axis = np.cross(v_from, v_to)
    s = math.sqrt((1.0 + dot) * 2.0)
    q = np.array([axis[0]/s, axis[1]/s, axis[2]/s, s/2.0])
    return q / np.linalg.norm(q)


def euler_from_quat(q: np.ndarray, order: str = "ZXY") -> np.ndarray:
    """Convert unit quaternion to Euler angles (degrees) in given order."""
    R = quat_to_rotation_matrix(q)
    if order == "ZXY":
        # Standard BVH decomposition order
        sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0.0
        return np.array([math.degrees(x), math.degrees(y), math.degrees(z)])
    elif order == "XYZ":
        x = math.atan2(-R[1, 2], R[2, 2])
        y = math.asin(max(-1, min(1, R[0, 2])))
        z = math.atan2(-R[0, 1], R[0, 0])
        return np.array([math.degrees(x), math.degrees(y), math.degrees(z)])
    raise ValueError(f"Unsupported Euler order: {order}")


# ---------------------------------------------------------------------------
# Animation skeleton hierarchy
# ---------------------------------------------------------------------------

@dataclass
class JointRotation:
    local_rotation: np.ndarray   # [4] quaternion xyzw
    global_rotation: np.ndarray  # [4]


@dataclass
class AnimationFrame:
    frame_index: int
    timestamp: float
    root_position: np.ndarray    # [3]
    joint_rotations: Dict[str, np.ndarray]   # name → [4] xyzw quaternion


@dataclass
class AnimationSkeleton:
    joints: List[str]                            # in hierarchy order
    parent_map: Dict[str, Optional[str]]         # joint → parent name or None
    rest_positions: Dict[str, np.ndarray]        # joint → [3] local offset
    rest_lengths: Dict[str, float]

    @property
    def root_joint(self) -> str:
        for name, parent in self.parent_map.items():
            if parent is None:
                return name
        return self.joints[0]

    def children_of(self, name: str) -> List[str]:
        return [j for j, p in self.parent_map.items() if p == name]


@dataclass
class AnimationData:
    frames: List[AnimationFrame]
    skeleton: AnimationSkeleton
    fps: float
    source_video: str = ""
    # Raw per-frame landmark positions [F, 33, 3] in Blender-space meters.
    # Required for backward-compatible JSON export (blender_apply.py format).
    sequence_3d: Optional[np.ndarray] = None
    # Root motion trajectory [F, 3] separate from body-centered positions.
    root_motion: Optional[np.ndarray] = None
    # 2D image landmarks [F, 33, 3]: normalized x,y + visibility score.
    # Required for skeleton overlay video generation.
    landmarks_image: Optional[np.ndarray] = None


# Default bone lengths for a 1.70m person (in meters)
_DEFAULT_LENGTHS = {
    "Hips":         0.0,
    "Spine":        0.15,
    "Spine1":       0.12,
    "Spine2":       0.12,
    "Neck":         0.10,
    "Head":         0.15,
    "LeftShoulder": 0.12,
    "LeftArm":      0.28,
    "LeftForeArm":  0.25,
    "LeftHand":     0.08,
    "RightShoulder":0.12,
    "RightArm":     0.28,
    "RightForeArm": 0.25,
    "RightHand":    0.08,
    "LeftUpLeg":    0.42,
    "LeftLeg":      0.40,
    "LeftFoot":     0.17,
    "LeftToeBase":  0.08,
    "RightUpLeg":   0.42,
    "RightLeg":     0.40,
    "RightFoot":    0.17,
    "RightToeBase": 0.08,
}

_BVH_HIERARCHY = [
    "Hips",
    "Spine", "Spine1", "Spine2",
    "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
]

_PARENT_MAP: Dict[str, Optional[str]] = {
    "Hips":          None,
    "Spine":         "Hips",
    "Spine1":        "Spine",
    "Spine2":        "Spine1",
    "Neck":          "Spine2",
    "Head":          "Neck",
    "LeftShoulder":  "Spine2",
    "LeftArm":       "LeftShoulder",
    "LeftForeArm":   "LeftArm",
    "LeftHand":      "LeftForeArm",
    "RightShoulder": "Spine2",
    "RightArm":      "RightShoulder",
    "RightForeArm":  "RightArm",
    "RightHand":     "RightForeArm",
    "LeftUpLeg":     "Hips",
    "LeftLeg":       "LeftUpLeg",
    "LeftFoot":      "LeftLeg",
    "LeftToeBase":   "LeftFoot",
    "RightUpLeg":    "Hips",
    "RightLeg":      "RightUpLeg",
    "RightFoot":     "RightLeg",
    "RightToeBase":  "RightFoot",
}


def build_animation_skeleton(body_proportions: Optional[dict] = None) -> AnimationSkeleton:
    lengths = dict(_DEFAULT_LENGTHS)
    if body_proportions:
        # Map MediaPipe bone names to BVH names where possible
        mapping = {
            "left_hip_left_knee": "LeftUpLeg",
            "left_knee_left_ankle": "LeftLeg",
            "right_hip_right_knee": "RightUpLeg",
            "right_knee_right_ankle": "RightLeg",
            "left_shoulder_left_elbow": "LeftArm",
            "left_elbow_left_wrist": "LeftForeArm",
            "right_shoulder_right_elbow": "RightArm",
            "right_elbow_right_wrist": "RightForeArm",
        }
        for mp_name, bvh_name in mapping.items():
            if mp_name in body_proportions:
                lengths[bvh_name] = body_proportions[mp_name]

    # Compute local rest offsets (Y-up, bones along +Y)
    rest_positions: Dict[str, np.ndarray] = {}
    for joint in _BVH_HIERARCHY:
        parent = _PARENT_MAP[joint]
        if parent is None:
            rest_positions[joint] = np.zeros(3)
        else:
            # Simple: each bone points along +Y by its length
            L = lengths.get(joint, 0.1)
            rest_positions[joint] = np.array([0.0, L, 0.0])

    # Adjust lateral offsets for shoulders/legs
    rest_positions["LeftShoulder"]  = np.array([-0.12, lengths.get("Spine2", 0.12), 0.0])
    rest_positions["RightShoulder"] = np.array([ 0.12, lengths.get("Spine2", 0.12), 0.0])
    rest_positions["LeftUpLeg"]     = np.array([-0.09, 0.0, 0.0])
    rest_positions["RightUpLeg"]    = np.array([ 0.09, 0.0, 0.0])

    return AnimationSkeleton(
        joints=_BVH_HIERARCHY,
        parent_map=dict(_PARENT_MAP),
        rest_positions=rest_positions,
        rest_lengths=lengths,
    )
